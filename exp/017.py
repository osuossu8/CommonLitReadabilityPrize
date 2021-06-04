import gc
import os
import math
import random
import time
import warnings
import sys
sys.path.append("/root/workspace/CommonLitReadabilityPrize")

import numpy as np
import pandas as pd
import transformers
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as torchdata

from pathlib import Path
from typing import List

from sklearn import model_selection
from sklearn import metrics

from tqdm import tqdm
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer

from apex import amp


class CFG:
    ######################
    # Globals #
    ######################
    EXP_ID = '017'
    seed = 71
    epochs = 10
    folds = [0, 1, 2, 3, 4]
    N_FOLDS = 5
    LR = 5e-5
    max_len = 220 # 260 # 220 # 256
    train_bs = 8
    valid_bs = 16


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_logger(log_file='train.log'):
    from logging import getLogger, INFO, FileHandler,  Formatter,  StreamHandler
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


def calc_loss(y_true, y_pred):
    return  np.sqrt(metrics.mean_squared_error(y_true, y_pred))


def convert_examples_to_head_and_tail_features(data, tokenizer, max_len):
    head_len = int(max_len//2)
    tail_len = head_len - 1
    
    data = data.replace('\n', '')
    len_tok = len(tokenizer.tokenize(data))
    
    tok = tokenizer.encode_plus(
        data, 
        max_length=max_len, 
        truncation=True,
        return_attention_mask=True,
        return_token_type_ids=True
    )
    curr_sent = {}
    if len_tok > max_len:
        head_ids = tok['input_ids'][:head_len]
        tail_ids = tok['input_ids'][-tail_len:]
        head_mask = tok['attention_mask'][:head_len]
        tail_mask = tok['attention_mask'][-tail_len:]
        curr_sent['input_ids'] = head_ids + [1] + tail_ids
        curr_sent['attention_mask'] = head_mask + [0] + tail_mask
    else:
        padding_length = max_len - len(tok['input_ids'])
        curr_sent['input_ids'] = tok['input_ids'] + ([1] * padding_length)
        curr_sent['attention_mask'] = tok['attention_mask'] + ([0] * padding_length)
    return curr_sent


class CommonLitDataset:
    def __init__(self, df, excerpt, tokenizer, max_len):
        self.excerpt = excerpt
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.df = df

    def __len__(self):
        return len(self.excerpt)

    def __getitem__(self, item):
        text = str(self.excerpt[item])

        inputs = convert_examples_to_head_and_tail_features(text, self.tokenizer, self.max_len)

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        targets = self.df["target"].values[item]

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "targets" : torch.tensor(targets, dtype=torch.float32),
        }


class AttentionBlock(nn.Module):
  def __init__(self, in_features, middle_features, out_features):
    super().__init__()
    self.in_features = in_features
    self.middle_features = middle_features
    self.out_features = out_features

    self.W = nn.Linear(in_features, middle_features)
    self.V = nn.Linear(middle_features, out_features)

  def forward(self, features):
    att = torch.tanh(self.W(features))

    score = self.V(att)

    attention_weights = torch.softmax(score, dim=1)

    context_vector = attention_weights * features
    context_vector = torch.sum(context_vector, dim=1)

    return context_vector


class SpatialDropout(nn.Dropout2d):
    def forward(self, x):
        x = x.unsqueeze(2)    # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x


class RoBERTaLarge(nn.Module):
    def __init__(self, model_path):
        super(RoBERTaLarge, self).__init__()
        self.in_features = 1024
        self.embedding_dropout = SpatialDropout(0.2) # 2.5 # 0.3
        self.dropout = nn.Dropout(0.3) # 2.0 # 2.5 # 0.3
        self.roberta = RobertaModel.from_pretrained(model_path)
        self.activation = nn.Tanh()
        self.l0 = nn.Linear(self.in_features, 1)

    def forward(self, ids, mask):
        roberta_outputs = self.roberta(
            ids,
            attention_mask=mask
        )
        
        hidden_state = roberta_outputs.last_hidden_state
        hidden_state = self.embedding_dropout(hidden_state)[:, -4:, :]      
        last_4_hidden = torch.mean(hidden_state, 1)

        x = self.activation(last_4_hidden)
        logits = self.l0(self.dropout(x))
        return logits.squeeze(-1)


# ====================================================
# Training helper functions
# ====================================================
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MetricMeter(object):
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.y_true = []
        self.y_pred = []
    
    def update(self, y_true, y_pred):
        self.y_true.extend(y_true.cpu().detach().numpy().tolist())
        self.y_pred.extend(y_pred.cpu().detach().numpy().tolist())

    @property
    def avg(self):
        self.rmse = calc_loss(self.y_true, self.y_pred)
       
        return {
            "RMSE" : self.rmse,
        }


class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss,self).__init__()

    def forward(self,x,y):
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(x, y))
        return loss


def loss_fn(logits, targets):
    loss_fct = RMSELoss()
    loss = loss_fct(logits, targets)
    return loss
        
        
def train_fn(model, data_loader, device, optimizer, scheduler):
    model.train()
    losses = AverageMeter()
    scores = MetricMeter()
    tk0 = tqdm(data_loader, total=len(data_loader))
    
    for data in tk0:
        optimizer.zero_grad()
        inputs = data['input_ids'].to(device)
        masks = data['attention_mask'].to(device)
        targets = data['targets'].to(device)
        outputs = model(inputs, masks)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.update(loss.item(), inputs.size(0))
        scores.update(targets, outputs)
        tk0.set_postfix(loss=losses.avg)
    return scores.avg, losses.avg


def valid_fn(model, data_loader, device):
    model.eval()
    losses = AverageMeter()
    scores = MetricMeter()
    tk0 = tqdm(data_loader, total=len(data_loader))

    with torch.no_grad():
        for data in tk0:
            inputs = data['input_ids'].to(device)
            masks = data['attention_mask'].to(device)
            targets = data['targets'].to(device)
            outputs = model(inputs, masks)
            loss = loss_fn(outputs, targets)
            losses.update(loss.item(), inputs.size(0))
            scores.update(targets, outputs)
            tk0.set_postfix(loss=losses.avg)
    return scores.avg, losses.avg


def calc_cv(model_paths):
    model_name = 'roberta-large'
    models = []
    for model_path in model_paths:
        model = RoBERTaLarge(model_name)
        model.to("cuda")
        model.load_state_dict(torch.load(model_path))
        model.eval()
        models.append(model)
    
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    
    df = pd.read_csv("inputs/train_folds.csv")
    y_true = []
    y_pred = []
    for fold, model in enumerate(models):
        val_df = df[df.kfold == fold].reset_index(drop=True)
    
        dataset = CommonLitDataset(df=val_df, excerpt=val_df.excerpt.values, tokenizer=tokenizer, max_len=CFG.max_len)
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=CFG.valid_bs, num_workers=0, pin_memory=True, shuffle=False
        )

        final_output = []
        for b_idx, data in tqdm(enumerate(data_loader)):
            with torch.no_grad():
                inputs = data['input_ids'].to(device)
                masks = data['attention_mask'].to(device)

                output = model(inputs, masks)
                output = output.detach().cpu().numpy().tolist()
                final_output.extend(output)
        logger.info(calc_loss(np.array(final_output), val_df['target'].values))
        y_pred.append(np.array(final_output))
        y_true.append(val_df['target'].values)
        torch.cuda.empty_cache()
        
    y_pred = np.concatenate(y_pred)
    y_true = np.concatenate(y_true)
    overall_cv_score = calc_loss(y_true, y_pred)
    print(overall_cv_score)
    return overall_cv_score
 
       
OUTPUT_DIR = f'outputs/{CFG.EXP_ID}/'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

warnings.filterwarnings("ignore")
logger = init_logger(log_file=Path("logs") / f"{CFG.EXP_ID}.log")

# environment
set_seed(CFG.seed)
device = get_device()

# data
train = pd.read_csv("inputs/train_folds.csv")

print(train.shape)
train.head()

# main loop
for fold in range(5):
    if fold not in CFG.folds:
        continue
    logger.info("=" * 120)
    logger.info(f"Fold {fold} Training")
    logger.info("=" * 120)

    trn_df = train[train.kfold != fold].reset_index(drop=True)
    val_df = train[train.kfold == fold].reset_index(drop=True)
    
    model_path = 'roberta-large'
    model = RoBERTaLarge(model_path)
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    
    train_dataset = CommonLitDataset(df=trn_df, excerpt=trn_df.excerpt.values, tokenizer=tokenizer, max_len=CFG.max_len)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=CFG.train_bs, num_workers=0, pin_memory=True, shuffle=True
    )
    
    valid_dataset = CommonLitDataset(df=val_df, excerpt=val_df.excerpt.values, tokenizer=tokenizer, max_len=CFG.max_len)
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=CFG.valid_bs, num_workers=0, pin_memory=True, shuffle=False
    )
    
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]

    num_train_steps = int(len(trn_df) / CFG.train_bs * CFG.epochs)   
    # optimizer = torch.optim.Adam(model.parameters(), lr=CFG.LR)
    optimizer = transformers.AdamW(optimizer_parameters, lr=CFG.LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=1e-5, T_max=CFG.epochs)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=CFG.epochs, T_mult=1)

    model = model.to(device)

    p = 0
    patience = 3
    min_loss = 999
    best_score = np.inf

    for epoch in range(CFG.epochs):

        logger.info("Starting {} epoch...".format(epoch+1))

        start_time = time.time()

        train_avg, train_loss = train_fn(model, train_dataloader, device, optimizer, scheduler)

        valid_avg, valid_loss = valid_fn(model, valid_dataloader, device)
        scheduler.step()
        
        elapsed = time.time() - start_time
        
        logger.info(f'Epoch {epoch+1} - avg_train_loss: {train_loss:.5f}  avg_val_loss: {valid_loss:.5f}  time: {elapsed:.0f}s')
        logger.info(f"Epoch {epoch+1} - train_rmse:{train_avg['RMSE']:0.5f}  valid_rmse:{valid_avg['RMSE']:0.5f}")

        if valid_avg['RMSE'] < best_score:
            logger.info(f">>>>>>>> Model Improved From {best_score} ----> {valid_avg['RMSE']}")
            torch.save(model.state_dict(), OUTPUT_DIR+f'fold-{fold}.bin')
            best_score = valid_avg['RMSE']
            p = 0
        if p > 0: 
            logger.info(f'best score is not updated while {p} epochs of training')
        p += 1
        if p > patience:
            logger.info(f'Early Stopping')
            break


if len(CFG.folds) == 1:
    pass
else:
    model_paths = [
        f'outputs/{CFG.EXP_ID}/fold-0.bin', 
        f'outputs/{CFG.EXP_ID}/fold-1.bin', 
        f'outputs/{CFG.EXP_ID}/fold-2.bin', 
        f'outputs/{CFG.EXP_ID}/fold-3.bin',
        f'outputs/{CFG.EXP_ID}/fold-4.bin',
    ]

    overall_cv_score = calc_cv(model_paths)
    logger.info(f'cv score {overall_cv_score}')
    print()

