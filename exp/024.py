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
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer, BertPreTrainedModel

from apex import amp


class CFG:
    ######################
    # Globals #
    ######################
    EXP_ID = '024'
    seed = 71
    epochs = 8 # 10
    folds = [0, 1, 2, 3, 4]
    N_FOLDS = 5
    LR = 5e-5
    max_len = 250
    train_bs = 16
    valid_bs = 32
    model_name = 'roberta-base'
    itpt_path = 'itpt/roberta_base/'


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


def convert_examples_to_features(data, tokenizer, max_len, is_test=False):
    data = data.replace('\n', '')
    tok = tokenizer.encode_plus(
        data, 
        max_length=max_len, 
        truncation=True,
        return_attention_mask=True,
        return_token_type_ids=True
    )
    curr_sent = {}
    padding_length = max_len - len(tok['input_ids'])
    curr_sent['input_ids'] = tok['input_ids'] + ([0] * padding_length)
    curr_sent['token_type_ids'] = tok['token_type_ids'] + \
        ([0] * padding_length)
    curr_sent['attention_mask'] = tok['attention_mask'] + \
        ([0] * padding_length)
    return curr_sent


def convert_examples_to_head_and_tail_features(data, tokenizer, max_len):
    head_len = int(max_len//2)
    tail_len = head_len
    
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
        head_type_ids = tok['token_type_ids'][:head_len]
        tail_type_ids = tok['token_type_ids'][-tail_len:]
        curr_sent['input_ids'] = head_ids + tail_ids
        curr_sent['token_type_ids'] = head_type_ids + tail_type_ids
        curr_sent['attention_mask'] = head_mask + tail_mask
    else:
        padding_length = max_len - len(tok['input_ids'])
        curr_sent['input_ids'] = tok['input_ids'] + ([1] * padding_length)
        curr_sent['token_type_ids'] = tok['token_type_ids'] + ([0] * padding_length)
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

        inputs = convert_examples_to_features(text, self.tokenizer, self.max_len)
        # inputs = self.tokenizer(
        #     text, 
        #     max_length=self.max_len, 
        #     padding="max_length", 
        #     truncation=True
        # )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]
        targets = self.df["target"].values[item]

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "targets" : torch.tensor(targets, dtype=torch.float32),
        }


class RoBERTaBase(nn.Module):
    def __init__(self, model_path, config):
        super(RoBERTaBase, self).__init__()
        self.config = config
        self.in_features = 768
        self.dropout = nn.Dropout(p=0.3)
        self.roberta = RobertaModel.from_pretrained(model_path, output_hidden_states=True)
        self.layer_norm = nn.LayerNorm(self.in_features)
        self.l0 = nn.Linear(self.in_features, 1)

        self._init_weights(self.layer_norm)
        self._init_weights(self.l0)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, ids, mask, token_type_ids):
        roberta_outputs = self.roberta(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids
        )

        sequence_output = roberta_outputs[1]
        sequence_output = self.layer_norm(sequence_output)

        logits = self.l0(self.dropout(sequence_output))

        # logits = logits + roberta_outputs[1][:, 0].reshape(-1, 1)

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
        token_type_ids = data["token_type_ids"].to(device)
        targets = data['targets'].to(device)
        outputs = model(inputs, masks, token_type_ids)
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
            token_type_ids = data["token_type_ids"].to(device)
            targets = data['targets'].to(device)
            outputs = model(inputs, masks, token_type_ids)
            loss = loss_fn(outputs, targets)
            losses.update(loss.item(), inputs.size(0))
            scores.update(targets, outputs)
            tk0.set_postfix(loss=losses.avg)
    return scores.avg, losses.avg


def calc_cv(model_paths):
    config = RobertaConfig.from_pretrained(CFG.model_name)
    models = []
    for model_path in model_paths:
        if CFG.itpt_path:
            model = RoBERTaBase(CFG.itpt_path, config)
            print('load itpt model')
        else:
            model = RoBERTaBase(CFG.model_name, config)
        model.to("cuda")
        model.load_state_dict(torch.load(model_path))
        model.eval()
        models.append(model)
    
    tokenizer = RobertaTokenizer.from_pretrained(CFG.model_name)

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
                token_type_ids = data['token_type_ids'].to(device)

                output = model(inputs, masks, token_type_ids)
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
 

def get_optimizer_params(model):
    # differential learning rate and weight decay
    param_optimizer = list(model.named_parameters())
    learning_rate = 5e-5
    no_decay = ['bias', 'gamma', 'beta']
    group1=['layer.0.','layer.1.','layer.2.','layer.3.']
    group2=['layer.4.','layer.5.','layer.6.','layer.7.']    
    group3=['layer.8.','layer.9.','layer.10.','layer.11.']
    group_all=['layer.0.','layer.1.','layer.2.','layer.3.','layer.4.','layer.5.','layer.6.','layer.7.','layer.8.','layer.9.','layer.10.','layer.11.']
    optimizer_parameters = [
        {'params': [p for n, p in model.roberta.named_parameters() if not any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all)],'weight_decay': 0.01},
        {'params': [p for n, p in model.roberta.named_parameters() if not any(nd in n for nd in no_decay) and any(nd in n for nd in group1)],'weight_decay': 0.01, 'lr': learning_rate/2.6},
        {'params': [p for n, p in model.roberta.named_parameters() if not any(nd in n for nd in no_decay) and any(nd in n for nd in group2)],'weight_decay': 0.01, 'lr': learning_rate},
        {'params': [p for n, p in model.roberta.named_parameters() if not any(nd in n for nd in no_decay) and any(nd in n for nd in group3)],'weight_decay': 0.01, 'lr': learning_rate*2.6},
        {'params': [p for n, p in model.roberta.named_parameters() if any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all)],'weight_decay': 0.0},
        {'params': [p for n, p in model.roberta.named_parameters() if any(nd in n for nd in no_decay) and any(nd in n for nd in group1)],'weight_decay': 0.0, 'lr': learning_rate/2.6},
        {'params': [p for n, p in model.roberta.named_parameters() if any(nd in n for nd in no_decay) and any(nd in n for nd in group2)],'weight_decay': 0.0, 'lr': learning_rate},
        {'params': [p for n, p in model.roberta.named_parameters() if any(nd in n for nd in no_decay) and any(nd in n for nd in group3)],'weight_decay': 0.0, 'lr': learning_rate*2.6},
        {'params': [p for n, p in model.named_parameters() if "roberta" not in n], 'lr':1e-3, "momentum" : 0.99},
    ]
    return optimizer_parameters

       
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
"""
# main loop
for fold in range(5):
    if fold not in CFG.folds:
        continue
    logger.info("=" * 120)
    logger.info(f"Fold {fold} Training")
    logger.info("=" * 120)

    trn_df = train[train.kfold != fold].reset_index(drop=True)
    val_df = train[train.kfold == fold].reset_index(drop=True)

    config = RobertaConfig.from_pretrained(CFG.model_name)
    if CFG.itpt_path:    
        model = RoBERTaBase(CFG.itpt_path, config)
        print('load itpt model')
    else:
        model = RoBERTaBase(CFG.model_name, config)

    tokenizer = RobertaTokenizer.from_pretrained(CFG.model_name)     
    
    train_dataset = CommonLitDataset(df=trn_df, excerpt=trn_df.excerpt.values, tokenizer=tokenizer, max_len=CFG.max_len)
    train_sampler = torchdata.RandomSampler(train_dataset)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=CFG.train_bs, num_workers=0, pin_memory=True, drop_last=False, sampler=train_sampler
    )
    
    valid_dataset = CommonLitDataset(df=val_df, excerpt=val_df.excerpt.values, tokenizer=tokenizer, max_len=CFG.max_len)
    valid_sampler = torchdata.SequentialSampler(valid_dataset)

    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=CFG.valid_bs, num_workers=0, pin_memory=True, drop_last=False, sampler=valid_sampler
    )
    
    # param_optimizer = list(model.named_parameters())
    # no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    # optimizer_parameters = [
    #     {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
    #     {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    # ]

    optimizer_parameters = get_optimizer_params(model)

    num_train_steps = int(len(trn_df) / CFG.train_bs * CFG.epochs)   
    optimizer = transformers.AdamW(optimizer_parameters, lr=CFG.LR, weight_decay=0.01)
    scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=num_train_steps
    )

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
        # if p > 0: 
        #     logger.info(f'best score is not updated while {p} epochs of training')
        # p += 1
        # if p > patience:
        #     logger.info(f'Early Stopping')
        #     break
"""

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

