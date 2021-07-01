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
    EXP_ID = '096'
    seed = 71
    epochs = 5
    folds = [0, 1, 2, 3, 4]
    N_FOLDS = 5
    LR = 2e-5
    max_len = 256
    train_bs = 8 * 2
    valid_bs = 16 * 2
    log_interval = 10
    model_name = 'distilbert-base-cased-distilled-squad'
    tokenizer_name = 'roberta-base'
    itpt_path = None # 'itpt/roberta_large_2/' 
    numerical_cols = [
       'excerpt_num_chars', 'excerpt_num_capitals', 'excerpt_caps_vs_length',
       'excerpt_num_exclamation_marks', 'excerpt_num_question_marks',
       'excerpt_num_punctuation', 'excerpt_num_symbols', 'excerpt_num_words',
       'excerpt_num_unique_words', 'excerpt_words_vs_unique'
    ]
 

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
        curr_sent['input_ids'] = head_ids + tail_ids
        curr_sent['attention_mask'] = head_mask + tail_mask
    else:
        padding_length = max_len - len(tok['input_ids'])
        curr_sent['input_ids'] = tok['input_ids'] + ([1] * padding_length)
        curr_sent['attention_mask'] = tok['attention_mask'] + ([0] * padding_length)
    return curr_sent


class CommonLitDataset:
    def __init__(self, df, excerpt, tokenizer, max_len, numerical_features, tfidf):
        self.excerpt = excerpt
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.df = df
        self.numerical_features = numerical_features
        self.tfidf_df = tfidf

    def __len__(self):
        return len(self.excerpt)

    def __getitem__(self, item):
        text = str(self.excerpt[item])
        inputs = self.tokenizer(
            text, 
            max_length=self.max_len, 
            padding="max_length", 
            truncation=True
        )

        # inputs = convert_examples_to_head_and_tail_features(text, tokenizer, self.max_len)

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        targets = self.df["target"].values[item]
        aux = self.df["aux_target"].values[item] + 4

        aux_targets = np.zeros(7, dtype=float)
        aux_targets[aux] = 1.0

        numerical_features = self.numerical_features[item]
        tfidf = self.tfidf_df.values[item]

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "targets" : torch.tensor(targets, dtype=torch.float32),
            "aux_targets" : torch.tensor(aux_targets, dtype=torch.float32),
            "numerical_features" : torch.tensor(numerical_features, dtype=torch.float32),
            "tfidf" : torch.tensor(tfidf, dtype=torch.float32),
        }


class AttentionHead(nn.Module):
    def __init__(self, in_features, hidden_dim, num_targets):
        super().__init__()
        self.in_features = in_features
        self.middle_features = hidden_dim
        self.W = nn.Linear(in_features, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)
        self.out_features = hidden_dim

    def forward(self, features):
        att = torch.tanh(self.W(features))
        score = self.V(att)
        attention_weights = torch.softmax(score, dim=1)
        context_vector = attention_weights * features
        context_vector = torch.sum(context_vector, dim=1)

        return context_vector


class RoBERTaLarge(nn.Module):
    def __init__(self, model_path):
        super(RoBERTaLarge, self).__init__()
        self.in_features = 768 # 1024
        self.roberta = RobertaModel.from_pretrained(model_path)
        self.head = AttentionHead(self.in_features,self.in_features,1)
        self.dropout = nn.Dropout(0.1)
        self.process_num = nn.Sequential(
            nn.Linear(10, 8),
            nn.BatchNorm1d(8),
            nn.PReLU(),
            nn.Dropout(0.1),
        )
        self.process_tfidf = nn.Sequential(
            nn.Linear(100, 32),
            nn.BatchNorm1d(32),
            nn.PReLU(),
            nn.Dropout(0.1),
        )
        self.l0 = nn.Linear(self.in_features + 8 + 32, 1)
        self.l1 = nn.Linear(self.in_features + 8 + 32, 7)

    def forward(self, ids, mask, numerical_features, tfidf):
        roberta_outputs = self.roberta(
            ids,
            attention_mask=mask
        )

        x1 = self.head(roberta_outputs[0]) # bs, 1024

        x2 = self.process_num(numerical_features) # bs, 8

        x3 = self.process_tfidf(tfidf) # bs, 32

        x = torch.cat([x1, x2, x3], 1) # bs, 1024 + 8 + 32

        logits = self.l0(self.dropout(x))
        aux_logits = torch.sigmoid(self.l1(self.dropout(x)))
        return logits.squeeze(-1), aux_logits


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
    # loss_fct = RMSELoss()
    loss_fct = nn.MSELoss()
    loss = loss_fct(logits, targets)
    return loss

def aux_loss_fn(logits, targets):
    loss_fct = nn.BCEWithLogitsLoss()
    loss = loss_fct(logits, targets)
    return loss
        
        
def train_fn(epoch, model, train_data_loader, valid_data_loader, device, optimizer, scheduler, best_score):
    model.train()
    losses = AverageMeter()
    scores = MetricMeter()
    tk0 = tqdm(train_data_loader, total=len(train_data_loader))
    
    for batch_idx, data in enumerate(tk0):
        optimizer.zero_grad()
        inputs = data['input_ids'].to(device)
        masks = data['attention_mask'].to(device)
        targets = data['targets'].to(device)
        aux_targets = data['aux_targets'].to(device)
        numerical_features = data['numerical_features'].to(device)
        tfidf = data['tfidf'].to(device)
        outputs, aux_outs = model(inputs, masks, numerical_features, tfidf)
        loss = loss_fn(outputs, targets) * 0.5 + aux_loss_fn(aux_outs, aux_targets) * 0.5
        loss.backward()
        optimizer.step()
        # scheduler.step()
        losses.update(loss.item(), inputs.size(0))
        scores.update(targets, outputs)
        tk0.set_postfix(loss=losses.avg)

        if (batch_idx > 0) and (batch_idx % CFG.log_interval == 0):
            valid_avg, valid_loss = valid_fn(model, valid_data_loader, device)

            logger.info(f"Epoch {epoch+1}, Step {batch_idx} - valid_rmse:{valid_avg['RMSE']:0.5f}")

            if valid_avg['RMSE'] < best_score:
                logger.info(f">>>>>>>> Model Improved From {best_score} ----> {valid_avg['RMSE']}")
                torch.save(model.state_dict(), OUTPUT_DIR+f'fold-{fold}.bin')
                best_score = valid_avg['RMSE']

            # RuntimeError: cudnn RNN backward can only be called in training mode (_cudnn_rnn_backward_input at /pytorch/aten/src/ATen/native/cudnn/RNN.cpp:877)
            # https://discuss.pytorch.org/t/pytorch-cudnn-rnn-backward-can-only-be-called-in-training-mode/80080/2
            # edge case in my code when doing eval on training step
            # model.train() 

    return scores.avg, losses.avg, valid_avg, valid_loss, best_score


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
            aux_targets = data['aux_targets'].to(device)
            numerical_features = data['numerical_features'].to(device)
            tfidf = data['tfidf'].to(device)
            outputs, aux_outs = model(inputs, masks, numerical_features, tfidf)
            loss = loss_fn(outputs, targets) * 0.5 + aux_loss_fn(aux_outs, aux_targets) * 0.5
            losses.update(loss.item(), inputs.size(0))
            scores.update(targets, outputs)
            tk0.set_postfix(loss=losses.avg)
    return scores.avg, losses.avg


def calc_cv(model_paths):
    models = []
    for p in model_paths:
        if CFG.itpt_path:
            model = RoBERTaLarge(CFG.itpt_path)
            logger.info('load itpt model')
        else:
            model = RoBERTaLarge(CFG.model_name)
        model.to("cuda")
        model.load_state_dict(torch.load(p))
        model.eval()
        models.append(model)
    
    # tokenizer = RobertaTokenizer.from_pretrained(CFG.model_name)
    tokenizer = RobertaTokenizer.from_pretrained(CFG.tokenizer_name)

    df = pd.read_csv("inputs/train_folds.csv")
    df['aux_target'] = np.round(df['target'], 0).astype(np.int8) # 7 classes
    df = get_sentence_features(df, 'excerpt')

    TP = TextPreprocessor()
    preprocessed_text = TP.preprocess(df['excerpt'])

    pipeline = make_pipeline(
                TfidfVectorizer(max_features=100000),
                make_union(
                    TruncatedSVD(n_components=50, random_state=42),
                    make_pipeline(
                        BM25Transformer(use_idf=True, k1=2.0, b=0.75),
                        TruncatedSVD(n_components=50, random_state=42)
                    ),
                    n_jobs=1,
                ),
             )

    z = pipeline.fit_transform(preprocessed_text)
    tfidf_df = pd.DataFrame(z, columns=[f'cleaned_excerpt_tf_idf_svd_{i}' for i in range(50*2)])

    y_true = []
    y_pred = []
    for fold, model in enumerate(models):
        val_df = df[df.kfold == fold].reset_index(drop=True)
    
        dataset = CommonLitDataset(df=val_df, excerpt=val_df.excerpt.values, tokenizer=tokenizer, max_len=CFG.max_len, numerical_features=df[CFG.numerical_cols].values, tfidf=tfidf_df)
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=CFG.valid_bs, num_workers=0, pin_memory=True, shuffle=False
        )

        final_output = []
        for b_idx, data in tqdm(enumerate(data_loader)):
            with torch.no_grad():
                inputs = data['input_ids'].to(device)
                masks = data['attention_mask'].to(device)
                numerical_features = data['numerical_features'].to(device)
                tfidf = data['tfidf'].to(device)
                output, _ = model(inputs, masks, numerical_features, tfidf)
                output = output.detach().cpu().numpy().tolist()
                final_output.extend(output)
        logger.info(calc_loss(np.array(final_output), val_df['target'].values))
        y_pred.append(np.array(final_output))
        y_true.append(val_df['target'].values)
        torch.cuda.empty_cache()
        
    y_pred = np.concatenate(y_pred)
    y_true = np.concatenate(y_true)
    overall_cv_score = calc_loss(y_true, y_pred)
    logger.info(f'cv score {overall_cv_score}')
    return overall_cv_score


import nltk
import re
import scipy as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.feature_extraction.text import _document_frequency
from sklearn.pipeline import make_pipeline, make_union
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


class BM25Transformer(BaseEstimator, TransformerMixin):
    def __init__(self, use_idf=True, k1=2.0, b=0.75):
        self.use_idf = use_idf
        self.k1 = k1
        self.b = b

    def fit(self, X):
        if not sp.sparse.issparse(X):
            X = sp.sparse.csc_matrix(X)
        if self.use_idf:
            n_samples, n_features = X.shape
            df = _document_frequency(X)
            idf = np.log((n_samples - df + 0.5) / (df + 0.5))
            self._idf_diag = sp.sparse.spdiags(idf, diags=0, m=n_features, n=n_features)

        doc_len = X.sum(axis=1)
        self._average_document_len = np.average(doc_len)

        return self

    def transform(self, X, copy=True):
        if hasattr(X, 'dtype') and np.issubdtype(X.dtype, np.float):
            X = sp.sparse.csr_matrix(X, copy=copy)
        else:
            X = sp.sparse.csr_matrix(X, dtype=np.float, copy=copy)

        n_samples, n_features = X.shape
        doc_len = X.sum(axis=1)
        sz = X.indptr[1:] - X.indptr[0:-1]
        rep = np.repeat(np.asarray(doc_len), sz)

        nom = self.k1 + 1
        denom = X.data + self.k1 * (1 - self.b + self.b * rep / self._average_document_len)
        data = X.data * nom / denom

        X = sp.sparse.csr_matrix((data, X.indices, X.indptr), shape=X.shape)

        if self.use_idf:
            check_is_fitted(self, '_idf_diag', 'idf vector is not fitted')

            expected_n_features = self._idf_diag.shape[0]
            if n_features != expected_n_features:
                raise ValueError("Input has n_features=%d while the model"
                                 " has been trained with n_features=%d" % (
                                     n_features, expected_n_features))
            X = X * self._idf_diag

        return X 


class TextPreprocessor(object):
    def __init__(self):
        self.puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£',
                       '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…',
                       '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─',
                       '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', '«',
                       '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', '（', '）', '～',
                       '➡', '％', '⇒', '▶', '「', '➄', '➆',  '➊', '➋', '➌', '➍', '⓪', '①', '②', '③', '④', '⑤', '⑰', '❶', '❷', '❸', '❹', '❺', '❻', '❼', '❽',  
                       '＝', '※', '㈱', '､', '△', '℮', 'ⅼ', '‐', '｣', '┝', '↳', '◉', '／', '＋', '○',
                       '【', '】', '✅', '☑', '➤', 'ﾞ', '↳', '〶', '☛', '｢', '⁺', '『', '≫',
                       ]

        self.numbers = ["0","1","2","3","4","5","6","7","8","9","０","１","２","３","４","５","６","７","８","９"]
        self.stopwords = nltk.corpus.stopwords.words('english')

    def _pre_preprocess(self, x):
        return str(x).lower() 

    def rm_num(self, x, use_num=True):
        x = re.sub('[0-9]{5,}', '', x)
        x = re.sub('[0-9]{4}', '', x)
        x = re.sub('[0-9]{3}', '', x)
        x = re.sub('[0-9]{2}', '', x)    
        for i in self.numbers:
            x = x.replace(str(i), '')        
        return x

    def clean_puncts(self, x):
        for punct in self.puncts:
            x = x.replace(punct, '')
        return x
    
    def clean_stopwords(self, x):
        list_x = x.split()
        res = []
        for w in list_x:
            if w not in self.stopwords:
                res.append(w)
        return ' '.join(res)

    def preprocess(self, sentence):
        sentence = sentence.fillna(" ")
        sentence = sentence.map(lambda x: self._pre_preprocess(x))
        sentence = sentence.map(lambda x: self.clean_puncts(x))
        sentence = sentence.map(lambda x: self.clean_stopwords(x))
        sentence = sentence.map(lambda x: self.rm_num(x))
        return sentence


def get_sentence_features(train, col):
    train[col + '_num_chars'] = train[col].apply(len)
    train[col + '_num_capitals'] = train[col].apply(lambda x: sum(1 for c in x if c.isupper()))
    train[col + '_caps_vs_length'] = train.apply(lambda row: row[col + '_num_chars'] / (row[col + '_num_capitals']+1e-5), axis=1)
    train[col + '_num_exclamation_marks'] = train[col].apply(lambda x: x.count('!'))
    train[col + '_num_question_marks'] = train[col].apply(lambda x: x.count('?'))
    train[col + '_num_punctuation'] = train[col].apply(lambda x: sum(x.count(w) for w in '.,;:'))
    train[col + '_num_symbols'] = train[col].apply(lambda x: sum(x.count(w) for w in '*&$%'))
    train[col + '_num_words'] = train[col].apply(lambda x: len(x.split()))
    train[col + '_num_unique_words'] = train[col].apply(lambda comment: len(set(w for w in comment.split())))
    train[col + '_words_vs_unique'] = train[col + '_num_unique_words'] / train[col + '_num_words'] 
    return train


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
train['aux_target'] = np.round(train['target'], 0).astype(np.int8) # 7 classes

train = get_sentence_features(train, 'excerpt')

TP = TextPreprocessor()
preprocessed_text = TP.preprocess(train['excerpt'])

pipeline = make_pipeline(
                TfidfVectorizer(max_features=100000),
                make_union(
                    TruncatedSVD(n_components=50, random_state=42),
                    make_pipeline(
                        BM25Transformer(use_idf=True, k1=2.0, b=0.75),
                        TruncatedSVD(n_components=50, random_state=42)
                    ),
                    n_jobs=1,
                ),
             )

z = pipeline.fit_transform(preprocessed_text)
tfidf_df = pd.DataFrame(z, columns=[f'cleaned_excerpt_tf_idf_svd_{i}' for i in range(50*2)])

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

    if CFG.itpt_path:
        model = RoBERTaLarge(CFG.itpt_path)
        logger.info('load itpt model')
    else:
        model = RoBERTaLarge(CFG.model_name)    

    # tokenizer = RobertaTokenizer.from_pretrained(CFG.model_name)
    tokenizer = RobertaTokenizer.from_pretrained(CFG.tokenizer_name)

    train_dataset = CommonLitDataset(df=trn_df, excerpt=trn_df.excerpt.values, tokenizer=tokenizer, max_len=CFG.max_len, numerical_features=trn_df[CFG.numerical_cols].values, tfidf=tfidf_df)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=CFG.train_bs, num_workers=0, pin_memory=True, shuffle=True
    )
    
    valid_dataset = CommonLitDataset(df=val_df, excerpt=val_df.excerpt.values, tokenizer=tokenizer, max_len=CFG.max_len, numerical_features=val_df[CFG.numerical_cols].values, tfidf=tfidf_df)
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=CFG.valid_bs, num_workers=0, pin_memory=True, shuffle=False
    )
    
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        # {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]

    num_train_steps = int(len(trn_df) / CFG.train_bs * CFG.epochs)   
    optimizer = transformers.AdamW(optimizer_parameters, lr=CFG.LR)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=1e-5, T_max=CFG.epochs)
    scheduler = None

    model = model.to(device)

    min_loss = 999
    best_score = np.inf

    for epoch in range(CFG.epochs):
        logger.info("Starting {} epoch...".format(epoch+1))

        start_time = time.time()

        train_avg, train_loss, valid_avg, valid_loss, best_score = train_fn(epoch, model, train_dataloader, valid_dataloader, device, optimizer, scheduler, best_score)

        # scheduler.step()
        
        elapsed = time.time() - start_time
        
        logger.info(f'Epoch {epoch+1} - avg_train_loss: {train_loss:.5f}  avg_val_loss: {valid_loss:.5f}  time: {elapsed:.0f}s')
        logger.info(f"Epoch {epoch+1} - train_rmse:{train_avg['RMSE']:0.5f}  valid_rmse:{valid_avg['RMSE']:0.5f}")

        if valid_avg['RMSE'] < best_score:
            logger.info(f">>>>>>>> Model Improved From {best_score} ----> {valid_avg['RMSE']}")
            torch.save(model.state_dict(), OUTPUT_DIR+f'fold-{fold}.bin')
            best_score = valid_avg['RMSE']


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
    print()

