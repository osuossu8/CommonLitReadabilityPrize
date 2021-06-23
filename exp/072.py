import copy
import gc
import itertools
import joblib
import math
import nltk
import os
import pickle
import random
import re
import string
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
from keras.preprocessing import text, sequence

from tqdm import tqdm
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer

from collections import Counter, defaultdict
from functools import partial
from pathlib import Path
from typing import Dict, List, Tuple

from gensim.models import Word2Vec
from numba import cuda
import plotly_express as px

from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.stem.lancaster import LancasterStemmer

from sklearn.model_selection import GroupKFold, KFold
from sklearn.utils import shuffle

import scipy as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.feature_extraction.text import _document_frequency
from sklearn.pipeline import make_pipeline, make_union
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from apex import amp


class CFG:
    ######################
    # Globals #
    ######################
    EXP_ID = '072'
    seed = 71
    epochs = 10
    folds = [0, 1, 2, 3, 4]
    N_FOLDS = 5
    LR = 1e-3
    max_len = 256
    train_bs = 8 * 2
    valid_bs = 16 * 2
    log_interval = 10
    model_name = 'roberta-large'
    itpt_path = 'itpt/roberta_large_2/' 
    numerical_cols = [
       'excerpt_num_chars', 'excerpt_num_capitals', 'excerpt_caps_vs_length',
       'excerpt_num_exclamation_marks', 'excerpt_num_question_marks',
       'excerpt_num_punctuation', 'excerpt_num_symbols', 'excerpt_num_words',
       'excerpt_num_unique_words', 'excerpt_words_vs_unique'
    ]
    EMBEDDING_PATH = 'embeddings/fasttext.pkl'
    max_features = 60000
    embed_size = 300
    USE_cols = [f'excerpt_use_{c}' for c in range(512)]

ps = PorterStemmer()
lc = LancasterStemmer()
sb = SnowballStemmer('english') 


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


def to_pickle(filename, obj):
    with open(filename, mode='wb') as f:
        pickle.dump(obj, f)


def unpickle(filename):
    with open(filename, mode='rb') as fo:
        p = pickle.load(fo)
    return p  


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
    def __init__(self, df, excerpt, tokenizer, max_len, numerical_features, tfidf, padded_tokens, use_features):
        self.excerpt = excerpt
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.df = df
        self.numerical_features = numerical_features
        self.tfidf_df = tfidf
        self.padded_tokens = padded_tokens
        self.use_features = use_features

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
        padded_tokens = self.padded_tokens[item]
        use_features = self.use_features[item]

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "targets" : torch.tensor(targets, dtype=torch.float32),
            "aux_targets" : torch.tensor(aux_targets, dtype=torch.float32),
            "numerical_features" : torch.tensor(numerical_features, dtype=torch.float32),
            "tfidf" : torch.tensor(tfidf, dtype=torch.float32),
            "padded_tokens": torch.tensor(padded_tokens, dtype=torch.long),
            "use_features" : torch.tensor(use_features, dtype=torch.float32),
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


# https://github.com/sakami0000/kaggle_jigsaw/blob/master/src/lstm_models/models.py
# https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/discussion/97471
class RoBERTaLarge(nn.Module):
    def __init__(self, model_path, embedding_matrix):
        super(RoBERTaLarge, self).__init__()
        self.in_features = 1024
        self.roberta = RobertaModel.from_pretrained(model_path)

        self.lstm_hidden_size = 128
        self.gru_hidden_size = 128
        self.embedding = nn.Embedding(*embedding_matrix.shape)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.embedding_dropout = nn.Dropout2d(0.2)
        self.lstm = nn.LSTM(embedding_matrix.shape[1], self.lstm_hidden_size, bidirectional=True, batch_first=True)
        self.gru = nn.GRU(self.lstm_hidden_size * 2, self.gru_hidden_size, bidirectional=True, batch_first=True)

        self.head = AttentionHead(self.lstm_hidden_size * 2, self.lstm_hidden_size * 2, 1)
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
        # self.linear = nn.Linear(self.in_features + 8 + 32 + 256 * 3 + 512, 512)
        # self.relu = nn.ReLU()
        self.l0 = nn.Linear(8 + 32 + 256 * 4 + 512, 1)
        self.l1 = nn.Linear(8 + 32 + 256 * 4 + 512, 7)

    def apply_spatial_dropout(self, h_embedding):
        h_embedding = h_embedding.transpose(1, 2).unsqueeze(2)
        h_embedding = self.embedding_dropout(h_embedding).squeeze(2).transpose(1, 2)
        return h_embedding

    def forward(self, ids, mask, numerical_features, tfidf, seqs, use_features):

        h_embedding = self.embedding(seqs)
        h_embedding = self.apply_spatial_dropout(h_embedding)

        h_lstm, _ = self.lstm(h_embedding)
        h_gru, hh_gru = self.gru(h_lstm)

        hh_gru = hh_gru.view(-1, self.gru_hidden_size * 2)

        avg_pool = torch.mean(h_gru, 1)
        max_pool, _ = torch.max(h_gru, 1)

        h_gru_attn = self.head(h_gru) # 256 * 2

        conc = torch.cat((hh_gru, avg_pool, max_pool, h_gru_attn), 1) # bs, 256 * 5

        x2 = self.process_num(numerical_features) # bs, 8

        x3 = self.process_tfidf(tfidf) # bs, 32

        x = torch.cat([x2, x3, conc, use_features], 1) # bs, 8 + 32 + 256 * 3 + 512

        # x = self.relu(self.linear(x)) # bs, 512
        # x = self.relu(x)

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
    loss_fct = RMSELoss()
    loss = loss_fct(logits, targets)
    return loss

def aux_loss_fn(logits, targets):
    loss_fct = nn.BCEWithLogitsLoss()
    loss = loss_fct(logits, targets)
    return loss
        
        
def train_fn(model, data_loader, device, optimizer, scheduler):
    model.train()
    losses = AverageMeter()
    scores = MetricMeter()
    tk0 = tqdm(data_loader, total=len(data_loader))
    
    for batch_idx, data in enumerate(tk0):
        optimizer.zero_grad()
        inputs = data['input_ids'].to(device)
        masks = data['attention_mask'].to(device)
        targets = data['targets'].to(device)
        aux_targets = data['aux_targets'].to(device)
        numerical_features = data['numerical_features'].to(device)
        tfidf = data['tfidf'].to(device)
        seqs = data['padded_tokens'].to(device)
        use_features = data['use_features'].to(device)
        outputs, aux_outs = model(inputs, masks, numerical_features, tfidf, seqs, use_features)
        loss = loss_fn(outputs, targets) * 0.5 + aux_loss_fn(aux_outs, aux_targets) * 0.5
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
            aux_targets = data['aux_targets'].to(device)
            numerical_features = data['numerical_features'].to(device)
            tfidf = data['tfidf'].to(device)
            seqs = data['padded_tokens'].to(device)
            use_features = data['use_features'].to(device)
            outputs, aux_outs = model(inputs, masks, numerical_features, tfidf, seqs, use_features)
            loss = loss_fn(outputs, targets) * 0.5 + aux_loss_fn(aux_outs, aux_targets) * 0.5
            losses.update(loss.item(), inputs.size(0))
            scores.update(targets, outputs)
            tk0.set_postfix(loss=losses.avg)
    return scores.avg, losses.avg


def calc_cv(model_paths):
    models = []
    for p in model_paths:
        if CFG.itpt_path:
            model = RoBERTaLarge(CFG.itpt_path, embedding_matrix)
            logger.info('load itpt model')
        else:
            model = RoBERTaLarge(CFG.model_name, embedding_matrix)
        model.to("cuda")
        model.load_state_dict(torch.load(p))
        model.eval()
        models.append(model)
    
    tokenizer = RobertaTokenizer.from_pretrained(CFG.model_name)
    
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

    USE_df = unpickle('inputs/excerpt_use_df.pkl')
    df = pd.merge(df, USE_df, on='id')

    y_true = []
    y_pred = []
    for fold, model in enumerate(models):
        val_df = df[df.kfold == fold].reset_index(drop=True)

        valid_tok = tokenize(val_df['excerpt'].values, vocab['token2id'], CFG.max_len)
        padded_tokens = sequence.pad_sequences(valid_tok, maxlen=CFG.max_len)
    
        dataset = CommonLitDataset(df=val_df, excerpt=val_df.excerpt.values, tokenizer=tokenizer, max_len=CFG.max_len, 
                                   numerical_features=df[CFG.numerical_cols].values, tfidf=tfidf_df, padded_tokens=padded_tokens,
                                   use_features=df[CFG.USE_cols].values)
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
                seqs = data['padded_tokens'].to(device)
                use_features = data['use_features'].to(device)
                output, _ = model(inputs, masks, numerical_features, tfidf, seqs, use_features)
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


misspell_dict = {"aren't": "are not", "can't": "cannot", "couldn't": "could not",
                 "didn't": "did not", "doesn't": "does not", "don't": "do not",
                 "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
                 "he'd": "he would", "he'll": "he will", "he's": "he is",
                 "i'd": "I had", "i'll": "I will", "i'm": "I am", "isn't": "is not",
                 "it's": "it is", "it'll": "it will", "i've": "I have", "let's": "let us",
                 "mightn't": "might not", "mustn't": "must not", "shan't": "shall not",
                 "she'd": "she would", "she'll": "she will", "she's": "she is",
                 "shouldn't": "should not", "that's": "that is", "there's": "there is",
                 "they'd": "they would", "they'll": "they will", "they're": "they are",
                 "they've": "they have", "we'd": "we would", "we're": "we are",
                 "weren't": "were not", "we've": "we have", "what'll": "what will",
                 "what're": "what are", "what's": "what is", "what've": "what have",
                 "where's": "where is", "who'd": "who would", "who'll": "who will",
                 "who're": "who are", "who's": "who is", "who've": "who have",
                 "won't": "will not", "wouldn't": "would not", "you'd": "you would",
                 "you'll": "you will", "you're": "you are", "you've": "you have",
                 "'re": " are", "wasn't": "was not", "we'll": " will", "tryin'": "trying"}


def replace_typical_misspell(text: str) -> str:
    misspell_re = re.compile('(%s)' % '|'.join(misspell_dict.keys()))

    def replace(match):
        return misspell_dict[match.group(0)]

    return misspell_re.sub(replace, text)


puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']',
          '>', '%', '=', '#', '*', '+', '\\', '•', '~', '@', '£', '·', '_', '{', '}', '©', '^',
          '®', '`', '<', '→', '°', '€', '™', '›', '♥', '←', '×', '§', '″', '′', 'Â', '█',
          '½', 'à', '…', '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶',
          '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', '▒', '：', '¼',
          '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲',
          'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', '∙', '）', '↓', '、', '│', '（', '»', '，', '♪',
          '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√']


def clean_text(text: str) -> str:
    text = str(text)
    for punct in puncts + list(string.punctuation):
        if punct in text:
            text = text.replace(punct, f' {punct} ')
    return text


def clean_numbers(text: str) -> str:
    return re.sub(r'\d+', ' ', text)


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = replace_typical_misspell(text)
    text = clean_text(text)
    text = clean_numbers(text)
    text = text.strip()
    return text


def build_vocab(texts: List[str], max_features: int = 100000) -> Dict[str, Dict]:
    counter = Counter()
    for text in texts:
        counter.update(text.split())

    vocab = {}
    vocab['token2id'] = {
        token: _id + 1 for _id, (token, count) in
        enumerate(counter.most_common(max_features))}
    vocab['token2id']['<PAD>'] = 0
    vocab['token2id']['<UNK>'] = len(vocab['token2id'])
    vocab['id2token'] = {v: k for k, v in vocab['token2id'].items()}
    vocab['word_freq'] = {
        **{'<PAD>': 0, '<UNK>': 0},
        **dict(counter.most_common(max_features)),
    }
    return vocab


def tokenize(texts: List[str],
             token2id: Dict[str, int],
             max_len: int = 200) -> List[List[int]]:
    
    def text2ids(text, token2id, max_len):
        return [
            token2id.get(token, len(token2id) - 1)
            for token in text.split()[:max_len]]
    
    tokenized = [
        text2ids(text, token2id, max_len)
        for text in texts]
    return tokenized


def load_embedding(embedding_path: str, word_index: Dict[str, int]) -> np.ndarray:
    embeddings_index = joblib.load(embedding_path)

    # word_index = tokenizer.word_index
    nb_words = min(CFG.max_features + 2, len(word_index))
    embedding_matrix = np.zeros((nb_words, CFG.embed_size))

    for key, i in word_index.items():
        word = key
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            continue
        word = key.lower()
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            continue
        word = key.upper()
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            continue
        word = key.capitalize()
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            continue
        word = ps.stem(key)
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            continue
        word = lc.stem(key)
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            continue
        word = sb.stem(key)
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            continue

    return embedding_matrix


def w2v_fine_tune(all_texts: List[str], vocab: Dict, embedding_matrix: np.ndarray) -> np.ndarray:
    model = Word2Vec(min_count=1, workers=1, epochs=3, vector_size=300)
    model.build_vocab_from_freq(vocab['word_freq'])
    idxmap = np.array(
        [vocab['token2id'][w] for w in model.wv.index_to_key])
    model.wv.vectors[:] = embedding_matrix[idxmap]
    # model.trainables.syn1neg[:] = embedding_matrix[idxmap]
    model.train(all_texts, total_examples=len(all_texts), epochs=model.epochs)
    embedding_matrix = np.vstack([np.zeros((1, 300)), model.wv.vectors, np.zeros((1, 300))])
    return embedding_matrix
 
       
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

USE_df = unpickle('inputs/excerpt_use_df.pkl')
train = pd.merge(train, USE_df, on='id')

print(train.shape)
train.head()


test = pd.read_csv("inputs/test.csv")
train_texts = train[['excerpt']].applymap(preprocess_text).values
test_texts = test[['excerpt']].applymap(preprocess_text).values
all_texts = list(itertools.chain(*train_texts, *test_texts))


vocab = build_vocab(itertools.chain(*train_texts, *test_texts), CFG.max_features)
# embedding_matrix = load_embedding(CFG.EMBEDDING_PATH, vocab['token2id'])
# embedding_matrix = w2v_fine_tune(all_texts, vocab, embedding_matrix)

# to_pickle('inputs/finetuned_embedding_matrix.pkl', embedding_matrix)

embedding_matrix = unpickle('inputs/finetuned_embedding_matrix.pkl')

print(embedding_matrix.shape)


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
        model = RoBERTaLarge(CFG.itpt_path, embedding_matrix)
        logger.info('load itpt model')
    else:
        model = RoBERTaLarge(CFG.model_name, embedding_matrix)    

    tokenizer = RobertaTokenizer.from_pretrained(CFG.model_name)

    train_tok = tokenize(trn_df['excerpt'].values, vocab['token2id'], CFG.max_len)
    train_padded_tokens = sequence.pad_sequences(train_tok, maxlen=CFG.max_len)

    valid_tok = tokenize(val_df['excerpt'].values, vocab['token2id'], CFG.max_len)
    valid_padded_tokens = sequence.pad_sequences(valid_tok, maxlen=CFG.max_len)
    
    train_dataset = CommonLitDataset(df=trn_df, excerpt=trn_df.excerpt.values, tokenizer=tokenizer, max_len=CFG.max_len, 
                                     numerical_features=trn_df[CFG.numerical_cols].values, tfidf=tfidf_df, padded_tokens=train_padded_tokens,
                                     use_features=trn_df[CFG.USE_cols].values)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=CFG.train_bs, num_workers=0, pin_memory=True, shuffle=True
    )
    
    valid_dataset = CommonLitDataset(df=val_df, excerpt=val_df.excerpt.values, tokenizer=tokenizer, max_len=CFG.max_len, 
                                     numerical_features=val_df[CFG.numerical_cols].values, tfidf=tfidf_df, padded_tokens=valid_padded_tokens,
                                     use_features=val_df[CFG.USE_cols].values)
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=CFG.valid_bs, num_workers=0, pin_memory=True, shuffle=False
    )
    
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]

    num_train_steps = int(len(trn_df) / CFG.train_bs * CFG.epochs)   
    optimizer = transformers.AdamW(optimizer_parameters, lr=CFG.LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=1e-5, T_max=CFG.epochs)

    model = model.to(device)

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

