import sys
sys.path.append("/root/workspace/CommonLitReadabilityPrize")
from sklearn import model_selection
from glob import glob
import os
import matplotlib.pyplot as plt
import json
from collections import defaultdict
import gc
gc.enable()
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.optimizer import Optimizer
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import (
    Dataset, DataLoader,
    SequentialSampler, RandomSampler
)
from transformers import AutoConfig
from transformers import (
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup
)
from transformers import AdamW
from transformers import AutoTokenizer
from transformers import AutoModel
from transformers import MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING
from IPython.display import clear_output
from tqdm import tqdm, trange


class CFG:
    ######################
    # Globals #
    ######################
    EXP_ID = '019'
    seed = 71
    epochs = 8 # 10
    folds = [0, 1, 2, 3, 4]
    N_FOLDS = 5
    LR = 5e-5
    max_len = 250 # 256 # 220
    train_bs = 16 # 8
    valid_bs = 32 # 16
    model_name = 'roberta-large'
    itpt_path = False # 'itpt/roberta_large/'


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


OUTPUT_DIR = f'outputs/{CFG.EXP_ID}/'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

warnings.filterwarnings("ignore")
logger = init_logger(log_file=Path("logs") / f"{CFG.EXP_ID}.log")

# environment
set_seed(CFG.seed)
device = get_device()

# data
train = pd.read_csv("inputs/train.csv")


def create_folds(data, num_splits):
    data["kfold"] = -1
    kf = model_selection.KFold(n_splits=num_splits, shuffle=True, random_state=2021)
    for f, (t_, v_) in enumerate(kf.split(X=data)):
        data.loc[v_, 'kfold'] = f
    return data
train = create_folds(train, num_splits=5)


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
    curr_sent['input_ids'] = tok['input_ids'] + ([tokenizer.pad_token_id] * padding_length)
    curr_sent['token_type_ids'] = tok['token_type_ids'] + \
        ([0] * padding_length)
    curr_sent['attention_mask'] = tok['attention_mask'] + \
        ([0] * padding_length)
    return curr_sent


class DatasetRetriever(Dataset):
    def __init__(self, data, tokenizer, max_len, is_test=False):
        self.data = data
        if 'excerpt' in self.data.columns:
            self.excerpts = self.data.excerpt.values.tolist()
        else:
            self.excerpts = self.data.text.values.tolist()
        self.targets = self.data.target.values.tolist()
        self.tokenizer = tokenizer
        self.is_test = is_test
        self.max_len = max_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, item):
        excerpt, label = self.excerpts[item], self.targets[item]
        features = convert_examples_to_features(
            excerpt, self.tokenizer, 
            self.max_len, self.is_test
        )
        return {
            'input_ids':torch.tensor(features['input_ids'], dtype=torch.long),
            'token_type_ids':torch.tensor(features['token_type_ids'], dtype=torch.long),
            'attention_mask':torch.tensor(features['attention_mask'], dtype=torch.long),
            'label':torch.tensor(label, dtype=torch.double),
        }


class CommonLitModel(nn.Module):
    def __init__(
        self, 
        model_name, 
        config,  
        multisample_dropout=False,
        output_hidden_states=False
    ):
        super(CommonLitModel, self).__init__()
        self.config = config
        self.roberta = AutoModel.from_pretrained(
            model_name, 
            output_hidden_states=output_hidden_states
        )
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        if multisample_dropout:
            self.dropouts = nn.ModuleList([
                nn.Dropout(0.5) for _ in range(5)
            ])
        else:
            self.dropouts = nn.ModuleList([nn.Dropout(0.3)])
        #self.regressor = nn.Linear(config.hidden_size*2, 1)
        self.regressor = nn.Linear(config.hidden_size, 1)
        self._init_weights(self.layer_norm)
        self._init_weights(self.regressor)
 
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
 
    def forward(
        self, 
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None
    ):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        sequence_output = outputs[1]
        sequence_output = self.layer_norm(sequence_output)
 
        # max-avg head
        # average_pool = torch.mean(sequence_output, 1)
        # max_pool, _ = torch.max(sequence_output, 1)
        # concat_sequence_output = torch.cat((average_pool, max_pool), 1)
 
        # multi-sample dropout
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                logits = self.regressor(dropout(sequence_output))
            else:
                logits += self.regressor(dropout(sequence_output))
        
        logits /= len(self.dropouts)
 
        # calculate loss
        loss = None
        if labels is not None:
            # regression task
            loss_fn = torch.nn.MSELoss()
            logits = logits.view(-1).to(labels.dtype)
            loss = torch.sqrt(loss_fn(logits, labels.view(-1)))
        
        output = (logits,) + outputs[1:]
        return ((loss,) + output) if loss is not None else output


class Lamb(Optimizer):
    # Reference code: https://github.com/cybertronai/pytorch-lamb

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0,
        clamp_value: float = 10,
        adam: bool = False,
        debias: bool = False,
    ):
        if lr <= 0.0:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if eps < 0.0:
            raise ValueError('Invalid epsilon value: {}'.format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                'Invalid beta parameter at index 0: {}'.format(betas[0])
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                'Invalid beta parameter at index 1: {}'.format(betas[1])
            )
        if weight_decay < 0:
            raise ValueError(
                'Invalid weight_decay value: {}'.format(weight_decay)
            )
        if clamp_value < 0.0:
            raise ValueError('Invalid clamp value: {}'.format(clamp_value))

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.clamp_value = clamp_value
        self.adam = adam
        self.debias = debias

        super(Lamb, self).__init__(params, defaults)

    def step(self, closure = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    msg = (
                        'Lamb does not support sparse gradients, '
                        'please consider SparseAdam instead'
                    )
                    raise RuntimeError(msg)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                # m_t
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # v_t
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Paper v3 does not use debiasing.
                if self.debias:
                    bias_correction = math.sqrt(1 - beta2 ** state['step'])
                    bias_correction /= 1 - beta1 ** state['step']
                else:
                    bias_correction = 1

                # Apply bias to lr to avoid broadcast.
                step_size = group['lr'] * bias_correction

                weight_norm = torch.norm(p.data).clamp(0, self.clamp_value)

                adam_step = exp_avg / exp_avg_sq.sqrt().add(group['eps'])
                if group['weight_decay'] != 0:
                    adam_step.add_(p.data, alpha=group['weight_decay'])

                adam_norm = torch.norm(adam_step)
                if weight_norm == 0 or adam_norm == 0:
                    trust_ratio = 1
                else:
                    trust_ratio = weight_norm / adam_norm
                state['weight_norm'] = weight_norm
                state['adam_norm'] = adam_norm
                state['trust_ratio'] = trust_ratio
                if self.adam:
                    trust_ratio = 1

                p.data.add_(adam_step, alpha=-step_size * trust_ratio)

        return loss


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


def make_model(model_name='itpt/roberta_base/', num_labels=1):
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    config = AutoConfig.from_pretrained(model_name)
    config.update({'num_labels':num_labels})
    model = CommonLitModel(model_name, config=config)
    return model, tokenizer

def make_optimizer(model, optimizer_name="AdamW"):
    optimizer_grouped_parameters = get_optimizer_params(model)
    kwargs = {
            'lr':5e-5,
            'weight_decay':0.01,
            # 'betas': (0.9, 0.98),
            # 'eps': 1e-06
    }
    if optimizer_name == "LAMB":
        optimizer = Lamb(optimizer_grouped_parameters, **kwargs)
        return optimizer
    elif optimizer_name == "Adam":
        from torch.optim import Adam
        optimizer = Adam(optimizer_grouped_parameters, **kwargs)
        return optimizer
    elif optimizer_name == "AdamW":
        optimizer = AdamW(optimizer_grouped_parameters, **kwargs)
        return optimizer
    else:
        raise Exception('Unknown optimizer: {}'.format(optimizer_name))

def make_scheduler(optimizer, decay_name='linear', t_max=None, warmup_steps=None):
    if decay_name == 'step':
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[30, 60, 90],
            gamma=0.1
        )
    elif decay_name == 'cosine':
        scheduler = lrs.CosineAnnealingLR(
            optimizer,
            T_max=t_max
        )
    elif decay_name == "cosine_warmup":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=t_max
        )
    elif decay_name == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=warmup_steps, 
            num_training_steps=t_max
        )
    else:
        raise Exception('Unknown lr scheduler: {}'.format(decay_type))    
    return scheduler    

def make_loader(
    data, 
    tokenizer, 
    max_len,
    batch_size,
    fold=0
):
    train_set, valid_set = data[data['kfold']!=fold], data[data['kfold']==fold]
    train_dataset = DatasetRetriever(train_set, tokenizer, max_len)
    valid_dataset = DatasetRetriever(valid_set, tokenizer, max_len)

    train_sampler = RandomSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=train_sampler, 
        pin_memory=True, 
        drop_last=False, 
        num_workers=4
    )

    valid_sampler = SequentialSampler(valid_dataset)
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=batch_size // 2, 
        sampler=valid_sampler, 
        pin_memory=True, 
        drop_last=False, 
        num_workers=4
    )

    return train_loader, valid_loader


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.max = 0
        self.min = 1e5

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if val > self.max:
            self.max = val
        if val < self.min:
            self.min = val


class Trainer:
    def __init__(self, model, optimizer, scheduler, scalar=None, log_interval=1, evaluate_interval=1):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scalar = scalar
        self.log_interval = log_interval
        self.evaluate_interval = evaluate_interval
        self.evaluator = Evaluator(self.model, self.scalar)

    def train(self, train_loader, valid_loader, epoch, 
              result_dict, tokenizer, fold):
        count = 0
        losses = AverageMeter()
        self.model.train()
        
        for batch_idx, batch_data in enumerate(train_loader):
            input_ids, attention_mask, token_type_ids, labels = batch_data['input_ids'], \
                batch_data['attention_mask'], batch_data['token_type_ids'], batch_data['label']
            input_ids, attention_mask, token_type_ids, labels = \
                input_ids.cuda(), attention_mask.cuda(), token_type_ids.cuda(), labels.cuda()
            
            if self.scalar is not None:
                with torch.cuda.amp.autocast():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        labels=labels
                    )
            else:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=labels
                )

            loss, logits = outputs[:2]
            count += labels.size(0)
            losses.update(loss.item(), input_ids.size(0))
            
            if self.scalar is not None:
                self.scalar.scale(loss).backward()
                self.scalar.step(self.optimizer)
                self.scalar.update()
            else:
                loss.backward()
                self.optimizer.step()

            self.scheduler.step()
            self.optimizer.zero_grad()

            if batch_idx % self.log_interval == 0:
                _s = str(len(str(len(train_loader.sampler))))
                ret = [
                    ('epoch: {:0>3} [{: >' + _s + '}/{} ({: >3.0f}%)]').format(epoch, count, len(train_loader.sampler), 100 * count / len(train_loader.sampler)),
                    'train_loss: {: >4.5f}'.format(losses.avg),
                ]
                logger.info(', '.join(ret))
            
            if batch_idx % self.evaluate_interval == 0:
                result_dict = self.evaluator.evaluate(
                    valid_loader, 
                    epoch, 
                    result_dict, 
                    tokenizer
                )
                if result_dict['val_loss'][-1] < result_dict['best_val_loss']:
                    logger.info("{} epoch, best epoch was updated! valid_loss: {: >4.5f}".format(epoch, result_dict['val_loss'][-1]))
                    result_dict["best_val_loss"] = result_dict['val_loss'][-1]
                    torch.save(self.model.state_dict(), OUTPUT_DIR+f"model{fold}.bin")

        result_dict['train_loss'].append(losses.avg)
        return result_dict


class Evaluator:
    def __init__(self, model, scalar=None):
        self.model = model
        self.scalar = scalar
    
    def worst_result(self):
        ret = {
            'loss':float('inf'),
            'accuracy':0.0
        }
        return ret

    def result_to_str(self, result):
        ret = [
            'epoch: {epoch:0>3}',
            'loss: {loss: >4.2e}'
        ]
        for metric in self.evaluation_metrics:
            ret.append('{}: {}'.format(metric.name, metric.fmtstr))
        return ', '.join(ret).format(**result)

    def save(self, result):
        with open('result_dict.json', 'w') as f:
            f.write(json.dumps(result, sort_keys=True, indent=4, ensure_ascii=False))
    
    def load(self):
        result = self.worst_result
        if os.path.exists('result_dict.json'):
            with open('result_dict.json', 'r') as f:
                try:
                    result = json.loads(f.read())
                except:
                    pass
        return result

    def evaluate(self, data_loader, epoch, result_dict, tokenizer):
        losses = AverageMeter()

        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(data_loader):
                input_ids, attention_mask, token_type_ids, labels = batch_data['input_ids'], \
                    batch_data['attention_mask'], batch_data['token_type_ids'], batch_data['label']
                input_ids, attention_mask, token_type_ids, labels = input_ids.cuda(), \
                    attention_mask.cuda(), token_type_ids.cuda(), labels.cuda()
                
                if self.scalar is not None:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            labels=labels
                        )
                else:
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        labels=labels
                    )
                
                loss, logits = outputs[:2]
                losses.update(loss.item(), input_ids.size(0))

        logger.info('----Validation Results Summary----')
        logger.info('Epoch: [{}] valid_loss: {: >4.5f}'.format(epoch, losses.avg))

        result_dict['val_loss'].append(losses.avg)        
        return result_dict


def config(fold=0):
    torch.manual_seed(2021)
    torch.cuda.manual_seed(2021)
    torch.cuda.manual_seed_all(2021)
    epochs = 8
    max_len = 250
    batch_size = 16

    model, tokenizer = make_model(model_name='itpt/roberta_base/', num_labels=1)
    train_loader, valid_loader = make_loader(
        train, tokenizer, max_len=max_len,
        batch_size=batch_size, fold=fold
    )

    import math
    num_update_steps_per_epoch = len(train_loader)
    max_train_steps = epochs * num_update_steps_per_epoch
    warmup_proportion = 0
    if warmup_proportion != 0:
        warmup_steps = math.ceil((max_train_steps * 2) / 100)
    else:
        warmup_steps = 0

    optimizer = make_optimizer(model, "AdamW")
    scheduler = make_scheduler(
        optimizer, decay_name='cosine_warmup', 
        t_max=max_train_steps, 
        warmup_steps=warmup_steps
    )    

    if torch.cuda.device_count() >= 1:
        logger.info('Model pushed to {} GPU(s), type {}.'.format(
            torch.cuda.device_count(), 
            torch.cuda.get_device_name(0))
        )
        model = model.cuda() 
    else:
        raise ValueError('CPU training is not supported')

    # scaler = torch.cuda.amp.GradScaler()
    scaler = None

    result_dict = {
        'epoch':[], 
        'train_loss': [], 
        'val_loss' : [], 
        'best_val_loss': np.inf
    }
    return (
        model, tokenizer, 
        optimizer, scheduler, 
        scaler, train_loader, 
        valid_loader, result_dict, 
        epochs
    )


def run(fold=0):
    model, tokenizer, optimizer, scheduler, scaler, \
        train_loader, valid_loader, result_dict, epochs = config(fold)
    
    import time
    trainer = Trainer(model, optimizer, scheduler, scaler)
    train_time_list = []

    for epoch in range(epochs):
        result_dict['epoch'] = epoch

        torch.cuda.synchronize()
        tic1 = time.time()

        result_dict = trainer.train(train_loader, valid_loader, epoch, 
                                    result_dict, tokenizer, fold)

        torch.cuda.synchronize()
        tic2 = time.time() 
        train_time_list.append(tic2 - tic1)

    torch.cuda.empty_cache()
    del model, tokenizer, optimizer, scheduler, \
        scaler, train_loader, valid_loader,
    gc.collect()
    return result_dict


result_list = []
for fold in range(5):
    logger.info('----')
    logger.info(f'FOLD: {fold}')
    result_dict = run(fold)
    result_list.append(result_dict)
    logger.info('----')


[logger.info("FOLD::", i, "Loss:: ", fold['best_val_loss']) for i, fold in enumerate(result_list)]


oof = np.zeros(len(train))
for fold in tqdm(range(5), total=5):
    model, tokenizer = make_model()
    model.load_state_dict(
        torch.load(f'model{fold}.bin')
    )
    model.cuda()
    model.eval()
    val_index = train[train.kfold==fold].index.tolist()
    train_loader, val_loader = make_loader(train, tokenizer, 250, 16, fold=fold)
    # scalar = torch.cuda.amp.GradScaler()
    scalar = None
    preds = []
    for index, data in enumerate(val_loader):
        input_ids, attention_mask, token_type_ids, labels = data['input_ids'], \
            data['attention_mask'], data['token_type_ids'], data['label']
        input_ids, attention_mask, token_type_ids, labels = input_ids.cuda(), \
            attention_mask.cuda(), token_type_ids.cuda(), labels.cuda()
        if scalar is not None:
            with torch.cuda.amp.autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=labels
                )
        else:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels
            )
        
        loss, logits = outputs[:2]
        preds += logits.cpu().detach().numpy().tolist()
    oof[val_index] = preds


from sklearn.metrics import mean_squared_error
logger.info(round(np.sqrt(mean_squared_error(train.target.values, oof)), 4))


