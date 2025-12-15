"""
This file is based on the DeepSC implementation from:

H. Xie, Z. Qin, G. Y. Li, and B. H. Juang,
"Deep Learning Enabled Semantic Communication Systems,"
IEEE Transactions on Signal Processing, 2021.

Original code:
https://github.com/13274086/DeepSC

--------------------------------------------------
Modifications and extensions (This work, 2025):
- Latent space disentanglement (z_sem, z_rob, z_snr)
- SNR-aware gating mechanism
--------------------------------------------------

# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 09:47:54 2020

@author: HQ Xie
utils.py
"""
import os 
import math
import torch
import time
import torch.nn as nn
import numpy as np
from w3lib.html import remove_tags
from nltk.translate.bleu_score import sentence_bleu
from models.mutual_info import sample_batch, mutual_information

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class BleuScore():
    def __init__(self, w1, w2, w3, w4):
        self.w1 = w1 # 1-gram weights
        self.w2 = w2 # 2-grams weights
        self.w3 = w3 # 3-grams weights
        self.w4 = w4 # 4-grams weights
    
    def compute_blue_score(self, real, predicted):
        score = []
        for (sent1, sent2) in zip(real, predicted):
            sent1 = remove_tags(sent1).split()
            sent2 = remove_tags(sent2).split()
            score.append(sentence_bleu([sent1], sent2, 
                          weights=(self.w1, self.w2, self.w3, self.w4)))
        return score
            

class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        # 将数组全部填充为某一个值
        true_dist.fill_(self.smoothing / (self.size - 2)) 
        # 按照index将input重新排列 
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence) 
        # 第一行加入了<strat> 符号，不需要加入计算
        true_dist[:, self.padding_idx] = 0 #
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist)


class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        self._weight_decay = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        weight_decay = self.weight_decay()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
            p['weight_decay'] = weight_decay
        self._rate = rate
        self._weight_decay = weight_decay
        # update weights
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
            
        # if step <= 3000 :
        #     lr = 1e-3
            
        # if step > 3000 and step <=9000:
        #     lr = 1e-4
             
        # if step>9000:
        #     lr = 1e-5
         
        lr = self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))
  
        return lr
    

        # return lr
    
    def weight_decay(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
            
        if step <= 3000 :
            weight_decay = 1e-3
            
        if step > 3000 and step <=9000:
            weight_decay = 0.0005
             
        if step>9000:
            weight_decay = 1e-4

        weight_decay =   0
        return weight_decay

            
class SeqtoText:
    def __init__(self, vocb_dictionary, end_idx):
        self.reverse_word_map = dict(zip(vocb_dictionary.values(), vocb_dictionary.keys()))
        self.end_idx = end_idx
        
    def sequence_to_text(self, list_of_indices):
        # Looking up words in dictionary
        words = []
        for idx in list_of_indices:
            if idx == self.end_idx:
                break
            else:
                words.append(self.reverse_word_map.get(idx))
        words = ' '.join(words)
        return(words) 


class Channels():

    def AWGN(self, Tx_sig, n_var):
        Rx_sig = Tx_sig + torch.normal(0, n_var, size=Tx_sig.shape).to(device)
        return Rx_sig

    def Rayleigh(self, Tx_sig, n_var):
        shape = Tx_sig.shape
        H_real = torch.normal(0, math.sqrt(1/2), size=[1]).to(device)
        H_imag = torch.normal(0, math.sqrt(1/2), size=[1]).to(device)
        H = torch.Tensor([[H_real, -H_imag], [H_imag, H_real]]).to(device)
        Tx_sig = torch.matmul(Tx_sig.view(shape[0], -1, 2), H)
        Rx_sig = self.AWGN(Tx_sig, n_var)
        # Channel estimation
        Rx_sig = torch.matmul(Rx_sig, torch.inverse(H)).view(shape)

        return Rx_sig

    def Rician(self, Tx_sig, n_var, K=1):
        shape = Tx_sig.shape
        mean = math.sqrt(K / (K + 1))
        std = math.sqrt(1 / (K + 1))
        H_real = torch.normal(mean, std, size=[1]).to(device)
        H_imag = torch.normal(mean, std, size=[1]).to(device)
        H = torch.Tensor([[H_real, -H_imag], [H_imag, H_real]]).to(device)
        Tx_sig = torch.matmul(Tx_sig.view(shape[0], -1, 2), H)
        Rx_sig = self.AWGN(Tx_sig, n_var)
        # Channel estimation
        Rx_sig = torch.matmul(Rx_sig, torch.inverse(H)).view(shape)

        return Rx_sig

def initNetParams(model):
    '''Init net parameters.'''
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
         
def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    # 产生下三角矩阵
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask)

    
def create_masks(src, trg, padding_idx):

    src_mask = (src == padding_idx).unsqueeze(-2).type(torch.FloatTensor) #[batch, 1, seq_len]

    trg_mask = (trg == padding_idx).unsqueeze(-2).type(torch.FloatTensor) #[batch, 1, seq_len]
    look_ahead_mask = subsequent_mask(trg.size(-1)).type_as(trg_mask.data)
    combined_mask = torch.max(trg_mask, look_ahead_mask)
    
    return src_mask.to(device), combined_mask.to(device)

def loss_function(x, trg, padding_idx, criterion):
    
    loss = criterion(x, trg)
    mask = (trg != padding_idx).type_as(loss.data)
    # a = mask.cpu().numpy()
    loss *= mask
    
    return loss.mean()

def PowerNormalize(x):
    
    x_square = torch.mul(x, x)
    power = torch.mean(x_square).sqrt()
    if power > 1:
        x = torch.div(x, power)
    
    return x


def SNR_to_noise(snr):
    snr = 10 ** (snr / 10)
    noise_std = 1 / np.sqrt(2 * snr)

    return noise_std

def train_step(model, src, trg, n_var, pad, opt, criterion, channel, mi_net=None):
    model.train()

    trg_inp = trg[:, :-1]
    trg_real = trg[:, 1:]

    channels = Channels()
    opt.zero_grad()
    
    src_mask, look_ahead_mask = create_masks(src, trg_inp, pad)
    
    enc_output = model.encoder(src, src_mask)
    channel_enc_output = model.channel_encoder(enc_output)
    Tx_sig = PowerNormalize(channel_enc_output)

    if channel == 'AWGN':
        Rx_sig = channels.AWGN(Tx_sig, n_var)
    elif channel == 'Rayleigh':
        Rx_sig = channels.Rayleigh(Tx_sig, n_var)
    elif channel == 'Rician':
        Rx_sig = channels.Rician(Tx_sig, n_var)
    else:
        raise ValueError("Please choose from AWGN, Rayleigh, and Rician")

    channel_dec_output = model.channel_decoder(Rx_sig)
    dec_output = model.decoder(trg_inp, channel_dec_output, look_ahead_mask, src_mask)
    pred = model.dense(dec_output)
    
    # pred = model(src, trg_inp, src_mask, look_ahead_mask, n_var)
    ntokens = pred.size(-1)
    
    #y_est = x +  torch.matmul(n, torch.inverse(H))
    #loss1 = torch.mean(torch.pow((x_est - y_est.view(x_est.shape)), 2))

    loss = loss_function(pred.contiguous().view(-1, ntokens), 
                         trg_real.contiguous().view(-1), 
                         pad, criterion)

    if mi_net is not None:
        mi_net.eval()
        joint, marginal = sample_batch(Tx_sig, Rx_sig)
        mi_lb, _, _ = mutual_information(joint, marginal, mi_net)
        loss_mine = -mi_lb
        loss = loss + 0.0009 * loss_mine
    # loss = loss_function(pred, trg_real, pad)

    loss.backward()
    opt.step()

    return loss.item()
    
# ADDED (This work):
def train_step_zsplit_semantic(model, src, trg, n_var,
                               pad, opt, criterion, channel,
                               mi_net=None):
    """
    DeepSCZSplit + Option 1:
    - encoder 출력 → (z_sem, z_rob, z_snr) → z_total
    - z_total은 채널 path로
    - z_sem은 decoder context에 직접 더해서 semantic 경로 강화
    """
    model.train()

    trg_inp = trg[:, :-1]
    trg_real = trg[:, 1:]

    channels = Channels()
    opt.zero_grad()

    # 마스크
    src_mask, look_ahead_mask = create_masks(src, trg_inp, pad)

    # 1) encoder
    enc_output = model.encoder(src, src_mask)              # [B,T,d_model]

    # 2) 3-way latent split
    z_total, z_dict = model.split_latent(enc_output, n_var)
    z_sem = z_dict["z_sem"]                                # [B,T,d_model]

    # 3) channel encoder + normalize (z_total만 채널로 보냄)
    channel_enc_output = model.channel_encoder(z_total)    # [B,T,16]
    Tx_sig = PowerNormalize(channel_enc_output)

    # 4) 채널 통과
    if channel == 'AWGN':
        Rx_sig = channels.AWGN(Tx_sig, n_var)
    elif channel == 'Rayleigh':
        Rx_sig = channels.Rayleigh(Tx_sig, n_var)
    elif channel == 'Rician':
        Rx_sig = channels.Rician(Tx_sig, n_var)
    else:
        raise ValueError("Please choose from AWGN, Rayleigh, and Rician")

    # 5) channel decoder
    channel_dec_output = model.channel_decoder(Rx_sig)     # [B,T,d_model]

    # 🔥 Option 1 핵심:
    # decoder가 보는 "context"에 z_sem을 직접 더해준다.
    # 즉, 채널을 통과한 정보 + encoder semantic latent를 같이 보게 됨.
    decoder_context = channel_dec_output + model.sem_weight * z_sem

    # 6) semantic decoder
    dec_output = model.decoder(trg_inp, decoder_context,
                               look_ahead_mask, src_mask)
    pred = model.dense(dec_output)                         # [B,T,vocab]
    ntokens = pred.size(-1)

    # 7) loss
    loss = loss_function(pred.contiguous().view(-1, ntokens),
                         trg_real.contiguous().view(-1),
                         pad, criterion)

    # (옵션) mutual information 등 추가
    if mi_net is not None:
        joint, marginal = sample_batch(Tx_sig, Rx_sig)
        mi_lb, _, _ = mutual_information(joint, marginal, mi_net)
        # 여기다가 loss에 조합해도 됨 (원하면 나중에 설계하자)

    loss.backward()
    opt.step()

    return loss.item()

# ADDED (This work):
def train_step_zsplit_robust(model, src, trg, n_var,
                             pad, opt, criterion, channel,
                             mi_net=None):
    """
    Step 2: robust-path only
    - encoder → (z_sem, z_rob, z_snr)
    - 채널에는 (z_rob + z_snr)/2 만 보냄
    - z_sem은 decoder shortcut으로만 사용
    """
    model.train()

    trg_inp = trg[:, :-1]
    trg_real = trg[:, 1:]

    channels = Channels()
    opt.zero_grad()

    src_mask, look_ahead_mask = create_masks(src, trg_inp, pad)

    # 1) encoder
    enc_output = model.encoder(src, src_mask)

    # 2) 3-way latent split
    _, z_dict = model.split_latent(enc_output, n_var)
    z_sem = z_dict["z_sem"]              # semantic shortcut
    z_rob = z_dict["z_rob"]              # robust branch
    z_snr = z_dict["z_snr"]              # snr-aware branch

    # 🔥 Step 2 핵심: 채널에는 z_sem을 빼고 보냄
    z_total = (z_rob + z_snr) / 2.0      # [B,T,d_model]

    # 3) channel encoder + normalize
    channel_enc_output = model.channel_encoder(z_total)
    Tx_sig = PowerNormalize(channel_enc_output)

    # 4) physical channel
    if channel == 'AWGN':
        Rx_sig = channels.AWGN(Tx_sig, n_var)
    elif channel == 'Rayleigh':
        Rx_sig = channels.Rayleigh(Tx_sig, n_var)
    elif channel == 'Rician':
        Rx_sig = channels.Rician(Tx_sig, n_var)
    else:
        raise ValueError("Please choose from AWGN, Rayleigh, and Rician")

    # 5) channel decoder
    channel_dec_output = model.channel_decoder(Rx_sig)

    # 6) semantic shortcut 그대로 유지
    decoder_context = channel_dec_output + model.sem_weight * z_sem

    # 7) decoder
    dec_output = model.decoder(trg_inp, decoder_context,
                               look_ahead_mask, src_mask)
    pred = model.dense(dec_output)
    ntokens = pred.size(-1)

    loss = loss_function(pred.contiguous().view(-1, ntokens),
                         trg_real.contiguous().view(-1),
                         pad, criterion)

    # (원하면 여기서 mi_net으로 mutual information regularization 추가 가능)
    loss.backward()
    opt.step()

    return loss.item()

# ADDED (This work):
def train_step_zsplit_gating(model, src, trg, n_var,
                             pad, opt, criterion, channel,
                             mi_net=None):
    """
    Step 3: SNR-aware gating
    - encoder → (z_sem, z_rob, z_snr)
    - z_snr로부터 gate(w_sem, w_rob)를 만들고
      z_total = w_sem * z_sem + w_rob * z_rob
    - z_total만 채널로 보내고,
      z_sem은 계속 decoder shortcut으로 사용
    """
    model.train()

    trg_inp = trg[:, :-1]
    trg_real = trg[:, 1:]

    channels = Channels()
    opt.zero_grad()

    src_mask, look_ahead_mask = create_masks(src, trg_inp, pad)

    # 1) encoder
    enc_output = model.encoder(src, src_mask)          # [B,T,d_model]

    # 2) 3-way latent split
    _, z_dict = model.split_latent(enc_output, n_var)
    z_sem = z_dict["z_sem"]                            # [B,T,d_model]
    z_rob = z_dict["z_rob"]
    z_snr = z_dict["z_snr"]                            # SNR-aware feature

    # 3) 🔥 gating: z_snr → [w_sem, w_rob]
    gate_logits = model.gate_net(z_snr)                # [B,T,2]
    gate = torch.softmax(gate_logits, dim=-1)          # [B,T,2], 합=1
    w_sem = gate[..., 0:1]                             # [B,T,1]
    w_rob = gate[..., 1:2]                             # [B,T,1]

    # 4) gated combination: z_total = w_sem*z_sem + w_rob*z_rob
    z_total = w_sem * z_sem + w_rob * z_rob            # [B,T,d_model]

    # 5) channel encoder + 채널
    channel_enc_output = model.channel_encoder(z_total)
    Tx_sig = PowerNormalize(channel_enc_output)

    if channel == 'AWGN':
        Rx_sig = channels.AWGN(Tx_sig, n_var)
    elif channel == 'Rayleigh':
        Rx_sig = channels.Rayleigh(Tx_sig, n_var)
    elif channel == 'Rician':
        Rx_sig = channels.Rician(Tx_sig, n_var)
    else:
        raise ValueError("Please choose from AWGN, Rayleigh, and Rician")

    # 6) channel decoder
    channel_dec_output = model.channel_decoder(Rx_sig)

    # 7) semantic shortcut 유지
    decoder_context = channel_dec_output + model.sem_weight * z_sem

    # 8) decoder + loss
    dec_output = model.decoder(trg_inp, decoder_context,
                               look_ahead_mask, src_mask)
    pred = model.dense(dec_output)
    ntokens = pred.size(-1)

    loss = loss_function(pred.contiguous().view(-1, ntokens),
                         trg_real.contiguous().view(-1),
                         pad, criterion)

    loss.backward()
    opt.step()

    return loss.item()

def train_mi(model, mi_net, src, n_var, padding_idx, opt, channel):
    mi_net.train()
    opt.zero_grad()
    channels = Channels()
    src_mask = (src == padding_idx).unsqueeze(-2).type(torch.FloatTensor).to(device)  # [batch, 1, seq_len]
    enc_output = model.encoder(src, src_mask)
    channel_enc_output = model.channel_encoder(enc_output)
    Tx_sig = PowerNormalize(channel_enc_output)

    if channel == 'AWGN':
        Rx_sig = channels.AWGN(Tx_sig, n_var)
    elif channel == 'Rayleigh':
        Rx_sig = channels.Rayleigh(Tx_sig, n_var)
    elif channel == 'Rician':
        Rx_sig = channels.Rician(Tx_sig, n_var)
    else:
        raise ValueError("Please choose from AWGN, Rayleigh, and Rician")

    joint, marginal = sample_batch(Tx_sig, Rx_sig)
    mi_lb, _, _ = mutual_information(joint, marginal, mi_net)
    loss_mine = -mi_lb

    loss_mine.backward()
    torch.nn.utils.clip_grad_norm_(mi_net.parameters(), 10.0)
    opt.step()

    return loss_mine.item()

def val_step(model, src, trg, n_var, pad, criterion, channel):
    channels = Channels()
    trg_inp = trg[:, :-1]
    trg_real = trg[:, 1:]

    src_mask, look_ahead_mask = create_masks(src, trg_inp, pad)

    enc_output = model.encoder(src, src_mask)
    channel_enc_output = model.channel_encoder(enc_output)
    Tx_sig = PowerNormalize(channel_enc_output)

    if channel == 'AWGN':
        Rx_sig = channels.AWGN(Tx_sig, n_var)
    elif channel == 'Rayleigh':
        Rx_sig = channels.Rayleigh(Tx_sig, n_var)
    elif channel == 'Rician':
        Rx_sig = channels.Rician(Tx_sig, n_var)
    else:
        raise ValueError("Please choose from AWGN, Rayleigh, and Rician")

    channel_dec_output = model.channel_decoder(Rx_sig)
    dec_output = model.decoder(trg_inp, channel_dec_output, look_ahead_mask, src_mask)
    pred = model.dense(dec_output)

    # pred = model(src, trg_inp, src_mask, look_ahead_mask, n_var)
    ntokens = pred.size(-1)
    loss = loss_function(pred.contiguous().view(-1, ntokens), 
                         trg_real.contiguous().view(-1), 
                         pad, criterion)
    # loss = loss_function(pred, trg_real, pad)
    
    return loss.item()

# ADDED (This work):
def val_step_zsplit_semantic(model, src, trg, n_var,
                             pad, criterion, channel):
    """
    DeepSCZSplit + Option 1 validation step
    - gradient 없이 train_step_zsplit_semantic와 같은 경로
    """
    model.eval()

    trg_inp = trg[:, :-1]
    trg_real = trg[:, 1:]

    channels = Channels()

    src_mask, look_ahead_mask = create_masks(src, trg_inp, pad)

    with torch.no_grad():
        enc_output = model.encoder(src, src_mask)
        z_total, z_dict = model.split_latent(enc_output, n_var)
        z_sem = z_dict["z_sem"]

        channel_enc_output = model.channel_encoder(z_total)
        Tx_sig = PowerNormalize(channel_enc_output)

        if channel == 'AWGN':
            Rx_sig = channels.AWGN(Tx_sig, n_var)
        elif channel == 'Rayleigh':
            Rx_sig = channels.Rayleigh(Tx_sig, n_var)
        elif channel == 'Rician':
            Rx_sig = channels.Rician(Tx_sig, n_var)
        else:
            raise ValueError("Please choose from AWGN, Rayleigh, and Rician")

        channel_dec_output = model.channel_decoder(Rx_sig)
        decoder_context = channel_dec_output + model.sem_weight * z_sem

        dec_output = model.decoder(trg_inp, decoder_context,
                                   look_ahead_mask, src_mask)
        pred = model.dense(dec_output)
        ntokens = pred.size(-1)

        loss = loss_function(pred.contiguous().view(-1, ntokens),
                             trg_real.contiguous().view(-1),
                             pad, criterion)

    return loss.item()

# ADDED (This work):
def val_step_zsplit_robust(model, src, trg, n_var,
                           pad, criterion, channel):
    """
    Step 2: robust-path only validation
    """
    model.eval()

    trg_inp = trg[:, :-1]
    trg_real = trg[:, 1:]

    channels = Channels()
    src_mask, look_ahead_mask = create_masks(src, trg_inp, pad)

    with torch.no_grad():
        enc_output = model.encoder(src, src_mask)
        _, z_dict = model.split_latent(enc_output, n_var)
        z_sem = z_dict["z_sem"]
        z_rob = z_dict["z_rob"]
        z_snr = z_dict["z_snr"]

        z_total = (z_rob + z_snr) / 2.0   # 🔥 robust-path only

        channel_enc_output = model.channel_encoder(z_total)
        Tx_sig = PowerNormalize(channel_enc_output)

        if channel == 'AWGN':
            Rx_sig = channels.AWGN(Tx_sig, n_var)
        elif channel == 'Rayleigh':
            Rx_sig = channels.Rayleigh(Tx_sig, n_var)
        elif channel == 'Rician':
            Rx_sig = channels.Rician(Tx_sig, n_var)
        else:
            raise ValueError("Please choose from AWGN, Rayleigh, and Rician")

        channel_dec_output = model.channel_decoder(Rx_sig)
        decoder_context = channel_dec_output + model.sem_weight * z_sem

        dec_output = model.decoder(trg_inp, decoder_context,
                                   look_ahead_mask, src_mask)
        pred = model.dense(dec_output)
        ntokens = pred.size(-1)

        loss = loss_function(pred.contiguous().view(-1, ntokens),
                             trg_real.contiguous().view(-1),
                             pad, criterion)

    return loss.item()

# ADDED (This work):
def val_step_zsplit_gating(model, src, trg, n_var,
                           pad, criterion, channel):
    """
    Step 3: gating 버전 validation
    """
    model.eval()

    trg_inp = trg[:, :-1]
    trg_real = trg[:, 1:]

    channels = Channels()
    src_mask, look_ahead_mask = create_masks(src, trg_inp, pad)

    with torch.no_grad():
        enc_output = model.encoder(src, src_mask)
        _, z_dict = model.split_latent(enc_output, n_var)
        z_sem = z_dict["z_sem"]
        z_rob = z_dict["z_rob"]
        z_snr = z_dict["z_snr"]

        gate_logits = model.gate_net(z_snr)
        gate = torch.softmax(gate_logits, dim=-1)
        w_sem = gate[..., 0:1]
        w_rob = gate[..., 1:2]

        z_total = w_sem * z_sem + w_rob * z_rob

        channel_enc_output = model.channel_encoder(z_total)
        Tx_sig = PowerNormalize(channel_enc_output)

        if channel == 'AWGN':
            Rx_sig = channels.AWGN(Tx_sig, n_var)
        elif channel == 'Rayleigh':
            Rx_sig = channels.Rayleigh(Tx_sig, n_var)
        elif channel == 'Rician':
            Rx_sig = channels.Rician(Tx_sig, n_var)
        else:
            raise ValueError("Please choose from AWGN, Rayleigh, and Rician")

        channel_dec_output = model.channel_decoder(Rx_sig)
        decoder_context = channel_dec_output + model.sem_weight * z_sem

        dec_output = model.decoder(trg_inp, decoder_context,
                                   look_ahead_mask, src_mask)
        pred = model.dense(dec_output)
        ntokens = pred.size(-1)

        loss = loss_function(pred.contiguous().view(-1, ntokens),
                             trg_real.contiguous().view(-1),
                             pad, criterion)

    return loss.item()

def greedy_decode(model, src, n_var, max_len, padding_idx, start_symbol, channel):
    """ 
    这里采用贪婪解码器，如果需要更好的性能情况下，可以使用beam search decode
    """
    # create src_mask
    channels = Channels()
    src_mask = (src == padding_idx).unsqueeze(-2).type(torch.FloatTensor).to(device) #[batch, 1, seq_len]

    enc_output = model.encoder(src, src_mask)
    channel_enc_output = model.channel_encoder(enc_output)
    Tx_sig = PowerNormalize(channel_enc_output)

    if channel == 'AWGN':
        Rx_sig = channels.AWGN(Tx_sig, n_var)
    elif channel == 'Rayleigh':
        Rx_sig = channels.Rayleigh(Tx_sig, n_var)
    elif channel == 'Rician':
        Rx_sig = channels.Rician(Tx_sig, n_var)
    else:
        raise ValueError("Please choose from AWGN, Rayleigh, and Rician")
            
    #channel_enc_output = model.blind_csi(channel_enc_output)
          
    memory = model.channel_decoder(Rx_sig)
    
    outputs = torch.ones(src.size(0), 1).fill_(start_symbol).type_as(src.data)

    for i in range(max_len - 1):
        # create the decode mask
        trg_mask = (outputs == padding_idx).unsqueeze(-2).type(torch.FloatTensor) #[batch, 1, seq_len]
        look_ahead_mask = subsequent_mask(outputs.size(1)).type(torch.FloatTensor)
#        print(look_ahead_mask)
        combined_mask = torch.max(trg_mask, look_ahead_mask)
        combined_mask = combined_mask.to(device)

        # decode the received signal
        dec_output = model.decoder(outputs, memory, combined_mask, None)
        pred = model.dense(dec_output)
        
        # predict the word
        prob = pred[: ,-1:, :]  # (batch_size, 1, vocab_size)
        #prob = prob.squeeze()

        # return the max-prob index
        _, next_word = torch.max(prob, dim = -1)
        #next_word = next_word.unsqueeze(1)
        
        #next_word = next_word.data[0]
        outputs = torch.cat([outputs, next_word], dim=1)

    return outputs




