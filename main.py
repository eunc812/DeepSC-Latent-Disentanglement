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
Created on Tue May 26 16:59:14 2020

@author: HQ Xie
"""
import os
import argparse
import time
import json
import torch
import random
import torch.nn as nn
import numpy as np

from utils import (
    SNR_to_noise, initNetParams,
    train_step, val_step, train_mi,
    train_step_zsplit_semantic, val_step_zsplit_semantic,
    train_step_zsplit_robust, val_step_zsplit_robust,
    train_step_zsplit_gating, val_step_zsplit_gating,
)

from dataset import EurDataset, collate_data
from models.transceiver import DeepSC, DeepSCZSplit
from models.mutual_info import Mine
from torch.utils.data import DataLoader
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument(
    '--arch', default='baseline',
    choices=['baseline', 'zsplit_sem', 'zsplit_robust', 'zsplit_gate'],
    help='baseline / zsplit_sem / zsplit_robust / zsplit_gate'
)

#parser.add_argument('--data-dir', default='data/train_data.pkl', type=str)
parser.add_argument('--vocab-file', default='europarl/vocab.json', type=str)
parser.add_argument('--checkpoint-path', default='checkpoints/deepsc-Rayleigh', type=str)
parser.add_argument('--channel', default='Rayleigh', type=str,
                    help='Please choose AWGN, Rayleigh, and Rician')
parser.add_argument('--MAX-LENGTH', default=30, type=int)
parser.add_argument('--MIN-LENGTH', default=4, type=int)
parser.add_argument('--d-model', default=128, type=int)
parser.add_argument('--dff', default=512, type=int)
parser.add_argument('--num-layers', default=4, type=int)
parser.add_argument('--num-heads', default=8, type=int)
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--epochs', default=80, type=int)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train(epoch, args, net):
    train_eur = EurDataset('train')
    train_iterator = DataLoader(
        train_eur, batch_size=args.batch_size,
        num_workers=0, pin_memory=True,
        collate_fn=collate_data
    )
    pbar = tqdm(train_iterator)

    #ADDED (This work):
    # 한 epoch 동안 고정된 SNR (원래 코드와 동일한 동작)
    noise_std = np.random.uniform(SNR_to_noise(5), SNR_to_noise(10), size=(1))

    for sents in pbar:
        sents = sents.to(device)

        if args.arch == 'baseline':
            loss = train_step(
                net, sents, sents, noise_std[0],
                pad_idx, optimizer, criterion, args.channel
            )
        elif args.arch == 'zsplit_sem':
            loss = train_step_zsplit_semantic(
                net, sents, sents, noise_std[0],
                pad_idx, optimizer, criterion, args.channel
            )
        elif args.arch == 'zsplit_robust':
            loss = train_step_zsplit_robust(
                net, sents, sents, noise_std[0],
                pad_idx, optimizer, criterion, args.channel
            )
        elif args.arch == 'zsplit_gate':
            loss = train_step_zsplit_gating(
                net, sents, sents, noise_std[0],
                pad_idx, optimizer, criterion, args.channel
            )
        else:
            raise ValueError(f"Unknown arch: {args.arch}")

        pbar.set_description(
            f"Epoch: {epoch+1}; Type: Train; Loss: {loss:.5f}"
        )


def validate(epoch, args, net):
    """
    검증 단계:
    - gradient / optimizer 갱신 없음
    - val_step_* 함수 사용
    - train()과 동일하게 한 번 샘플한 noise_std를 사용
    """
    test_eur = EurDataset('test')
    test_iterator = DataLoader(
        test_eur, batch_size=args.batch_size,
        num_workers=0, pin_memory=True,
        collate_fn=collate_data
    )
    net.eval()
    pbar = tqdm(test_iterator)
    total = 0.0
    # ADDED (This work):
    # 🔥 FIX 1: noise_std 정의
    noise_std = np.random.uniform(SNR_to_noise(5), SNR_to_noise(10), size=(1))

    with torch.no_grad():
        for sents in pbar:
            sents = sents.to(device)

            if args.arch == 'baseline':
                loss = val_step(
                    net, sents, sents, noise_std[0],
                    pad_idx, criterion, args.channel
                )
            elif args.arch == 'zsplit_sem':
                loss = val_step_zsplit_semantic(
                    net, sents, sents, noise_std[0],
                    pad_idx, criterion, args.channel
                )
            elif args.arch == 'zsplit_robust':
                loss = val_step_zsplit_robust(
                    net, sents, sents, noise_std[0],
                    pad_idx, criterion, args.channel
                )
            elif args.arch == 'zsplit_gate':
                loss = val_step_zsplit_gating(
                    net, sents, sents, noise_std[0],
                    pad_idx, criterion, args.channel
                )
            else:
                raise ValueError(f"Unknown arch: {args.arch}")

            # 🔥 FIX 2: total/loss 집계는 if-elif 바깥에서
            total += loss
            pbar.set_description(
                f"Epoch: {epoch+1}; Type: VAL; Loss: {loss:.5f}"
            )

    return total / len(test_iterator)


if __name__ == '__main__':
    # setup_seed(10)
    args = parser.parse_args()
    args.vocab_file = '/import/antennas/Datasets/hx301/' + args.vocab_file

    """ preparing the dataset """
    vocab = json.load(open(args.vocab_file, 'rb'))
    token_to_idx = vocab['token_to_idx']
    num_vocab = len(token_to_idx)
    pad_idx = token_to_idx["<PAD>"]
    start_idx = token_to_idx["<START>"]
    end_idx = token_to_idx["<END>"]

    """ define model, optimizer and loss function """
    # ADDED (This work):
    if args.arch == 'baseline':
        deepsc = DeepSC(
            args.num_layers, num_vocab, num_vocab,
            num_vocab, num_vocab, args.d_model,
            args.num_heads, args.dff, 0.1
        ).to(device)
    else:
        # zsplit_sem, zsplit_robust, zsplit_gate 모두 같은 모델 클래스 사용
        deepsc = DeepSCZSplit(
            args.num_layers, num_vocab, num_vocab,
            num_vocab, num_vocab, args.d_model,
            args.num_heads, args.dff, 0.1
        ).to(device)

    mi_net = Mine().to(device)
    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.Adam(
        deepsc.parameters(),
        lr=1e-4, betas=(0.9, 0.98), eps=1e-8, weight_decay=5e-4
    )
    mi_opt = torch.optim.Adam(mi_net.parameters(), lr=1e-3)

    initNetParams(deepsc)

    # 체크포인트 폴더 미리 생성
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)
    # ADDED (This work):
    # 🔥 FIX 3: record_acc는 epoch 루프 바깥에서 한 번만 초기화
    best_loss = float('inf')

    for epoch in range(args.epochs):
        start = time.time()

        train(epoch, args, deepsc)
        avg_loss = validate(epoch, args, deepsc)

        # ----- (1) 매 epoch마다 현재 모델 저장 -----
        ckpt_path = os.path.join(
            args.checkpoint_path,
            f'checkpoint_{str(epoch + 1).zfill(2)}.pth'
        )
        torch.save(deepsc.state_dict(), ckpt_path)

        # ----- (2) best 모델은 별도 파일로 저장 -----
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(
                args.checkpoint_path,
                'checkpoint_best.pth'
            )
            torch.save(deepsc.state_dict(), best_path)

        print(f"[Epoch {epoch+1}] VAL loss: {avg_loss:.5f}, best: {best_loss:.5f}")



    

        



