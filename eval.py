# -*- coding: utf-8 -*-
# ======================================
# 1) Import + Matplotlib 스타일 설정
# ======================================
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 논문용 그래프 스타일 (폰트/선굵기/범례 등)
plt.rcParams.update({
    "font.size": 13,
    "axes.labelsize": 13,
    "axes.titlesize": 15,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "lines.linewidth": 2.0,
})

# main.py parser 불러오기
import os, sys

PROJECT_ROOT = "/content/DeepSC"  # ✅ 너의 프로젝트 루트 경로로 바꾸기
sys.path.append(PROJECT_ROOT)

print("PROJECT_ROOT:", PROJECT_ROOT)
print("exists?:", os.path.exists(PROJECT_ROOT))

import main

from dataset import EurDataset, collate_data
from models.transceiver import DeepSC, DeepSCZSplit
from utils import (
    Channels, PowerNormalize, subsequent_mask,
    SNR_to_noise, SeqtoText, BleuScore, device
)

# ---- NLTK BLEU + smoothing ----
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
smooth_fn = SmoothingFunction().method1


# ======================================
# 2) smoothed BLEU 계산 함수
# ======================================
def compute_bleu_smooth(refs, hyps, weights=(0.25,0.25,0.25,0.25)):
    scores = []
    for r, h in zip(refs, hyps):
        r_tok = r.split()
        h_tok = h.split()
        if len(h_tok) == 0:
            scores.append(0.0)
            continue
        score = sentence_bleu(
            [r_tok], h_tok,
            weights=weights,
            smoothing_function=smooth_fn
        )
        scores.append(score)
    return scores


# ======================================
# 3) args 기본값 불러오기 (main.py)
# ======================================
args = main.parser.parse_args([])
args.vocab_file = '/import/antennas/Datasets/hx301/' + args.vocab_file

vocab = json.load(open(args.vocab_file, "rb"))
token_to_idx = vocab["token_to_idx"]
num_vocab = len(token_to_idx)

pad_idx   = token_to_idx["<PAD>"]
start_idx = token_to_idx["<START>"]
end_idx   = token_to_idx["<END>"]

# test loader
test_eur = EurDataset('test')
test_iterator = DataLoader(
    test_eur,
    batch_size=args.batch_size,
    num_workers=0,
    pin_memory=True,
    collate_fn=collate_data
)

seq2text = SeqtoText(token_to_idx, end_idx)


# ======================================
# 4) SBERT sentence similarity
# ======================================
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F

print("Loading SBERT model ...")
sim_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
if device.type == "cuda":
    sim_model = sim_model.to(device)
print("SBERT loaded.")

def sentence_similarity(refs, hyps, batch_size=128):
    sims = []
    for i in range(0, len(refs), batch_size):
        ref_batch = refs[i:i+batch_size]
        hyp_batch = hyps[i:i+batch_size]

        ref_emb = sim_model.encode(ref_batch, convert_to_tensor=True)
        hyp_emb = sim_model.encode(hyp_batch, convert_to_tensor=True)

        cos = F.cosine_similarity(ref_emb, hyp_emb)
        sims.extend(cos.detach().cpu().tolist())
    return sims


# ======================================
# 5) 모델 생성 + checkpoint 로드
# ======================================
def build_model(arch):
    if arch == "baseline":
        net = DeepSC(
            args.num_layers, num_vocab, num_vocab,
            num_vocab, num_vocab, args.d_model,
            args.num_heads, args.dff, 0.1
        ).to(device)
    else:
        net = DeepSCZSplit(
            args.num_layers, num_vocab, num_vocab,
            num_vocab, num_vocab, args.d_model,
            args.num_heads, args.dff, 0.1
        ).to(device)
    return net

def load_model(arch, ckpt):
    net = build_model(arch)
    state = torch.load(ckpt, map_location=device)
    net.load_state_dict(state)
    net.eval()
    return net


# ======================================
# 6) build_memory (baseline / sem / robust / gate)
# ======================================
def build_memory_baseline(model, src, n_var, padding_idx, channel):
    channels = Channels()
    src_mask = (src == padding_idx).unsqueeze(-2).float().to(device)

    enc = model.encoder(src, src_mask)
    enc2 = model.channel_encoder(enc)
    Tx  = PowerNormalize(enc2)

    if channel=='AWGN':
        Rx = channels.AWGN(Tx, n_var)
    elif channel=='Rayleigh':
        Rx = channels.Rayleigh(Tx, n_var)
    else:
        Rx = channels.Rician(Tx, n_var)

    dec = model.channel_decoder(Rx)
    return dec, src_mask, src.size(1)


def build_memory_zsplit_sem(model, src, n_var, padding_idx, channel):
    channels = Channels()
    src_mask = (src == padding_idx).unsqueeze(-2).float().to(device)

    enc = model.encoder(src, src_mask)
    z_total, z_dict = model.split_latent(enc, n_var)
    z_sem = z_dict["z_sem"]

    enc2 = model.channel_encoder(z_total)
    Tx   = PowerNormalize(enc2)

    if channel=='AWGN':
        Rx = channels.AWGN(Tx, n_var)
    elif channel=='Rayleigh':
        Rx = channels.Rayleigh(Tx, n_var)
    else:
        Rx = channels.Rician(Tx, n_var)

    dec = model.channel_decoder(Rx)
    return dec + model.sem_weight * z_sem, src_mask, src.size(1)


def build_memory_zsplit_robust(model, src, n_var, padding_idx, channel):
    channels = Channels()
    src_mask = (src == padding_idx).unsqueeze(-2).float().to(device)

    enc = model.encoder(src, src_mask)
    _, z_dict = model.split_latent(enc, n_var)
    z_sem = z_dict["z_sem"]
    z_rob = z_dict["z_rob"]
    z_snr = z_dict["z_snr"]
    z_total = (z_rob + z_snr) / 2.0

    enc2 = model.channel_encoder(z_total)
    Tx   = PowerNormalize(enc2)

    if channel=='AWGN':
        Rx = channels.AWGN(Tx, n_var)
    elif channel=='Rayleigh':
        Rx = channels.Rayleigh(Tx, n_var)
    else:
        Rx = channels.Rician(Tx, n_var)

    dec = model.channel_decoder(Rx)
    return dec + model.sem_weight * z_sem, src_mask, src.size(1)


def build_memory_zsplit_gate(model, src, n_var, padding_idx, channel):
    channels = Channels()
    src_mask = (src == padding_idx).unsqueeze(-2).float().to(device)

    enc = model.encoder(src, src_mask)
    _, z_dict = model.split_latent(enc, n_var)
    z_sem = z_dict["z_sem"]
    z_rob = z_dict["z_rob"]
    z_snr = z_dict["z_snr"]

    gate_logits = model.gate_net(z_snr)
    gate = torch.softmax(gate_logits, dim=-1)
    w_sem = gate[...,0:1]
    w_rob = gate[...,1:2]

    z_total = w_sem*z_sem + w_rob*z_rob

    enc2 = model.channel_encoder(z_total)
    Tx   = PowerNormalize(enc2)

    if channel=='AWGN':
        Rx = channels.AWGN(Tx, n_var)
    elif channel=='Rayleigh':
        Rx = channels.Rayleigh(Tx, n_var)
    else:
        Rx = channels.Rician(Tx, n_var)

    dec = model.channel_decoder(Rx)
    return dec + model.sem_weight * z_sem, src_mask, src.size(1)


def build_memory(arch, model, src, n_var):
    if arch=="baseline":
        return build_memory_baseline(model, src, n_var, pad_idx, args.channel)
    elif arch=="zsplit_sem":
        return build_memory_zsplit_sem(model, src, n_var, pad_idx, args.channel)
    elif arch=="zsplit_robust":
        return build_memory_zsplit_robust(model, src, n_var, pad_idx, args.channel)
    else:
        return build_memory_zsplit_gate(model, src, n_var, pad_idx, args.channel)


# ======================================
# 7) Greedy decoding
# ======================================
def greedy_decode_from_memory(model, memory, src_mask, max_len, start_symbol):
    B = memory.size(0)
    ys = torch.full((B, 1), start_symbol,
                    dtype=torch.long, device=device)

    for _ in range(max_len - 1):
        trg_mask = (ys == pad_idx).unsqueeze(-2).type(torch.FloatTensor).to(device)
        look_ahead = subsequent_mask(ys.size(1)).type(torch.FloatTensor).to(device)
        mask = torch.max(trg_mask, look_ahead)

        dec = model.decoder(ys, memory, mask, src_mask)
        logits = model.dense(dec[:, -1, :])
        next_tok = torch.argmax(logits, dim=-1, keepdim=True)
        ys = torch.cat([ys, next_tok], dim=1)

    return ys


# ======================================
# 8) 평가 함수 (BLEU + SIM)
# ======================================
def eval_bleu_and_sim_vs_snr(arch, ckpt_path, snr_list):
    print(f"\n=== {arch} ===")
    net = load_model(arch, ckpt_path)

    bleu_res = []
    sim_res  = []

    for snr in snr_list:
        n_var = SNR_to_noise(snr)
        refs, hyps = [], []

        with torch.no_grad():
            for sents in test_iterator:
                sents = sents.to(device)

                memory, src_mask, max_len = build_memory(arch, net, sents, n_var)
                outs = greedy_decode_from_memory(
                    net, memory, src_mask, max_len, start_symbol=start_idx
                )

                for src_seq, out_seq in zip(sents, outs):
                    refs.append(seq2text.sequence_to_text(src_seq.tolist()))
                    hyps.append(seq2text.sequence_to_text(out_seq.tolist()))

        # smoothed BLEU
        bleu_scores = compute_bleu_smooth(refs, hyps)
        avg_bleu = float(np.mean(bleu_scores))
        bleu_res.append(avg_bleu)

        # sentence similarity
        sim_scores = sentence_similarity(refs, hyps)
        avg_sim = float(np.mean(sim_scores))
        sim_res.append(avg_sim)

        print(f"SNR={snr:2d} dB | BLEU={avg_bleu:.4f} | SIM={avg_sim:.4f}")

    return bleu_res, sim_res


# ======================================
# 9) 체크포인트 경로 설정 (여기만 수정하면 됨)
# ======================================
arch_ckpts = {
    "baseline":      "checkpoints/checkpoints/baseline_80epoch/checkpoint_best.pth",
    "zsplit_sem":    "checkpoints/checkpoints/zsplit_sem_80epoch/checkpoint_best.pth",
    "zsplit_robust": "checkpoints/checkpoints/zsplit_robust_80epoch/checkpoint_best.pth",
    "zsplit_gate":   "checkpoints/checkpoints/zsplit_gate_80epoch/checkpoint_best.pth",
}

snr_list = [0,5,10,15,20]


# ======================================
# 10) 실제 평가 실행
# ======================================
bleu_results = {}
sim_results  = {}

for arch, ckpt in arch_ckpts.items():
    bleu_results[arch], sim_results[arch] = eval_bleu_and_sim_vs_snr(
        arch, ckpt, snr_list
    )



# ======================================
# 12) 숫자 요약 출력
# ======================================
print("\n=== Summary ===")
for arch in arch_ckpts.keys():
    print(
        f"{arch:12s} | BLEU:",
        ['%.3f'%x for x in bleu_results[arch]],
        "| SIM:",
        ['%.3f'%x for x in sim_results[arch]]
    )

# ===========================================
# 11) Graphs with AUTO Zoom-in (논문용) - baseline vs zsplit_gate only
# ===========================================
markers    = {"baseline":"o","zsplit_sem":"s","zsplit_robust":"D","zsplit_gate":"^"}
linestyles = {"baseline":"-","zsplit_sem":"--","zsplit_robust":":","zsplit_gate":"-"}

# ✅ 여기만 조절하면 됨
plot_archs = ["baseline", "zsplit_gate"]

# --------------------------
# (A) BLEU Full-range Plot
# --------------------------
fig, ax = plt.subplots(figsize=(7,4), dpi=300)
for arch in plot_archs:
    ax.plot(
        snr_list, bleu_results[arch],
        marker=markers.get(arch, "o"),
        linestyle=linestyles.get(arch, "-"),
        label=arch
    )
ax.set_xlabel("SNR (dB)")
ax.set_ylabel("BLEU")
ax.set_title("BLEU vs SNR")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig("bleu_full.pdf", bbox_inches="tight")
plt.show()

# --------------------------
# (C) SIM Full-range Plot
# --------------------------
fig, ax = plt.subplots(figsize=(7,4), dpi=300)
for arch in plot_archs:
    ax.plot(
        snr_list, sim_results[arch],
        marker=markers.get(arch, "o"),
        linestyle=linestyles.get(arch, "-"),
        label=arch
    )
ax.set_xlabel("SNR (dB)")
ax.set_ylabel("Sentence Similarity (cosine)")
ax.set_title("Sentence Similarity vs SNR")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig("sim_full.pdf", bbox_inches="tight")
plt.show()
