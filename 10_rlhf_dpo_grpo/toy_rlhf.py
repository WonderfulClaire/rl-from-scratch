"""10 · 玩具语言模型上的 PPO-RLHF / DPO / GRPO 三算法对照.

小世界设定 (README 第 5 节):
  词表: a b c d e + EOS, 生成长度 <= 8 的字符串.
  真实奖励 (上帝视角): +1/每个 'a', -0.5/每处相邻重复, 模拟"人类偏好".
  参考模型 pi_ref: 在均匀随机串上做过 SFT 的 GRU LM.

三条路线 (同一参考模型出发):
  1) PPO-RLHF: 偏好对 -> 训练 RM (公式 2) -> 带每 token KL 惩罚的策略梯度优化 (公式 3)
  2) DPO:      偏好对 -> 直接最小化 L_DPO (公式 6)
  3) GRPO:     可验证奖励 -> 组内标准化优势 (公式 7) + clip (公式 8)

评估: 各自训练后采样 500 条, 比较真实奖励均值与对 pi_ref 的 KL.

运行: python 10_rlhf_dpo_grpo/toy_rlhf.py
"""

import copy

import numpy as np
import torch
import torch.nn as nn

VOCAB = ["a", "b", "c", "d", "e", "<eos>"]
V, EOS = len(VOCAB), len(VOCAB) - 1
MAX_LEN = 8


def true_reward(tokens):
    """上帝奖励: 'a'(id=0) 越多越好, 相邻重复扣分."""
    toks = [t for t in tokens if t != EOS]
    r = sum(1.0 for t in toks if t == 0)
    r -= 0.5 * sum(1.0 for i in range(1, len(toks)) if toks[i] == toks[i - 1])
    return r


class TinyLM(nn.Module):
    """GRU 语言模型: 无条件生成 (prompt 恒为 BOS)."""

    def __init__(self, hidden=64):
        super().__init__()
        self.emb = nn.Embedding(V + 1, hidden)  # +1: BOS = V
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.head = nn.Linear(hidden, V)

    def logits(self, seq):
        """seq: (B, T) 含 BOS 前缀. 返回每步对下一 token 的 logits."""
        h, _ = self.gru(self.emb(seq))
        return self.head(h)

    @torch.no_grad()
    def sample(self, batch, temperature=1.0):
        """采样 batch 条串, 返回 (tokens_list, padded_tensor)."""
        seq = torch.full((batch, 1), V, dtype=torch.long)  # BOS
        alive = torch.ones(batch, dtype=torch.bool)
        for _ in range(MAX_LEN):
            logit = self.logits(seq)[:, -1] / temperature
            nxt = torch.distributions.Categorical(logits=logit).sample()
            nxt[~alive] = EOS
            seq = torch.cat([seq, nxt.unsqueeze(1)], dim=1)
            alive &= nxt != EOS
        return seq  # (B, 1+MAX_LEN), 含 BOS

    def seq_logprob(self, seq):
        """整条串的 log pi(y) (对 EOS 之后的 padding 不计)."""
        logits = self.logits(seq[:, :-1])
        lp = torch.log_softmax(logits, dim=-1)
        tgt = seq[:, 1:]
        tok_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)  # (B, T)
        mask = make_mask(tgt)
        return (tok_lp * mask).sum(-1), tok_lp, mask


def make_mask(tgt):
    """有效 token 掩码: 到第一个 EOS 为止 (含 EOS 本身)."""
    B, T = tgt.shape
    mask = torch.ones(B, T)
    for b in range(B):
        eos_pos = (tgt[b] == EOS).nonzero()
        if len(eos_pos):
            mask[b, eos_pos[0, 0] + 1:] = 0.0
    return mask


class RewardModel(nn.Module):
    """序列级奖励模型 r_psi(y)."""

    def __init__(self, hidden=64):
        super().__init__()
        self.emb = nn.Embedding(V + 1, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, seq):
        h, _ = self.gru(self.emb(seq))
        return self.head(h[:, -1]).squeeze(-1)


# ---------------------------------------------------------------- 准备工作
def pretrain_ref(steps=800, batch=64, seed=0):
    """SFT: 在均匀随机串上做 MLE, 得到参考模型."""
    torch.manual_seed(seed)
    lm = TinyLM()
    opt = torch.optim.Adam(lm.parameters(), lr=1e-3)
    for _ in range(steps):
        lens = np.random.randint(3, MAX_LEN, size=batch)
        seqs = torch.full((batch, 1 + MAX_LEN), EOS, dtype=torch.long)
        seqs[:, 0] = V
        for b in range(batch):
            seqs[b, 1:1 + lens[b]] = torch.randint(0, V - 1, (lens[b],))
        logp, _, _ = lm.seq_logprob(seqs)
        loss = -logp.mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return lm


def make_pref_data(ref, n_pairs=2000, seed=0):
    """用参考模型采样对儿, 按真实奖励标注偏好 (模拟人类标注)."""
    torch.manual_seed(seed)
    ys = ref.sample(n_pairs * 2)
    rs = np.array([true_reward(list(s[1:].numpy())) for s in ys])
    pairs = []
    for i in range(n_pairs):
        y1, y2, r1, r2 = ys[2 * i], ys[2 * i + 1], rs[2 * i], rs[2 * i + 1]
        if r1 == r2:
            continue
        w, l = (y1, y2) if r1 > r2 else (y2, y1)
        pairs.append((w, l))
    return pairs


def evaluate(lm, ref, n=500):
    """真实奖励均值 + 对参考模型的序列级 KL 估计."""
    seqs = lm.sample(n)
    rewards = [true_reward(list(s[1:].numpy())) for s in seqs]
    lp_pi, _, _ = lm.seq_logprob(seqs)
    lp_ref, _, _ = ref.seq_logprob(seqs)
    kl = float((lp_pi - lp_ref).mean())  # E_pi[log pi - log ref]
    return float(np.mean(rewards)), kl


# ---------------------------------------------------------------- 1) PPO-RLHF
def train_rm(pairs, seed=0, epochs=3, batch=64):
    """公式 (2): -log sigma(r_w - r_l)."""
    torch.manual_seed(seed)
    rm = RewardModel()
    opt = torch.optim.Adam(rm.parameters(), lr=1e-3)
    for _ in range(epochs):
        np.random.shuffle(pairs)
        for i in range(0, len(pairs) - batch, batch):
            w = torch.stack([p[0] for p in pairs[i:i + batch]])
            l = torch.stack([p[1] for p in pairs[i:i + batch]])
            loss = -torch.log(torch.sigmoid(rm(w) - rm(l)) + 1e-8).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return rm


def run_ppo_rlhf(ref, rm, iters=150, batch=64, beta=0.05, lr=1e-4, seed=0):
    """公式 (3) 的简化 PPO: 序列级优势 = RM 分数 - KL 惩罚 - 组均值基线.

    教学简化: 单次更新 (不做多 epoch clip), 本质是带 KL 整形奖励的
    REINFORCE + 均值基线; PPO 的完整 clip 机制见 07 章与下方 GRPO.
    """
    torch.manual_seed(seed)
    lm = copy.deepcopy(ref)
    opt = torch.optim.Adam(lm.parameters(), lr=lr)
    for _ in range(iters):
        seqs = lm.sample(batch)
        with torch.no_grad():
            r_rm = rm(seqs)
            lp_ref, _, _ = ref.seq_logprob(seqs)
        lp_pi, _, _ = lm.seq_logprob(seqs)
        # 整形奖励: RM 分数 - beta * (log pi - log ref)  [序列级 KL 惩罚]
        shaped = r_rm - beta * (lp_pi.detach() - lp_ref)
        adv = shaped - shaped.mean()
        loss = -(lp_pi * adv).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return lm


# ---------------------------------------------------------------- 2) DPO
def run_dpo(ref, pairs, epochs=3, batch=64, beta=0.1, lr=1e-4, seed=0):
    """公式 (6): -log sigma(beta*[log pi/ref (y_w) - log pi/ref (y_l)])."""
    torch.manual_seed(seed)
    lm = copy.deepcopy(ref)
    opt = torch.optim.Adam(lm.parameters(), lr=lr)
    for _ in range(epochs):
        np.random.shuffle(pairs)
        for i in range(0, len(pairs) - batch, batch):
            w = torch.stack([p[0] for p in pairs[i:i + batch]])
            l = torch.stack([p[1] for p in pairs[i:i + batch]])
            lp_w, _, _ = lm.seq_logprob(w)
            lp_l, _, _ = lm.seq_logprob(l)
            with torch.no_grad():
                lp_w_ref, _, _ = ref.seq_logprob(w)
                lp_l_ref, _, _ = ref.seq_logprob(l)
            margin = beta * ((lp_w - lp_w_ref) - (lp_l - lp_l_ref))
            loss = -torch.log(torch.sigmoid(margin) + 1e-8).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return lm


# ---------------------------------------------------------------- 3) GRPO
def run_grpo(ref, iters=150, n_prompts=8, group=8, clip_eps=0.2,
             beta=0.02, lr=1e-4, seed=0):
    """公式 (7)(8): 组内标准化优势 + PPO-clip + KL 正则 (可验证奖励)."""
    torch.manual_seed(seed)
    lm = copy.deepcopy(ref)
    opt = torch.optim.Adam(lm.parameters(), lr=lr)
    for _ in range(iters):
        # 无条件生成 => "同一 prompt" 即 BOS; 采 n_prompts*group 条, 按组算优势
        seqs = lm.sample(n_prompts * group)
        rewards = torch.tensor([true_reward(list(s[1:].numpy())) for s in seqs],
                               dtype=torch.float32).view(n_prompts, group)
        adv = (rewards - rewards.mean(dim=1, keepdim=True)) / \
            (rewards.std(dim=1, keepdim=True) + 1e-4)          # 公式 (7)
        adv = adv.flatten()

        with torch.no_grad():
            _, tok_lp_old, mask = lm.seq_logprob(seqs)
            _, tok_lp_ref, _ = ref.seq_logprob(seqs)
        _, tok_lp, _ = lm.seq_logprob(seqs)

        rho = torch.exp(tok_lp - tok_lp_old)                    # 每 token 比值
        a = adv.unsqueeze(1)
        surr = torch.min(rho * a, torch.clamp(rho, 1 - clip_eps, 1 + clip_eps) * a)
        # k3 KL 估计器 (Schulman): pi_ref/pi - log(pi_ref/pi) - 1, 逐 token
        log_ratio_ref = tok_lp_ref - tok_lp
        kl = torch.exp(log_ratio_ref) - log_ratio_ref - 1.0
        per_tok = (surr - beta * kl) * mask
        loss = -(per_tok.sum(1) / mask.sum(1).clamp(min=1)).mean()  # 公式 (8)
        opt.zero_grad(); loss.backward(); opt.step()
    return lm


def main():
    torch.manual_seed(0); np.random.seed(0)
    print("准备: SFT 参考模型 + 偏好数据 ...")
    ref = pretrain_ref()
    pairs = make_pref_data(ref)
    r0, _ = evaluate(ref, ref)
    print(f"参考模型: 真实奖励均值 {r0:.3f}, 偏好对 {len(pairs)} 组\n")

    print("1) PPO-RLHF: 训练 RM -> KL 惩罚下优化 ...")
    rm = train_rm(list(pairs))
    lm_ppo = run_ppo_rlhf(ref, rm)
    r_ppo, kl_ppo = evaluate(lm_ppo, ref)

    print("2) DPO: 偏好对直接优化 ...")
    lm_dpo = run_dpo(ref, list(pairs))
    r_dpo, kl_dpo = evaluate(lm_dpo, ref)

    print("3) GRPO: 可验证奖励 + 组内基线 ...")
    lm_grpo = run_grpo(ref)
    r_grpo, kl_grpo = evaluate(lm_grpo, ref)

    print("\n===== 结果对比 (真实奖励越高越好, KL 是对齐税) =====")
    print(f"{'模型':14s} {'真实奖励':>8s} {'KL(pi||ref)':>12s}")
    print(f"{'参考(SFT)':14s} {r0:8.3f} {0.0:12.3f}")
    print(f"{'PPO-RLHF':14s} {r_ppo:8.3f} {kl_ppo:12.3f}")
    print(f"{'DPO':14s} {r_dpo:8.3f} {kl_dpo:12.3f}")
    print(f"{'GRPO':14s} {r_grpo:8.3f} {kl_grpo:12.3f}")

    print("\n各模型样例 (前 5 条):")
    for name, m in [("ref", ref), ("ppo", lm_ppo), ("dpo", lm_dpo), ("grpo", lm_grpo)]:
        seqs = m.sample(5)
        strs = ["".join(VOCAB[t] for t in s[1:] if t != EOS) for s in seqs]
        print(f"  {name:5s}: {strs}")

    ok = min(r_ppo, r_dpo, r_grpo) > r0
    print(f"\n三种方法真实奖励均高于参考模型: {'✓' if ok else '✗'}")


if __name__ == "__main__":
    main()
