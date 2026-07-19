"""GRPO post-training.

For each prompt the policy samples a group of slates, each slate is scored
against what that user actually did next, and the advantage of a slate is its
reward relative to the rest of its own group. No value network and no reward
model: the group is the baseline and the held-out behaviour is the reward.

A KL term against the frozen SFT policy keeps the sampler from collapsing onto
whatever narrow region of the catalog pays best early.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..data.dataset import encode_prompt, user_histories
from ..model.generate import sample_slate, sequence_logprobs
from ..model.trie import CatalogTrie
from ..reward import SlateReward, group_advantages
from ..utils.logging import JsonlLogger, get_logger

logger = get_logger(__name__)


@dataclass
class RolloutPrompt:
    user: int
    history: list[int]
    targets: set[int]


def build_prompts(
    train_df: pd.DataFrame,
    heldout_df: pd.DataFrame,
    max_history_items: int = 50,
) -> list[RolloutPrompt]:
    """One prompt per user active in the held-out window.

    The prompt is that user's history up to the window; the reward looks at
    everything they went on to do inside it.
    """
    past = user_histories(train_df)
    future: dict[int, set[int]] = {}
    for user, item in zip(heldout_df["user"], heldout_df["item"]):
        future.setdefault(int(user), set()).add(int(item))

    prompts = []
    for user, targets in future.items():
        history = [i for i, _ in past.get(user, [])][-max_history_items:]
        prompts.append(RolloutPrompt(user=user, history=history, targets=targets))
    prompts.sort(key=lambda p: p.user)
    return prompts


def reward_visible_mask(prompts: list[RolloutPrompt], n_items: int, cold_mask: np.ndarray) -> np.ndarray:
    """Cold items the reward could ever have paid out on.

    An item is reward-visible if it appears in some prompt's held-out set, i.e.
    if a rollout that surfaced it would have been scored for it. Everything else
    is invisible to the reward for the whole of post-training.
    """
    mask = np.zeros(n_items, dtype=bool)
    for prompt in prompts:
        for item in prompt.targets:
            mask[item] = True
    return mask & cold_mask


class GRPOTrainer:
    def __init__(
        self,
        model,
        trie: CatalogTrie,
        item_tokens: np.ndarray,
        cold_mask: np.ndarray,
        cfg,
        device: str | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.ref_model = copy.deepcopy(model).to(self.device).eval()
        for param in self.ref_model.parameters():
            param.requires_grad_(False)

        self.trie = trie
        self.item_tokens = torch.as_tensor(item_tokens, dtype=torch.long)
        self.cfg = cfg
        self.reward_fn = SlateReward(
            cold_mask=cold_mask,
            cold_lambda=cfg.reward["cold_lambda"],
            cold_cap=cfg.reward["cold_cap"],
            metric_k=cfg.reward["metric_k"],
        )
        self.opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.train["lr"],
            weight_decay=cfg.train["weight_decay"],
        )

    def _lr_at(self, step: int, total: int) -> float:
        warmup = self.cfg.train["warmup_steps"]
        base, floor = self.cfg.train["lr"], self.cfg.train["min_lr"]
        if step < warmup:
            return base * (step + 1) / warmup
        progress = (step - warmup) / max(1, total - warmup)
        return floor + 0.5 * (base - floor) * (1.0 + math.cos(math.pi * progress))

    def rollout(self, prompt: RolloutPrompt) -> tuple[list[dict], np.ndarray]:
        """Sample a group of slates for one prompt and score every one of them."""
        prompt_tokens = encode_prompt(prompt.history, self.item_tokens)
        group = []
        rewards = []
        self.model.eval()
        for _ in range(self.cfg.rollout["group_size"]):
            items, sequences = sample_slate(
                self.model,
                prompt_tokens,
                self.trie,
                slate_size=self.cfg.rollout["slate_size"],
                oversample=self.cfg.rollout["oversample"],
                temperature=self.cfg.rollout["temperature"],
                top_k=self.cfg.rollout["top_k"],
                beam_backfill=self.cfg.rollout["beam_backfill"],
                device=self.device,
            )
            breakdown = self.reward_fn(items, prompt.targets)
            with torch.no_grad():
                old_logprobs = sequence_logprobs(self.model, prompt_tokens, sequences, self.trie, self.device)
                ref_logprobs = sequence_logprobs(self.ref_model, prompt_tokens, sequences, self.trie, self.device)
            group.append(
                {
                    "prompt_tokens": prompt_tokens,
                    "sequences": sequences,
                    "old_logprobs": old_logprobs,
                    "ref_logprobs": ref_logprobs,
                    "reward": breakdown,
                }
            )
            rewards.append(breakdown.total)
        return group, np.asarray(rewards)

    def step(self, prompts: list[RolloutPrompt]) -> dict[str, float]:
        groups, all_rewards, advantages = [], [], []
        for prompt in prompts:
            group, rewards = self.rollout(prompt)
            adv = group_advantages(rewards, normalize=self.cfg.reward["normalize_advantage"])
            groups.append(group)
            all_rewards.append(rewards)
            advantages.append(adv)

        self.model.train()
        self.opt.zero_grad(set_to_none=True)

        total_loss = torch.zeros((), device=self.device)
        total_kl = torch.zeros((), device=self.device)
        clipped = 0.0
        n_rollouts = 0
        eps = self.cfg.train["clip_eps"]

        for group, adv in zip(groups, advantages):
            for rollout, advantage in zip(group, adv):
                if advantage == 0.0:
                    continue
                logprobs = sequence_logprobs(
                    self.model, rollout["prompt_tokens"], rollout["sequences"], self.trie, self.device
                )
                ratio = (logprobs - rollout["old_logprobs"]).exp()
                unclipped = ratio * advantage
                clipped_term = ratio.clamp(1 - eps, 1 + eps) * advantage
                policy_loss = -torch.min(unclipped, clipped_term).mean()

                # k3 estimator: non-negative and lower variance than the naive one
                log_ratio = rollout["ref_logprobs"] - logprobs
                kl = (log_ratio.exp() - log_ratio - 1.0).mean()

                total_loss = total_loss + policy_loss + self.cfg.train["kl_coef"] * kl
                total_kl = total_kl + kl.detach()
                clipped += float(((ratio < 1 - eps) | (ratio > 1 + eps)).float().mean())
                n_rollouts += 1

        if n_rollouts:
            loss = total_loss / n_rollouts
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train["grad_clip"])
            self.opt.step()
        else:
            loss = total_loss

        flat = np.concatenate(all_rewards)
        ndcgs = [r["reward"].ndcg for g in groups for r in g]
        bonuses = [r["reward"].cold_bonus for g in groups for r in g]
        return {
            "loss": float(loss.detach()),
            "reward_mean": float(flat.mean()),
            "reward_std": float(flat.std()),
            "ndcg_mean": float(np.mean(ndcgs)),
            "cold_bonus_mean": float(np.mean(bonuses)),
            "adv_std": float(np.concatenate(advantages).std()),
            "kl": float(total_kl / max(1, n_rollouts)),
            "clip_frac": clipped / max(1, n_rollouts),
            "active_rollouts": n_rollouts,
        }

    def train(
        self,
        prompts: list[RolloutPrompt],
        log_path: str | Path,
        eval_fn=None,
    ) -> dict:
        rng = np.random.default_rng(self.cfg.train["seed"])
        per_step = self.cfg.train["prompts_per_step"]
        steps_per_epoch = len(prompts) // per_step
        total_steps = steps_per_epoch * self.cfg.train["epochs"]

        sink = JsonlLogger(log_path)
        start = time.time()
        step = 0
        best = {"step": -1, "metric": -1.0}

        for epoch in range(1, self.cfg.train["epochs"] + 1):
            order = rng.permutation(len(prompts))
            for i in range(steps_per_epoch):
                lr = self._lr_at(step, total_steps)
                for group in self.opt.param_groups:
                    group["lr"] = lr

                batch = [prompts[j] for j in order[i * per_step : (i + 1) * per_step]]
                stats = self.step(batch)

                if step % self.cfg.train["log_every"] == 0:
                    sink.log(
                        step=step,
                        epoch=epoch,
                        lr=round(lr, 9),
                        elapsed=round(time.time() - start, 1),
                        **{k: round(v, 6) for k, v in stats.items()},
                    )

                if eval_fn is not None and step > 0 and step % self.cfg.train["eval_every_steps"] == 0:
                    metrics = eval_fn(self.model, step)
                    sink.log(step=step, epoch=epoch, eval=True, **metrics)
                    if metrics.get("select_metric", -1.0) > best["metric"]:
                        best = {"step": step, "metric": metrics["select_metric"]}
                step += 1

        sink.close()
        return {"steps": step, "best": best}

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": self.model.state_dict(),
                "cold_lambda": self.cfg.reward["cold_lambda"],
                "cold_cap": self.cfg.reward["cold_cap"],
            },
            path,
        )
