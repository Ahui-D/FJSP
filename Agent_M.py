from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class AgentMConfig:
    mode: str = "hard"  # hard | dynamic | dynamic_fallback

    hard_topk: int = 2

    dynamic_k_base: int = 2
    dynamic_entropy_gain: float = 1.0
    dynamic_k_max: int = 4

    # pair ranking score = w_eet * EET + w_pt * PT + w_queue * queue
    w_eet: float = 1.0
    w_pt: float = 0.35
    w_queue: float = 0.12

    # fallback controls
    fallback_entropy_threshold: float = 0.95
    fallback_entropy_extra_k: int = 1
    fallback_quality_gap_threshold: float = 0.08
    fallback_quality_extra_k: int = 1
    fallback_keep_best_ect: bool = True


class Agent_M:
    def __init__(self, config: AgentMConfig):
        self.config = config

    def _compute_k(self, entropy: float, n_candidates: int) -> int:
        cfg = self.config
        if self.config.mode == "hard":
            k = int(cfg.hard_topk)
        else:
            k = int(round(float(cfg.dynamic_k_base) + float(cfg.dynamic_entropy_gain) * float(max(entropy, 0.0))))
            k = min(k, int(cfg.dynamic_k_max))
        return int(max(1, min(int(n_candidates), k)))

    def _pair_score(self, pair_feat_row: np.ndarray) -> float:
        eet = float(pair_feat_row[4]) if pair_feat_row.shape[0] > 4 else 0.0
        pt = float(pair_feat_row[0]) if pair_feat_row.shape[0] > 0 else 0.0
        queue = float(pair_feat_row[19]) if pair_feat_row.shape[0] > 19 else 0.0
        return float(self.config.w_eet * eet + self.config.w_pt * pt + self.config.w_queue * queue)

    def select_pair_indices(
        self,
        obs: Dict[str, object],
        selected_ops: List[int],
        c_entropy: float,
    ) -> Dict[str, object]:
        pairs = list(obs.get("candidate_pairs", []))
        pair_feat = np.asarray(obs.get("pair_feat", []), dtype=np.float32)
        selected_set = {int(x) for x in selected_ops}

        if len(pairs) == 0:
            return {
                "keep_indices": [],
                "k_used": 0,
                "quality_gap": 0.0,
                "fallback_triggered": 0.0,
            }

        per_op_idx: Dict[int, List[int]] = {}
        for idx, (op_id, _mch_id) in enumerate(pairs):
            op_id = int(op_id)
            if op_id not in selected_set:
                continue
            per_op_idx.setdefault(op_id, []).append(int(idx))

        keep_indices: List[int] = []
        quality_gaps: List[float] = []
        fallback_triggered = 0.0
        used_k: List[int] = []

        for op_id, idxs in per_op_idx.items():
            ranked = sorted(
                idxs,
                key=lambda i: (
                    self._pair_score(pair_feat[i]) if pair_feat.ndim == 2 and pair_feat.shape[0] > i else float(i),
                    int(pairs[i][1]),
                ),
            )

            k = self._compute_k(entropy=float(c_entropy), n_candidates=len(ranked))
            chosen = ranked[:k]
            used_k.append(k)

            if pair_feat.ndim == 2 and pair_feat.shape[0] > 0 and pair_feat.shape[1] > 4:
                op_eet = [float(pair_feat[i, 4]) for i in ranked]
                best_full = float(np.min(op_eet)) if len(op_eet) > 0 else 0.0
                best_keep = float(np.min([float(pair_feat[i, 4]) for i in chosen])) if len(chosen) > 0 else best_full
            else:
                best_full = 0.0
                best_keep = 0.0

            gap = max(0.0, (best_keep - best_full) / max(abs(best_full), 1e-6))
            quality_gaps.append(gap)

            if self.config.mode == "dynamic_fallback":
                extra = 0
                if float(c_entropy) > float(self.config.fallback_entropy_threshold):
                    extra += int(max(0, self.config.fallback_entropy_extra_k))
                if gap > float(self.config.fallback_quality_gap_threshold):
                    extra += int(max(0, self.config.fallback_quality_extra_k))
                if extra > 0:
                    fallback_triggered = 1.0
                    chosen = ranked[: min(len(ranked), len(chosen) + extra)]

                if self.config.fallback_keep_best_ect and len(ranked) > 0:
                    best_idx = ranked[0]
                    if best_idx not in chosen:
                        chosen.append(best_idx)

            keep_indices.extend(chosen)

        keep_indices = sorted(set(int(i) for i in keep_indices))
        return {
            "keep_indices": keep_indices,
            "k_used": float(np.mean(used_k)) if len(used_k) > 0 else 0.0,
            "quality_gap": float(np.mean(quality_gaps)) if len(quality_gaps) > 0 else 0.0,
            "fallback_triggered": float(fallback_triggered),
        }


__all__ = [
    "AgentMConfig",
    "Agent_M",
]
