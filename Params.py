from __future__ import annotations

import argparse
from typing import Optional, Sequence


def str2bool(v):
    """
    Robust boolean parser for argparse.
    Supports:
        true / false
        yes / no
        1 / 0
        y / n
        t / f
    """
    if isinstance(v, bool):
        return v
    if v is None:
        return False

    s = str(v).strip().lower()
    if s in {"true", "1", "yes", "y", "t"}:
        return True
    if s in {"false", "0", "no", "n", "f"}:
        return False

    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {v}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Arguments for PPO/FJSP training")

    # ------------------------------------------------------------------
    # device
    # ------------------------------------------------------------------
    parser.add_argument("--device", type=str, default="cuda", help="Device name, e.g. cuda or cpu")

    # ------------------------------------------------------------------
    # environment
    # ------------------------------------------------------------------
    parser.add_argument("--n_j", type=int, default=6, help="Number of jobs in an instance")
    parser.add_argument("--n_m", type=int, default=6, help="Number of machines in an instance")

    parser.add_argument("--rewardscale", type=float, default=0.0, help="Reward scale for positive rewards")
    parser.add_argument(
        "--init_quality_flag",
        type=str2bool,
        default=False,
        help="If True, initial quality is forced to 0; otherwise use LB-based initialization",
    )

    parser.add_argument("--low", type=int, default=-1, help="Lower bound of generated processing times")
    parser.add_argument(
        "--high",
        type=int,
        default=1,
        help="Upper bound reference of generated processing times / legacy sentinel reference",
    )

    parser.add_argument("--np_seed_train", type=int, default=200, help="Numpy seed for training")
    parser.add_argument("--np_seed_validation", type=int, default=200, help="Numpy seed for validation")
    parser.add_argument("--torch_seed", type=int, default=600, help="Torch seed")

    parser.add_argument(
        "--et_normalize_coef",
        type=float,
        default=1.0,
        help="Normalization constant for end-time based node features",
    )
    parser.add_argument(
        "--wkr_normalize_coef",
        type=float,
        default=1.0,
        help="Normalization constant for work-remaining features",
    )

    # ------------------------------------------------------------------
    # network
    # ------------------------------------------------------------------
    parser.add_argument("--num_layers", type=int, default=3, help="Number of GNN/feature extraction layers")
    parser.add_argument("--neighbor_pooling_type", type=str, default="average", help="Neighbor pooling type")
    parser.add_argument("--graph_pool_type", type=str, default="average", help="Graph pooling type")
    parser.add_argument("--input_dim", type=int, default=2, help="Raw node feature dimension")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dim of feature extractor")
    parser.add_argument(
        "--num_mlp_layers_feature_extract",
        type=int,
        default=3,
        help="Number of MLP layers in feature extractor",
    )
    parser.add_argument("--num_mlp_layers_actor", type=int, default=2, help="Number of MLP layers in actor")
    parser.add_argument("--hidden_dim_actor", type=int, default=32, help="Hidden dim in actor")
    parser.add_argument("--num_mlp_layers_critic", type=int, default=2, help="Number of MLP layers in critic")
    parser.add_argument("--hidden_dim_critic", type=int, default=32, help="Hidden dim in critic")

    # ------------------------------------------------------------------
    # PPO
    # ------------------------------------------------------------------
    parser.add_argument("--ppo_step", type=int, default=2, help="Number of PPO rollout/update steps")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--num_ins", type=int, default=12800, help="Number of instances")
    parser.add_argument("--num_envs", type=int, default=4, help="Number of parallel environments")
    parser.add_argument("--max_updates", type=int, default=10000, help="Maximum PPO updates")

    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--decayflag", type=str2bool, default=False, help="Whether to enable LR decay")
    parser.add_argument("--decay_step_size", type=int, default=2000, help="LR decay step size")
    parser.add_argument("--decay_ratio", type=float, default=0.96, help="LR decay ratio")

    parser.add_argument("--gamma", type=float, default=1.0, help="Discount factor")
    parser.add_argument("--k_epochs", type=int, default=3, help="Number of PPO epochs per update")
    parser.add_argument("--eps_clip", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--vloss_coef", type=float, default=1.0, help="Critic loss coefficient")
    parser.add_argument("--ploss_coef", type=float, default=2.0, help="Policy loss coefficient")
    parser.add_argument("--entloss_coef", type=float, default=0.01, help="Entropy loss coefficient")

    # ------------------------------------------------------------------
    # Agent_C / candidate control
    # 这些字段已经在你优化后的环境代码中被使用，建议统一放到配置里
    # ------------------------------------------------------------------
    parser.add_argument(
        "--agent_c_pairs_per_rule",
        type=int,
        default=4,
        help="Maximum candidate pairs kept per compound rule",
    )
    parser.add_argument(
        "--agent_c_topk_jobs",
        type=int,
        default=2,
        help="Top-k jobs selected by each job rule",
    )
    parser.add_argument(
        "--agent_c_topk_machines",
        type=int,
        default=2,
        help="Top-k machines selected by each machine rule",
    )
    parser.add_argument(
        "--agent_c_extra_explore_pairs",
        type=int,
        default=2,
        help="Extra exploration pairs added to the candidate pool",
    )
    parser.add_argument(
        "--agent_c_fixed_global_feat",
        type=str2bool,
        default=True,
        help="If True, build fixed-size global features that are invariant to dataset n_j/n_m",
    )
    parser.add_argument(
        "--agent_c_two_stage_refine_enabled",
        type=str2bool,
        default=True,
        help="Enable safety-oriented two-stage candidate refine for Agent_C",
    )
    parser.add_argument(
        "--agent_c_refine_max_pairs_per_op",
        type=int,
        default=2,
        help="Max pairs per operation after refine",
    )
    parser.add_argument(
        "--agent_c_refine_keep_global_best_ect",
        type=int,
        default=1,
        help="Whether to keep global best ECT pair (0/1)",
    )
    parser.add_argument(
        "--agent_c_refine_keep_global_best_pt",
        type=int,
        default=1,
        help="Whether to keep global best PT pair (0/1)",
    )
    parser.add_argument(
        "--agent_c_refine_score_w_eet",
        type=float,
        default=1.0,
        help="Weight of normalized ECT in refine score",
    )
    parser.add_argument(
        "--agent_c_refine_score_w_pt",
        type=float,
        default=0.3,
        help="Weight of normalized PT in refine score",
    )
    parser.add_argument(
        "--agent_c_refine_score_w_queue",
        type=float,
        default=0.2,
        help="Weight of normalized machine queue length in refine score",
    )
    parser.add_argument(
        "--agent_c_refine_score_w_support",
        type=float,
        default=0.5,
        help="Weight of normalized rule support in refine score",
    )
    parser.add_argument(
        "--agent_c_refine_explore_bonus",
        type=float,
        default=0.1,
        help="Explore-source bonus subtracted from refine score",
    )
    parser.add_argument("--agent_c_candidate_v2_enabled", type=str2bool, default=True)
    parser.add_argument("--agent_c_critical_pairs_enabled", type=str2bool, default=True)
    parser.add_argument("--agent_c_critical_pairs_per_op", type=int, default=2)
    parser.add_argument("--agent_c_refine_max_pairs_per_machine", type=int, default=0)
    parser.add_argument("--agent_c_refine_global_reserve_size", type=int, default=2)
    parser.add_argument("--agent_c_refine_diversity_min_machines", type=int, default=2)
    parser.add_argument("--agent_c_refine_keep_explore_min", type=int, default=1)
    parser.add_argument("--agent_c_candidate_debug_print", type=str2bool, default=False)

    # ------------------------------------------------------------------
    # reward shaping
    # ------------------------------------------------------------------
    parser.add_argument(
        "--reward_terminal_coef",
        type=float,
        default=1.0,
        help="Coefficient of terminal makespan reward",
    )
    parser.add_argument(
        "--reward_lb_coef",
        type=float,
        default=1.0,
        help="Coefficient of LB-improvement reward",
    )
    parser.add_argument(
        "--reward_use_terminal",
        type=str2bool,
        default=True,
        help="Whether to use terminal makespan reward",
    )

    return parser


def get_configs(args: Optional[Sequence[str]] = None):
    """
    Parse configuration safely.

    Parameters
    ----------
    args:
        - None: parse from command line
        - list/tuple: parse from provided argument list
    """
    parser = build_parser()
    return parser.parse_args(args=args)


# 为保持你现有代码中 `from Params import configs` 的兼容性，保留这一行
configs = get_configs()


if __name__ == "__main__":
    print(configs)