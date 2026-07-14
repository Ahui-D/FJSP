import numpy as np

from FJSP_Env_Agent import FJSP
from uniform_instance import uni_instance_gen


def pair_metrics(env, state, pairs):
    if len(pairs) == 0:
        return {
            "ops": 0,
            "global_best_ect": None,
            "global_best_pt": None,
        }

    by_op = {}
    infos = []
    for op_id, mch_id in pairs:
        op_id = int(op_id)
        mch_id = int(mch_id)
        row, col = env._task_to_row_col(op_id, batch_idx=0)
        pt = float(state["dur"][row, col, mch_id])
        ect = max(float(state["job_time"][row]), float(state["mch_time"][mch_id])) + pt
        infos.append((op_id, mch_id, ect, pt))
        by_op.setdefault(op_id, 0)
        by_op[op_id] += 1

    global_best_ect = min(infos, key=lambda x: (x[2], x[3], x[0], x[1]))
    global_best_pt = min(infos, key=lambda x: (x[3], x[2], x[0], x[1]))
    return {
        "ops": len(by_op),
        "counts_by_op": by_op,
        "global_best_ect": (int(global_best_ect[0]), int(global_best_ect[1])),
        "global_best_pt": (int(global_best_pt[0]), int(global_best_pt[1])),
    }


def main():
    n_j, n_m = 10, 5
    episodes = 30
    seed0 = 20260410

    each_job = np.asarray([[n_m] * n_j], dtype=np.int32)
    env = FJSP(n_j=n_j, n_m=n_m, EachJob_num_operation=each_job)

    reductions = []
    refined_counts = []
    original_counts = []
    op_coverage_ratios = []
    op_with_alt_ratios = []
    keep_global_ect = 0
    keep_global_pt = 0
    fallback_like = 0
    total_states = 0

    for ep in range(episodes):
        data = uni_instance_gen(n_j=n_j, n_m=n_m, low=1, high=100, seed=seed0 + ep).astype(np.float32)
        env.reset(data[np.newaxis, ...], rule="FIFO_SPT")

        done = False
        while not done:
            state = env._get_runtime_state(batch_idx=0, copy_arrays=False)

            env.two_stage_refine_enabled = False
            pairs_raw, _ = env.get_rule_candidate_pairs(batch_idx=0)

            env.two_stage_refine_enabled = True
            pairs_refined, _ = env.get_rule_candidate_pairs(batch_idx=0)

            raw_n = len(pairs_raw)
            ref_n = len(pairs_refined)
            if raw_n > 0:
                reductions.append((raw_n - ref_n) / float(raw_n))
                op_raw = len({int(a) for a, _ in pairs_raw})
                op_ref = len({int(a) for a, _ in pairs_refined})
                op_coverage_ratios.append(op_ref / float(max(op_raw, 1)))

                by_op_ref = {}
                for a, _ in pairs_refined:
                    by_op_ref.setdefault(int(a), 0)
                    by_op_ref[int(a)] += 1
                if len(by_op_ref) > 0:
                    op_with_alt = sum(1 for _, c in by_op_ref.items() if c >= 2)
                    op_with_alt_ratios.append(op_with_alt / float(len(by_op_ref)))

            original_counts.append(raw_n)
            refined_counts.append(ref_n)

            raw_m = pair_metrics(env, state, pairs_raw)
            ref_set = {(int(a), int(b)) for a, b in pairs_refined}
            if raw_m["global_best_ect"] in ref_set:
                keep_global_ect += 1
            if raw_m["global_best_pt"] in ref_set:
                keep_global_pt += 1

            if ref_n <= 1:
                fallback_like += 1

            total_states += 1

            if ref_n == 0:
                break

            best = None
            for op_id, mch_id in pairs_refined:
                row, col = env._task_to_row_col(int(op_id), batch_idx=0)
                pt = float(state["dur"][row, col, int(mch_id)])
                ect = max(float(state["job_time"][row]), float(state["mch_time"][int(mch_id)])) + pt
                key = (ect, pt, int(op_id), int(mch_id))
                if best is None or key < best[0]:
                    best = (key, int(op_id), int(mch_id))

            _, op, mch = best
            env.step_with_pair(op_id=op, mch_id=mch, batch_idx=0)
            done = env.done(batch_idx=0)

    def p(arr, q):
        if len(arr) == 0:
            return float("nan")
        return float(np.percentile(np.asarray(arr, dtype=np.float32), q))

    print("=" * 72)
    print("Two-stage refine evaluation summary")
    print(f"states={total_states}, episodes={episodes}, size={n_j}x{n_m}")
    print("-" * 72)
    print(f"original_count: mean={np.mean(original_counts):.3f}, p50={p(original_counts,50):.1f}, p90={p(original_counts,90):.1f}")
    print(f"refined_count : mean={np.mean(refined_counts):.3f}, p50={p(refined_counts,50):.1f}, p90={p(refined_counts,90):.1f}")
    print(f"reduction_ratio: mean={np.mean(reductions):.3%}, p50={p(reductions,50):.3%}, p90={p(reductions,90):.3%}")
    print(f"op_coverage_ratio(ref/raw): mean={np.mean(op_coverage_ratios):.3%}, p50={p(op_coverage_ratios,50):.3%}")
    print(f"op_with_alternative_ratio(>=2 pairs/op): mean={np.mean(op_with_alt_ratios):.3%}, p50={p(op_with_alt_ratios,50):.3%}")
    print(f"keep_global_best_ect: {keep_global_ect}/{total_states} ({keep_global_ect/float(max(total_states,1)):.3%})")
    print(f"keep_global_best_pt : {keep_global_pt}/{total_states} ({keep_global_pt/float(max(total_states,1)):.3%})")
    print(f"very_small_pool(refined<=1): {fallback_like}/{total_states} ({fallback_like/float(max(total_states,1)):.3%})")
    print("=" * 72)


if __name__ == "__main__":
    main()
