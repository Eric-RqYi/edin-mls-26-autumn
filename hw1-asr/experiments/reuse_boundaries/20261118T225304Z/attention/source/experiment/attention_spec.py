"""Two stages: attention ablations, then a GEMM-dispatch x attention factorial."""
FLAGS = "-O3 -std=c++17 -lineinfo -diag-suppress 177 -Xptxas=-v"
WARPS = (1, 2, 4, 8, 16)
ATTENTION_FIELDS = ("case", "batch", "num_heads", "num_kv_heads", "seq_q", "seq_k",
                    "head_dim", "causal", "q_range", "pattern")
# pattern: 0=random, 1=zero queries, 2=increasing scores, 3=all-negative scores.
ATTENTION_CASES = [
    ("tiny", 1, 1, 1, 4, 5, 32, 0, 1, 0),
    ("odd_gqa", 2, 4, 2, 17, 37, 64, 0, 2, 0),
    ("peaky", 1, 2, 2, 9, 100, 32, 0, 8, 0),
    ("very_peaky", 1, 2, 1, 33, 65, 128, 0, 16, 0),
    ("late_max", 1, 2, 2, 5, 97, 64, 0, 1, 2),
    ("negative_scores", 1, 1, 1, 7, 65, 32, 0, 1, 3),
    ("uniform", 1, 4, 2, 11, 71, 128, 0, 1, 1),
    ("causal", 1, 4, 2, 70, 70, 128, 1, 4, 0),
    ("cache_q5", 2, 4, 2, 5, 37, 32, 1, 4, 0),
    ("cache_q1", 1, 16, 4, 1, 71, 128, 1, 4, 0),
    ("masked_prefix", 1, 2, 1, 33, 5, 64, 1, 4, 0),
    ("dim96", 1, 3, 1, 19, 35, 96, 0, 4, 0),
    ("decoder", 1, 16, 4, 64, 64, 128, 1, 4, 0),
    ("encoder", 1, 20, 20, 1500, 1500, 64, 0, 4, 0),
]


def configs(stage, selected_warps=4, selected_online=1):
    if stage not in ("sweep", "factorial") or selected_warps not in WARPS or selected_online not in (0, 1):
        raise ValueError("Invalid stage/attention configuration")
    if stage == "sweep":
        rows = [dict(name="w4_standard", warps=4, online=0, hybrid=0)]
        for w in WARPS:
            for online, label in ((0, "standard"), (1, "direct")):
                if w == 4 and online == 0:
                    continue
                rows.append(dict(name=f"w{w}_{label}", warps=w, online=online, hybrid=0))
        return rows
    return [
        dict(name="base", warps=4, online=0, hybrid=0),
        dict(name="hybrid", warps=4, online=0, hybrid=1),
        dict(name="attention", warps=selected_warps, online=selected_online, hybrid=0),
        dict(name="combined", warps=selected_warps, online=selected_online, hybrid=1),
    ]


def variant_flags(v):
    return (f"{FLAGS} -DASR_TASK3_M1_ENABLED=1 -DASR_TASK3_M1_WARPS=4 "
            f"-DASR_TASK3_GEMM_IMPL=1 -DASR_TASK3_TILE=16 -DASR_TASK3_RM=4 -DASR_TASK3_RN=4 "
            f"-DASR_TASK3_HYBRID_KV={v['hybrid']} -DASR_ATTN_WARPS={v['warps']} "
            f"-DASR_ATTN_ONLINE_MODE={v['online']}")
