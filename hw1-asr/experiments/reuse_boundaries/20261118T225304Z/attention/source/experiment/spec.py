"""Fixed experiment factors and representative cases (no CUDA dependencies)."""
BASELINE = "tiled16"
FLAGS = "-O3 -std=c++17 -lineinfo -diag-suppress 177 -Xptxas=-v"
VARIANTS = {
    "tiled16": dict(impl=0, tile=16, rm=2, rn=2),
    "tiled8": dict(impl=0, tile=8, rm=2, rn=2),
    "tiled32": dict(impl=0, tile=32, rm=2, rn=2),
    "rb2x2": dict(impl=1, tile=16, rm=2, rn=2),
    "rb2x4": dict(impl=1, tile=16, rm=2, rn=4),
    "rb4x4": dict(impl=1, tile=16, rm=4, rn=4),
}
CASES = [
    # case, M, N, K, bias, act (1 = exact GELU)
    ("m1_tail_control", 1, 33, 33, 1, 1),
    ("m1_lm_head_control", 1, 59264, 2048, 0, 0),
    ("edge_m2", 2, 5, 7, 0, 0),
    ("edge_bias_gelu", 3, 7, 33, 1, 1),
    ("edge_odd", 37, 100, 33, 1, 0),
    ("edge_gelu", 45, 77, 129, 1, 1),
    ("edge_tiles", 17, 65, 31, 0, 1),
    ("edge_short_k", 65, 33, 7, 1, 0),
    ("edge_k1", 31, 17, 1, 0, 0),
    ("decoder_q59", 59, 2048, 2048, 0, 0),
    ("decoder_q64", 64, 2048, 2048, 0, 0),
    ("decoder_q71", 71, 2048, 2048, 0, 0),
    ("decoder_kv64", 64, 512, 2048, 0, 0),
    ("decoder_gate64", 64, 6144, 2048, 0, 0),
    ("decoder_down64", 64, 2048, 6144, 0, 0),
    ("encoder_q", 1500, 1280, 1280, 1, 0),
    ("encoder_fc1", 1500, 5120, 1280, 1, 1),
    ("encoder_fc2", 1500, 1280, 5120, 1, 0),
    ("conv1", 3000, 1280, 384, 1, 1),
    ("conv2", 1500, 1280, 3840, 1, 1),
    ("projector1", 375, 4096, 5120, 1, 1),
    ("projector2", 375, 2048, 4096, 1, 0),
]
CASE_FIELDS = ("case", "M", "N", "K", "bias", "act")


def variant_flags(name):
    v = VARIANTS[name]
    return (f"{FLAGS} -DASR_TASK3_M1_ENABLED=1 -DASR_TASK3_M1_WARPS=4 "
            f"-DASR_TASK3_GEMM_IMPL={v['impl']} -DASR_TASK3_TILE={v['tile']} "
            f"-DASR_TASK3_RM={v['rm']} -DASR_TASK3_RN={v['rn']}")
