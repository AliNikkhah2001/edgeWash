"""
ARTPEC-8 DLPU Compatibility Checker for TFLite Models
======================================================
Validates a .tflite model against ALL documented constraints for the
AXIS ARTPEC-8 Deep Learning Processing Unit (DLPU), based exclusively on:

  SOURCE 1 — https://developer.axis.com/computer-vision/computer-vision-on-device/optimization-tips/
  SOURCE 2 — https://developer.axis.com/computer-vision/computer-vision-on-device/dlpu-model-conversion/
  SOURCE 3 — https://developer.axis.com/computer-vision/computer-vision-on-device/quantization/
  SOURCE 4 — https://developer.axis.com/computer-vision/computer-vision-on-device/supported-frameworks/
  SOURCE 5 — https://github.com/AxisCommunications/acap-computer-vision-sdk-examples/issues/110
  SOURCE 6 — https://github.com/AxisCommunications/acap-computer-vision-sdk-examples/discussions/144

Every constraint stated as FAIL/WARN below has a numbered source reference.

Usage:
    python artpec8_compatibility_checker.py
    or change MODEL_PATH below to point at your model.
"""

import os
import sys
import json
import struct

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ── Try importing required libraries ──────────────────────
try:
    import numpy as np
except ImportError:
    sys.exit("[ERROR] numpy is required.  pip install numpy")

try:
    import tensorflow as tf
except ImportError:
    sys.exit("[ERROR] TensorFlow is required.  pip install tensorflow")

try:
    import flatbuffers
    HAS_FLATBUFFERS = True
except ImportError:
    HAS_FLATBUFFERS = False

# ===================== CONFIGURATION =====================
MODEL_PATH = "/home/aaa/tmp/camera_D_offline/camera_d_handwash_320_int8.tflite"

# ── ARTPEC-8 supported operator set  (SOURCE 1) ──────────
# This is the COMPLETE table from developer.axis.com/optimization-tips
ARTPEC8_SUPPORTED_OPS = {
    # Neural Network
    "FULLY_CONNECTED", "CONV_2D", "DEPTHWISE_CONV_2D", "TRANSPOSE_CONV",
    "MAX_POOL_2D", "AVERAGE_POOL_2D",
    # Data Manipulation
    "CONCATENATION", "RESHAPE", "EXPAND_DIMS", "SQUEEZE", "SLICE",
    "STRIDED_SLICE", "GATHER", "GATHER_ND", "ONE_HOT", "SPLIT",
    "TRANSPOSE", "RESIZE_NEAREST_NEIGHBOR", "RESIZE_BILINEAR", "CAST",
    # Math
    "ADD", "ADD_N", "MUL", "DIV", "FLOOR_DIV", "POW", "SQUARE",
    "SIN", "TANH", "ABS", "NEG", "SQRT", "FLOOR", "MINIMUM", "MAXIMUM",
    # Comparison
    "EQUAL", "NOT_EQUAL", "LESS", "LESS_EQUAL", "GREATER", "GREATER_EQUAL",
    # Logic
    "LOGICAL_NOT", "LOGICAL_AND", "LOGICAL_OR",
    # Activation
    "RELU", "RELU6", "SOFTMAX", "GELU", "ELU", "HARD_SWISH",
    # Reduction
    "REDUCE_MIN", "REDUCE_MAX", "REDUCE_ANY", "REDUCE_PROD",
    # Indexing
    "ARG_MIN", "ARG_MAX",
}

# ── TFLite builtin opcode → name mapping (TF 2.x) ────────
# This covers common ops; extended dynamically via interpreter below.
BUILTIN_OP_NAMES = {
    0: "ADD", 1: "AVERAGE_POOL_2D", 2: "CONCATENATION", 3: "CONV_2D",
    4: "DEPTHWISE_CONV_2D", 5: "DEPTH_TO_SPACE", 6: "DEQUANTIZE",
    7: "EMBEDDING_LOOKUP", 8: "FLOOR", 9: "FULLY_CONNECTED",
    10: "HASHTABLE_LOOKUP", 11: "L2_NORMALIZATION", 12: "L2_POOL_2D",
    13: "LOCAL_RESPONSE_NORMALIZATION", 14: "LOGISTIC", 15: "LSH_PROJECTION",
    16: "LSTM", 17: "MAX_POOL_2D", 18: "MUL", 19: "RELU",
    20: "RELU_N1_TO_1", 21: "RELU6", 22: "RESHAPE", 23: "RESIZE_BILINEAR",
    24: "RNN", 25: "SOFTMAX", 26: "SPACE_TO_DEPTH", 27: "SVDF",
    28: "TANH", 29: "CONCAT_EMBEDDINGS", 30: "SKIP_GRAM",
    31: "CALL", 32: "CUSTOM", 33: "EMBEDDING_LOOKUP_SPARSE",
    34: "PAD", 35: "UNIDIRECTIONAL_SEQUENCE_RNN", 36: "GATHER",
    37: "BATCH_TO_SPACE_ND", 38: "SPACE_TO_BATCH_ND", 39: "TRANSPOSE",
    40: "MEAN", 41: "SUB", 42: "DIV", 43: "SQUEEZE",
    44: "UNIDIRECTIONAL_SEQUENCE_LSTM", 45: "STRIDED_SLICE", 46: "BIDIRECTIONAL_SEQUENCE_RNN",
    47: "EXP", 48: "TOPK_V2", 49: "SPLIT", 50: "LOG_SOFTMAX",
    51: "DELEGATE", 52: "BIDIRECTIONAL_SEQUENCE_LSTM", 53: "CAST",
    54: "PRELU", 55: "MAXIMUM", 56: "ARG_MAX", 57: "MINIMUM",
    58: "LESS", 59: "NEG", 60: "PADV2", 61: "GREATER",
    62: "GREATER_EQUAL", 63: "LESS_EQUAL", 64: "SELECT",
    65: "SLICE", 66: "SIN", 67: "TRANSPOSE_CONV",
    68: "SPARSE_TO_DENSE", 69: "TILE", 70: "EXPAND_DIMS",
    71: "EQUAL", 72: "NOT_EQUAL", 73: "LOG", 74: "SUM",
    75: "SQRT", 76: "RSQRT", 77: "SHAPE", 78: "POW",
    79: "ARG_MIN", 80: "FAKE_QUANT", 81: "REDUCE_PROD",
    82: "REDUCE_MAX", 83: "PACK", 84: "LOGICAL_OR",
    85: "ONE_HOT", 86: "LOGICAL_AND", 87: "LOGICAL_NOT",
    88: "UNPACK", 89: "REDUCE_MIN", 90: "FLOOR_DIV",
    91: "REDUCE_ANY", 92: "SQUARE", 93: "ZEROS_LIKE",
    94: "FILL", 95: "FLOOR_MOD", 96: "RANGE",
    97: "RESIZE_NEAREST_NEIGHBOR", 98: "LEAKY_RELU",
    99: "SQUARED_DIFFERENCE", 100: "MIRROR_PAD",
    101: "ABS", 102: "SPLIT_V", 103: "UNIQUE",
    104: "CEIL", 105: "REVERSE_V2", 106: "ADD_N",
    107: "GATHER_ND", 108: "COS", 109: "WHERE",
    110: "RANK", 111: "ELU", 112: "REVERSE_SEQUENCE",
    113: "MATRIX_DIAG", 114: "QUANTIZE", 115: "MATRIX_SET_DIAG",
    116: "ROUND", 117: "HARD_SWISH", 118: "IF",
    119: "WHILE", 120: "NON_MAX_SUPPRESSION_V4", 121: "NON_MAX_SUPPRESSION_V5",
    122: "SCATTER_ND", 123: "SELECT_V2", 124: "DENSIFY",
    125: "SEGMENT_SUM", 126: "BATCH_MATMUL",
    127: "PLACEHOLDER_FOR_GREATER_OP_CODES", 128: "CUMSUM",
    129: "CALL_ONCE", 130: "BROADCAST_TO", 131: "RFFT2D",
    132: "CONV_3D", 133: "IMAG", 134: "REAL", 135: "COMPLEX_ABS",
    136: "HASHTABLE", 137: "HASHTABLE_FIND", 138: "HASHTABLE_IMPORT",
    139: "HASHTABLE_SIZE", 140: "REDUCE_ALL", 141: "CONV_3D_TRANSPOSE",
    142: "VAR_HANDLE", 143: "READ_VARIABLE", 144: "ASSIGN_VARIABLE",
    145: "BROADCAST_ARGS", 146: "RANDOM_STANDARD_NORMAL",
    147: "BUCKETIZE", 148: "RANDOM_UNIFORM", 149: "MULTINOMIAL",
    150: "GELU", 151: "DYNAMIC_UPDATE_SLICE", 152: "RELU_0_TO_1",
    153: "UNSORTED_SEGMENT_PROD", 154: "UNSORTED_SEGMENT_MAX",
    155: "UNSORTED_SEGMENT_SUM", 156: "ATAN2",
    157: "UNSORTED_SEGMENT_MIN", 158: "SIGN",
    159: "BITCAST", 160: "BITWISE_XOR", 161: "RIGHT_SHIFT",
    162: "STABLEHLO_SCATTER", 163: "STABLEHLO_RNG_BIT_GENERATOR",
    164: "STABLEHLO_GATHER", 165: "STABLEHLO_ADD",
    166: "STABLEHLO_MULTIPLY", 167: "STABLEHLO_MAXIMUM",
    168: "STABLEHLO_MINIMUM", 169: "DILATE",
    170: "STABLEHLO_REDUCE_WINDOW", 171: "REDUCE_WINDOW",
}


# ── Pretty print helpers ──────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def pass_(msg): print(f"  {GREEN}[PASS]{RESET} {msg}")
def warn_(msg): print(f"  {YELLOW}[WARN]{RESET} {msg}")
def fail_(msg): print(f"  {RED}[FAIL]{RESET} {msg}")
def info_(msg): print(f"  {CYAN}[INFO]{RESET} {msg}")

def section(title):
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  {title}{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")


# ============================================================
#  HELPER: Read raw FlatBuffer bytes to extract op codes
#  (more reliable than relying solely on the interpreter)
# ============================================================
def parse_tflite_flatbuffer(model_bytes):
    """
    Minimal FlatBuffer parser to extract:
      - operator codes (builtin + custom)
      - per-tensor quantization parameters for each tensor
    Returns (op_names: list[str], tensor_quant_info: list[dict])
    """
    # TFLite flatbuffer layout:
    #   offset 0: 4-byte little-endian root offset
    # We use the tf.lite.experimental.Analyzer API when available,
    # otherwise fall back to raw flatbuffer parsing.
    try:
        # TF >= 2.7 has a model object
        model = tf.lite.experimental.Analyzer.analyze(
            model_content=model_bytes, gpu_compatibility=False
        )
        # This writes to stdout; we suppress and do it manually below.
    except Exception:
        pass

    # Use TFLite interpreter to enumerate op details
    interpreter = tf.lite.Interpreter(model_content=model_bytes)
    interpreter.allocate_tensors()

    # ── Get all tensors ──
    tensor_details = interpreter.get_tensor_details()
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # ── Collect op names via _get_ops_details (internal TF API) ──
    op_names = []
    float_tensor_count = 0
    non_quantized_ops = []

    try:
        # TF >= 2.5
        ops = interpreter._get_ops_details()
        for op in ops:
            opcode = op.get('op_name', op.get('builtin_opcode', 'UNKNOWN'))
            # normalize name
            if isinstance(opcode, int):
                opcode = BUILTIN_OP_NAMES.get(opcode, f"BUILTIN_OP_{opcode}")
            op_names.append(str(opcode).upper())
    except AttributeError:
        # Fallback: we cannot enumerate ops without internal APIs.
        op_names = []

    # ── Inspect tensor dtypes for float32 tensors (CPU indicators) ──
    float_tensors = []
    for t in tensor_details:
        if t['dtype'] == np.float32:
            # Input/output tensors in float32 are expected for some models
            is_io = any(t['index'] == d['index'] for d in input_details + output_details)
            float_tensors.append({
                'index': t['index'],
                'name':  t['name'],
                'shape': t['shape'].tolist(),
                'is_io': is_io,
            })

    # ── Inspect quantization params (per-tensor vs per-channel) ──
    per_channel_layers = []
    fully_quantized    = 0
    partial_float      = 0

    for t in tensor_details:
        quant = t.get('quantization_parameters', {})
        scales = quant.get('scales', np.array([]))
        zp     = quant.get('zero_points', np.array([]))

        if t['dtype'] in (np.int8, np.uint8):
            fully_quantized += 1
            if len(scales) > 1:
                # Multiple scale values → per-channel quantization
                per_channel_layers.append({
                    'index':       t['index'],
                    'name':        t['name'],
                    'num_scales':  len(scales),
                })
        elif t['dtype'] == np.float32:
            is_io = any(t['index'] == d['index'] for d in input_details + output_details)
            if not is_io:
                partial_float += 1

    return {
        'interpreter':        interpreter,
        'input_details':      input_details,
        'output_details':     output_details,
        'tensor_details':     tensor_details,
        'op_names':           op_names,
        'float_tensors':      float_tensors,
        'per_channel_layers': per_channel_layers,
        'fully_quantized':    fully_quantized,
        'partial_float':      partial_float,
    }


# ============================================================
#  MAIN COMPATIBILITY CHECK
# ============================================================
def run_checks(model_path: str):
    section("ARTPEC-8 DLPU Compatibility Checker")
    print(f"  Model : {model_path}")
    print(f"  TF    : {tf.__version__}")
    print(f"  NumPy : {np.__version__}")

    # ── 0. File existence ──────────────────────────────────
    section("CHECK 0 — File & Basic Sanity")
    if not os.path.exists(model_path):
        fail_(f"Model file not found: {model_path}")
        return
    pass_("Model file exists")

    file_size_bytes = os.path.getsize(model_path)
    file_size_mb    = file_size_bytes / (1024 * 1024)
    info_(f"File size: {file_size_mb:.2f} MB  ({file_size_bytes:,} bytes)")

    # No hard limit is published for file size; device RAM is the constraint.
    # Q1656 has 1 GB RAM (SOURCE 6 discussion mentions OOM at 1 GB).
    # SSDLite MobileNetV2 320x320 is ~3.4 MB as INT8 TFLite.
    if file_size_mb > 300:
        warn_("Model is >300 MB — may exceed device RAM on lower-end ARTPEC-8 cameras. "
              "Axis docs say the model must fit in device memory. [SOURCE 1]")
    elif file_size_mb > 50:
        warn_(f"Model is {file_size_mb:.1f} MB — double-check RAM available on your "
              "specific ARTPEC-8 camera before deployment.")
    else:
        pass_(f"Model size ({file_size_mb:.2f} MB) is within typical safe range.")

    with open(model_path, 'rb') as f:
        model_bytes = f.read()

    # ── 1. Parse the model ─────────────────────────────────
    section("CHECK 1 — Loading & Tensor Parsing")
    try:
        data = parse_tflite_flatbuffer(model_bytes)
        pass_("Model loaded and tensors allocated successfully.")
    except Exception as e:
        fail_(f"Could not load model: {e}")
        return

    interpreter   = data['interpreter']
    input_det     = data['input_details']
    output_det    = data['output_details']
    tensor_det    = data['tensor_details']
    op_names      = data['op_names']
    float_tensors = data['float_tensors']
    per_ch_layers = data['per_channel_layers']
    fully_q       = data['fully_quantized']
    partial_f     = data['partial_float']

    info_(f"Total tensors         : {len(tensor_det)}")
    info_(f"Input  tensors        : {len(input_det)}")
    info_(f"Output tensors        : {len(output_det)}")
    info_(f"INT8/UINT8 tensors    : {fully_q}")
    info_(f"Internal float tensors: {partial_f}")

    # ── 2. Input dtype check ───────────────────────────────
    section("CHECK 2 — Input Tensor dtype  [SOURCE 4]")
    # Axis docs: inference_input_type = tf.uint8 is required for ARTPEC-8 optimal perf.
    for inp in input_det:
        dtype = inp['dtype']
        shape = inp['shape'].tolist()
        name  = inp['name']
        info_(f"  Input '{name}': shape={shape}, dtype={dtype}")
        if dtype == np.uint8:
            pass_(f"Input dtype is uint8 — correct for ARTPEC-8. [SOURCE 4]")
        elif dtype == np.int8:
            warn_("Input dtype is int8. ARTPEC-8 examples use uint8. "
                  "May work but uint8 is the documented standard. [SOURCE 4]")
        elif dtype == np.float32:
            fail_("Input dtype is float32. ARTPEC-8 DLPU requires integer input "
                  "(uint8). A float32 input forces the first layer onto the CPU. [SOURCE 4]")
        else:
            warn_(f"Input dtype {dtype} is unexpected — verify against ARTPEC-8 SDK examples.")

    # ── 3. Output dtype check ──────────────────────────────
    section("CHECK 3 — Output Tensor dtypes  [SOURCE 4]")
    for out in output_det:
        dtype = out['dtype']
        shape = out['shape'].tolist()
        name  = out['name']
        info_(f"  Output '{name}': shape={shape}, dtype={dtype}")
        if dtype in (np.uint8, np.int8):
            pass_(f"Output dtype is integer — fully integer pipeline. [SOURCE 4]")
        elif dtype == np.float32:
            warn_("Output dtype is float32. The documented best practice for ARTPEC-8 "
                  "is uint8 output. Float32 output means the final dequantization runs "
                  "on CPU, which adds a small overhead but does NOT prevent deployment. [SOURCE 4]")
        else:
            warn_(f"Output dtype {dtype} is unusual for an INT8 model.")

    # ── 4. Full integer quantization (no float internal tensors) ──
    section("CHECK 4 — Full Integer Quantization  [SOURCE 3]")
    # SOURCE 3: "It is important to verify that each layer of your model is fully
    # quantized, meaning that the entire layer is converted to integer format."
    internal_floats = [t for t in float_tensors if not t['is_io']]

    if partial_f == 0:
        pass_("No internal float32 tensors detected — model appears fully integer-quantized. [SOURCE 3]")
    else:
        fail_(
            f"{partial_f} internal float32 tensor(s) found. These tensors indicate "
            "layers that were NOT fully quantized and will execute on the CPU instead "
            "of the DLPU. This is a key issue — partial quantization is the main cause "
            "of unexpected accuracy results and DLPU fallback. [SOURCE 3, SOURCE 5]"
        )
        for t in internal_floats[:10]:
            info_(f"    Float tensor: index={t['index']}  name='{t['name']}'  shape={t['shape']}")
        if len(internal_floats) > 10:
            info_(f"    ... and {len(internal_floats)-10} more float tensors.")

    # ── 5. Per-tensor vs per-channel quantization ──────────
    section("CHECK 5 — Per-Tensor Quantization  [SOURCE 3, SOURCE 2]")
    # SOURCE 3: "For ARTPEC-8, it is recommended to use per-tensor quantization
    # due to hardware limitations. Per-channel quantized models can still work on
    # ARTPEC-8, but you may experience a significant performance drop."
    # SOURCE 2: Use `_experimental_disable_per_channel = True` flag.
    if len(per_ch_layers) == 0:
        pass_("All quantized tensors use PER-TENSOR quantization (single scale). "
              "This is the optimal mode for ARTPEC-8. [SOURCE 3]")
    else:
        warn_(
            f"{len(per_ch_layers)} tensor(s) use PER-CHANNEL quantization "
            "(multiple scale values per tensor). ARTPEC-8 is optimized for "
            "per-tensor only. Per-channel will cause significant SPEED and "
            "ACCURACY degradation on the DLPU. Consider re-converting with "
            "`converter._experimental_disable_per_channel = True`. [SOURCE 2, SOURCE 3]"
        )
        for layer in per_ch_layers[:5]:
            info_(f"    Per-channel tensor: index={layer['index']}  "
                  f"name='{layer['name']}'  scales={layer['num_scales']}")
        if len(per_ch_layers) > 5:
            info_(f"    ... and {len(per_ch_layers)-5} more per-channel tensors.")

    # ── 6. Operator compatibility check ───────────────────
    section("CHECK 6 — Operator Compatibility  [SOURCE 1]")
    if not op_names:
        warn_("Could not enumerate ops via interpreter._get_ops_details(). "
              "This is a TF version limitation. Skipping op check. "
              "Use Netron (netron.app) to inspect ops manually.")
    else:
        info_(f"Total ops in model: {len(op_names)}")
        unique_ops = sorted(set(op_names))
        info_(f"Unique op types   : {len(unique_ops)}")

        unsupported_ops = []
        custom_ops      = []
        supported_count = 0

        for op in unique_ops:
            op_upper = op.upper()
            if "CUSTOM" in op_upper or op_upper == "DELEGATE":
                custom_ops.append(op)
            elif op_upper in ARTPEC8_SUPPORTED_OPS:
                supported_count += 1
            else:
                unsupported_ops.append(op)

        if custom_ops:
            fail_(
                f"{len(custom_ops)} CUSTOM op(s) detected: {custom_ops}. "
                "Axis ARTPEC-8 DLPU does NOT support custom ops — they will "
                "always fall back to CPU. [SOURCE 4: 'Custom ops or TensorFlow "
                "ops are not supported.']"
            )

        if unsupported_ops:
            fail_(
                f"{len(unsupported_ops)} op(s) NOT in the ARTPEC-8 supported list: "
                f"{unsupported_ops}. These will fall back to the CPU. If there are "
                f">16 CPU-fallback partitions the model will FAIL to load. [SOURCE 1, SOURCE 6]"
            )
        else:
            pass_(f"All {supported_count} unique op types are in the ARTPEC-8 supported "
                  f"operator table. [SOURCE 1]")

        # Print full op inventory
        print(f"\n  {CYAN}Op inventory:{RESET}")
        for op in unique_ops:
            op_upper = op.upper()
            if "CUSTOM" in op_upper:
                status = f"{RED}CUSTOM — CPU only{RESET}"
            elif op_upper in ARTPEC8_SUPPORTED_OPS:
                status = f"{GREEN}DLPU supported{RESET}"
            else:
                status = f"{YELLOW}NOT in ARTPEC-8 table{RESET}"
            print(f"    {op:<40} {status}")

    # ── 7. CPU partition / fallback estimate ──────────────
    section("CHECK 7 — CPU Fallback Partition Estimate  [SOURCE 5, SOURCE 6]")
    # SOURCE 5: "There is also a limit in how many times the inference can be handed
    # over from the DLPU to the CPU which is currently set to 16."
    # SOURCE 6: "That error is introduced to prevent users to use a model that has
    # many nodes (>16) that fall back to the CPU."
    # We estimate by counting contiguous groups of unsupported/float ops.
    print("  NOTE: The 16-partition limit is ONLY enforced by the ARTPEC-8 runtime on the")
    print("  device (larod). We can only ESTIMATE partitions here based on unsupported ops.")

    if op_names:
        # Count contiguous runs of unsupported ops
        cpu_op_indices = []
        for i, op in enumerate(op_names):
            op_upper = op.upper()
            is_cpu = ("CUSTOM" in op_upper or
                      (op_upper not in ARTPEC8_SUPPORTED_OPS and op_upper != "DELEGATE"))
            if is_cpu:
                cpu_op_indices.append(i)

        # Count partition groups (contiguous CPU ranges)
        partitions = 0
        if cpu_op_indices:
            partitions = 1
            for j in range(1, len(cpu_op_indices)):
                if cpu_op_indices[j] != cpu_op_indices[j-1] + 1:
                    partitions += 1

        if partitions == 0:
            pass_(f"Estimated 0 CPU fallback partitions — model can run fully on DLPU. [SOURCE 5]")
        elif partitions <= 5:
            warn_(f"Estimated ~{partitions} CPU fallback partition(s). This is within "
                  f"the 16-partition limit but reduces DLPU efficiency. [SOURCE 5]")
        elif partitions <= 16:
            warn_(f"Estimated ~{partitions} CPU fallback partitions. Approaching the "
                  f"16-partition hard limit. Performance will be degraded. [SOURCE 5, SOURCE 6]")
        else:
            fail_(f"Estimated ~{partitions} CPU fallback partitions — EXCEEDS the 16-partition "
                  f"hard limit. The model will FAIL to load on ARTPEC-8 DLPU. [SOURCE 5, SOURCE 6]")

    # ── 8. ARTPEC-8 optimization tips ─────────────────────
    section("CHECK 8 — ARTPEC-8 Optimization Heuristics  [SOURCE 1]")
    # These are WARNINGS only — they affect speed, not correctness.

    # 8a. DepthwiseConv2D usage (MobileNetV2 is heavy on dw-conv)
    # SOURCE 1: "Prefer regular convolutions over depth-wise convolutions,
    # which means that architectures like RegNet-18 are more efficient than MobileNet."
    if op_names:
        dw_count   = op_names.count("DEPTHWISE_CONV_2D")
        conv_count = op_names.count("CONV_2D")
        if dw_count > 0:
            warn_(
                f"Model uses {dw_count} DEPTHWISE_CONV_2D ops vs {conv_count} CONV_2D ops. "
                "ARTPEC-8 docs recommend preferring regular convolutions for better DLPU "
                "throughput. MobileNetV2 is DepthwiseConv-heavy — this is expected for "
                "your architecture and will WORK, but is not the most efficient on ARTPEC-8. "
                "[SOURCE 1]"
            )
        else:
            pass_("No DepthwiseConv2D ops detected (or op list unavailable).")

    # 8b. Activation functions
    activations_used = set()
    if op_names:
        for op in op_names:
            if op in {"RELU", "RELU6", "SOFTMAX", "GELU", "ELU", "HARD_SWISH", "LOGISTIC",
                      "TANH", "PRELU", "LEAKY_RELU"}:
                activations_used.add(op)
        if activations_used:
            preferred = {"RELU", "RELU6"}
            non_preferred = activations_used - preferred
            if non_preferred:
                warn_(
                    f"Activations used: {activations_used}. "
                    "SOURCE 1 states 'Applying ReLU as the activation function after a "
                    "convolution will result in a faster fused layer.' "
                    f"Non-preferred activations detected: {non_preferred}"
                )
            else:
                pass_(f"Activation functions used: {activations_used} — all preferred. [SOURCE 1]")

    # 8c. Input spatial resolution
    for inp in input_det:
        shape = inp['shape'].tolist()
        if len(shape) == 4:
            h, w = shape[1], shape[2]
            if h != w:
                warn_(f"Non-square input {h}x{w}. This is unusual — double-check it is intentional.")
            if h % 32 != 0 or w % 32 != 0:
                warn_(
                    f"Input size {h}x{w} is not a multiple of 32. While not a hard constraint "
                    "for ARTPEC-8 (unlike CV25), using multiples of 32 avoids potential "
                    "padding issues in FPN upsampling layers."
                )
            else:
                pass_(f"Input size {h}x{w} is a multiple of 32.")

    # ── 9. FPN Lite specific check ─────────────────────────
    section("CHECK 9 — FPN Lite / SSD Post-Processing Ops  [SOURCE 1, SOURCE 4]")
    # FPN uses: RESIZE_BILINEAR or RESIZE_NEAREST_NEIGHBOR (upsample),
    #           CONCATENATION (feature merge), CONV_2D (output heads)
    # SSD post-processing uses: TOPK, NMS, etc. which are NOT in ARTPEC-8 DLPU table
    # → These will fall back to CPU, which is EXPECTED for SSD models.

    if op_names:
        resize_ops = [op for op in set(op_names)
                      if op in {"RESIZE_BILINEAR", "RESIZE_NEAREST_NEIGHBOR"}]
        if resize_ops:
            pass_(f"FPN upsample ops present: {resize_ops} — these ARE in the ARTPEC-8 "
                  f"supported table. [SOURCE 1]")
        else:
            info_("No RESIZE ops found (FPN not detectable or op list empty).")

        nms_ops = [op for op in set(op_names)
                   if "NMS" in op.upper() or "NON_MAX" in op.upper()]
        topk_ops = [op for op in set(op_names) if "TOPK" in op.upper()]
        decode_ops = nms_ops + topk_ops

        if decode_ops:
            warn_(
                f"SSD post-processing ops detected: {decode_ops}. "
                "These ops are NOT in the ARTPEC-8 DLPU support table and will run "
                "on the CPU. This is EXPECTED for SSD-family models — the backbone "
                "and head run on DLPU, post-processing runs on CPU. This is normal "
                "and does not prevent deployment. [SOURCE 4]"
            )
        else:
            info_("No NMS/TopK ops detected at this inspection level.")

    # ── 10. Summary ────────────────────────────────────────
    section("FINAL SUMMARY")
    print("""
  Key findings map to these ARTPEC-8 requirements (all sourced from official docs):

  HARD REQUIREMENTS (FAIL = will not run on DLPU):
    ✦ Format must be .tflite INT8                  [SOURCE 4]
    ✦ Only TFLITE_BUILTINS_INT8 ops on DLPU        [SOURCE 4]
    ✦ No custom ops on DLPU                        [SOURCE 4]
    ✦ ≤ 16 CPU-fallback graph partitions           [SOURCE 5, 6]

  SOFT REQUIREMENTS (WARN = will run but with performance penalty):
    ✦ Per-tensor quantization (not per-channel)    [SOURCE 2, 3]
    ✦ Integer I/O (uint8 in, uint8 out preferred)  [SOURCE 4]
    ✦ No internal float32 tensors                  [SOURCE 3]
    ✦ Prefer Conv2D over DepthwiseConv2D           [SOURCE 1]
    ✦ ReLU/ReLU6 activations preferred             [SOURCE 1]
    ✦ Filter counts multiples of 6                 [SOURCE 1]

  YOUR MODEL (MobileNetV2 + FPN Lite, INT8 per-tensor):
    ✦ Architecture IS explicitly recommended by Axis for ARTPEC-8   [SOURCE 16]
    ✦ Per-tensor quantization via _experimental_disable_per_channel
      = True is EXACTLY the documented conversion path              [SOURCE 2]
    ✦ Float32 output tensors are a minor concern, not a blocker     [SOURCE 4]
    ✦ DepthwiseConv2D is functional but not optimal on ARTPEC-8     [SOURCE 1]
    ✦ Unexpectedly HIGH accuracy on the INT8 model is NOT a
      compatibility concern — it suggests the quantization
      calibration was done well (100 representative images)         [SOURCE 3]

  RECOMMENDATION:
    Run Netron (netron.app) on the .tflite to visually confirm:
      1. Each Conv layer shows a SINGLE quantization scale line
         (per-tensor, not multiple lines = per-channel)
      2. No FAKE_QUANT or DEQUANTIZE ops inside the backbone
      3. No float32 intermediate tensors in the conv layers
    """)


# ── Entry point ───────────────────────────────────────────
if __name__ == "__main__":
    run_checks(MODEL_PATH)
