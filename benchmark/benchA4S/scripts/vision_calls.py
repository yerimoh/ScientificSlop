"""Minimal replacement for the census's vision_calls, which lived in fig_graph/roi_extract and
went away when that item was retired. pairs165_retranscribe only needs model() -> (model, processor).

Same instrument as the paired run: Qwen2.5-VL-32B-Instruct, bf16, greedy, from the local cache.
"""
import functools, os

MODEL_ID = os.environ.get("VLM_MODEL", "Qwen/Qwen2.5-VL-32B-Instruct")


@functools.lru_cache(maxsize=1)
def model():
    import torch
    from transformers import AutoProcessor
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as VLM
    except ImportError:
        from transformers import Qwen2VLForConditionalGeneration as VLM
    m = VLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")
    m.eval()
    return m, AutoProcessor.from_pretrained(MODEL_ID)
