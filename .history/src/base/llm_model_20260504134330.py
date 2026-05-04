import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Try to import LangChain's HuggingFacePipeline from known locations; if not
# available, we'll fall back to a small local wrapper so the project still
# imports cleanly on environments (like Colab) where langchain layout differs.
try:
    # older langchain layout
    from langchain.llms.huggingface_pipeline import HuggingFacePipeline  # type: ignore
except Exception:
    try:
        # some installs expose it at top-level
        from langchain import HuggingFacePipeline  # type: ignore
    except Exception:
        HuggingFacePipeline = None


class _SimpleHFWrapper:
    """A minimal wrapper around a transformers pipeline that mimics the
    interface used in this repo (callable returning text). This is a safe
    fallback for CPU/Colab testing when LangChain's pipeline wrapper isn't
    installed or the layout differs between versions.
    """

    def __init__(self, hf_pipeline, model_kwargs=None):
        self.pipeline = hf_pipeline
        self.model_kwargs = model_kwargs or {}

    def __call__(self, prompt: str) -> str:
        out = self.pipeline(prompt, **self.model_kwargs)
        # transformers text-generation returns a list of dicts
        if isinstance(out, list) and len(out) > 0 and "generated_text" in out[0]:
            return out[0]["generated_text"]
        # fallback: stringify
        return str(out)


def get_hf_llm(
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
    max_new_token=1024,
    use_4bit: bool = False,
    device: str | int = "auto",
    **kwargs
):
    """Load a HuggingFace causal LM and return an LLM-like object.

    On environments without bitsandbytes or a supported CUDA device (e.g.
    typical Colab CPU runtimes) this function will avoid 4-bit quantization
    and return a CPU-friendly pipeline wrapper.
    """

    # Determine whether to use GPU/quantization
    use_gpu = False
    try:
        use_gpu = torch.cuda.is_available()
    except Exception:
        use_gpu = False

    # If user requested 4-bit but environment doesn't support it, disable it
    use_quant = bool(use_4bit and use_gpu)

    # Load model without forcing BitsAndBytes quantization unless available
    model_kwargs = dict(low_cpu_mem_usage=True)
    if use_quant:
        # import here to avoid hard dependency at module import time
        try:
            from transformers import BitsAndBytesConfig

            nf4_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                # compute dtype left to transformers default
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=nf4_config,
                **model_kwargs,
            )
        except Exception:
            # fallback if bitsandbytes not installed or failed
            model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # choose device for pipeline: -1 -> CPU, 0 -> first GPU, "auto" -> let HF choose
    pipeline_device = device
    if device == "auto":
        pipeline_device = 0 if use_gpu else -1

    model_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_token,
        pad_token_id=tokenizer.eos_token_id,
        device=pipeline_device,
    )

    # If LangChain's HuggingFacePipeline is available, use it; otherwise use shim
    if HuggingFacePipeline is not None:
        return HuggingFacePipeline(pipeline=model_pipeline, model_kwargs=kwargs)

    return _SimpleHFWrapper(model_pipeline, model_kwargs=kwargs)


