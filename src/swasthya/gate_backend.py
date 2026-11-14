"""Torch backend for the gate (Kaggle / any GPU box): Qwen2.5-3B, optionally with the G1 QLoRA adapter.

Uses exactly the read-out of training and of run_llm_prompted.py: the logits of the letters A-D at the answer
position (prompts.letter_logits), softmaxed. torch / transformers / peft are imported lazily so the rest of the
package (and its tests) do not need them. Not covered by the local tests: verify on a GPU by comparing its
predictions on the D3 dev split with outputs/results for the same adapter.
"""
from . import prompts


class LetterBackend:
    def __init__(self, model, tok, device, shots=(), max_len=768, bs=8):
        self.model, self.tok, self.device = model, tok, device
        self.shots, self.max_len, self.bs = list(shots), max_len, bs
        self.letter_ids = prompts.letter_token_ids(tok)

    def probs(self, texts):
        import torch
        out = []
        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(texts), self.bs):
                lg = prompts.letter_logits(self.model, self.tok, texts[i:i + self.bs], self.letter_ids,
                                           self.device, shots=self.shots, max_len=self.max_len)
                out += torch.softmax(lg, -1).cpu().tolist()
        return out


def load_backend(model_name="Qwen/Qwen2.5-3B-Instruct", adapter_dir=None, bits=4, shots=(), max_len=768):
    """adapter_dir=None -> prompted V0 (pass shots for few-shot); adapter_dir=<folder> -> tuned G1."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = "left"
    tok.truncation_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cuda = torch.cuda.is_available()
    kw = {}
    if bits == 4:
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16)
        kw["device_map"] = {"": 0}
    dt = torch.float16 if cuda else torch.float32
    try:   # newer transformers call it `dtype`
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dt, **kw)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dt, **kw)
    if adapter_dir:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_dir)
    if bits != 4:
        model = model.to("cuda" if cuda else "cpu")
    model.eval()
    return LetterBackend(model, tok, "cuda" if cuda else "cpu", shots, max_len)
