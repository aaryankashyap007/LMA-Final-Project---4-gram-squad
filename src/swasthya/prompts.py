"""Prompt + letter-scoring helpers for the LLM gate (prompted V0 and QLoRA G1).

The model is asked to answer with ONE letter. We never generate: we read the next-token logits of
the four letters at the last position and softmax them -> calibrated-ish probabilities, one forward pass.
The rubric below is deliberately generic (category-level), not copied from any dev/test case.
"""
import random
from collections import defaultdict

from .schema import LABELS

LETTERS = ["A", "B", "C", "D"]          # same order as schema.LABELS

SYSTEM = (
    "You are a triage assistant at the front desk of a public clinic in India. A patient or a relative "
    "sends a short message in English, Hindi (Devanagari) or Hinglish (Hindi in Latin letters). "
    "Classify how urgent it is:\n"
    "A) EMERGENCY - life or limb in immediate danger (for example breathing trouble, collapse or "
    "unresponsiveness, seizure, severe bleeding, poisoning, stroke or heart-attack signs, serious danger "
    "in pregnancy or to a newborn, suicidal intent). Staff must be alerted right now.\n"
    "B) URGENT - needs a doctor the same day but is not immediately life-threatening.\n"
    "C) ROUTINE - can wait for a normal appointment (mild or long-standing problems, check-ups, reports, refills).\n"
    "D) INSUFFICIENT - too vague to judge; more questions are needed.\n"
    "If torn between two levels, choose the more urgent one. Reply with only the letter."
)


def user_prompt(text):
    return f"Patient message:\n{text}\n\nAnswer with a single letter (A, B, C or D)."


def build_messages(text, shots=()):
    msgs = [{"role": "system", "content": SYSTEM}]
    for ex in shots:
        msgs.append({"role": "user", "content": user_prompt(ex["text"])})
        msgs.append({"role": "assistant", "content": LETTERS[LABELS.index(ex["label"])]})
    msgs.append({"role": "user", "content": user_prompt(text)})
    return msgs


def pick_shots(train_rows, k, seed=13):
    """k//4 examples per label, deterministic, scripts mixed."""
    if k <= 0:
        return []
    by = defaultdict(list)
    for r in sorted(train_rows, key=lambda r: r["id"]):
        by[r["label"]].append(r)
    rng = random.Random(seed)
    shots = []
    for lab in LABELS:
        shots += rng.sample(by[lab], k // len(LABELS))
    rng.shuffle(shots)
    return shots


def render(tok, text, shots=()):
    return tok.apply_chat_template(build_messages(text, shots), tokenize=False, add_generation_prompt=True)


def letter_token_ids(tok):
    ids = []
    for L in LETTERS:
        t = tok.encode(L, add_special_tokens=False)
        if len(t) != 1:
            raise ValueError(f"letter {L!r} is not a single token for this tokenizer: {t}")
        ids.append(t[0])
    return ids


def letter_logits(model, tok, texts, letter_ids, device, shots=(), max_len=768):
    """(B,4) float32 logits for the four letters at the answer position. Needs left padding."""
    import torch
    prompts = [render(tok, t, shots) for t in texts]
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_len,
              add_special_tokens=False).to(device)
    use_amp = torch.device(device).type == "cuda"
    with torch.autocast(device_type=torch.device(device).type, dtype=torch.float16, enabled=use_amp):
        try:   # only the last position is needed: saves GBs of vocab-sized logits on long few-shot prompts
            logits = model(**enc, logits_to_keep=1).logits[:, -1, :]
        except TypeError:
            logits = model(**enc).logits[:, -1, :]
    return logits[:, letter_ids].float()
