"""Programmatic romanisation noise for Hinglish TRAINING text (augmentation only; never applied to dev or test).

Patients type Hinglish with no fixed spelling ("nahi" / "nahin" / "nhi", "bahut" / "bohot"). The D3 renderings were
written carefully, so a tuned gate could overfit to one spelling. These transformations imitate common spelling
variation without changing meaning: digits and non-Latin text are never touched, negation words only take
well-known variants, and at most one change is applied to a word.
"""
import re

VARIANTS = {
    "mujhe": ["mjhe", "mujhey"], "meri": ["mri", "merii"], "mera": ["mra", "meraa"], "hai": ["hy", "he"],
    "hain": ["hen", "hai"], "hoon": ["hun", "hu"], "raha": ["rha", "rahaa"], "rahi": ["rhi", "rahii"],
    "rahe": ["rhe", "rahey"], "nahi": ["nahin", "nhi"], "nahin": ["nahi", "nhi"], "nhi": ["nahi", "nahin"],
    "mein": ["me", "mai"], "aur": ["or", "aor"], "bahut": ["bohot", "bahot", "bhut"],
    "zyada": ["jyada", "zyda", "jada"], "kal": ["kl"], "aaj": ["aj"], "bhi": ["bi"], "kya": ["kia", "kyaa"],
    "bacha": ["bachha", "baccha"], "bachche": ["bacche", "bachhe"], "bukhar": ["bukhaar", "bukhr"],
    "ulti": ["ultee"], "saans": ["sans"],      # never "saas": that is a different Hindi word (mother-in-law) "seene": ["sine", "sene"], "chhati": ["chati", "chhaati"],
    "khoon": ["khun", "khon"], "behosh": ["behos", "bahosh"], "paseena": ["pasina", "pasena"],
    "chakkar": ["chakar", "chakkr"], "tez": ["tej"], "thoda": ["thora", "thodaa"],
}
NEGATION = {"nahi", "nahin", "nhi", "na", "mat", "bina", "koi", "no", "not", "never"}
LATIN = re.compile(r"[A-Za-z]+")
VOWELS = "aeiou"


def _generic(w, rng):
    """One spelling change to a lowercase word of length >= 4; returns it unchanged if no rule applies."""
    ops = []
    for a, b in (("aa", "a"), ("ee", "i"), ("oo", "u"), ("ii", "i")):
        if a in w:
            ops.append(w.replace(a, b, 1))
    if "z" in w:
        ops.append(w.replace("z", "j", 1))
    for a, b in (("chh", "ch"), ("bh", "b")):
        if a in w:
            ops.append(w.replace(a, b, 1))
    if len(w) >= 5:
        inner = [i for i in range(1, len(w) - 1) if w[i] in VOWELS and w[i - 1] not in VOWELS]
        if inner:
            i = rng.choice(inner)
            ops.append(w[:i] + w[i + 1:])
    return rng.choice(ops) if ops else w


def noisify(text, rng, p_word=0.35, p_join=0.08):
    """rng: random.Random. Deterministic for a given rng state. p_word: chance a word is altered;
    p_join: chance two adjacent short Latin words lose the space between them."""
    def repl(m):
        w, lw = m.group(0), m.group(0).lower()
        if rng.random() >= p_word:
            return w
        if lw in VARIANTS:
            new = rng.choice(VARIANTS[lw])
        elif lw in NEGATION or len(lw) < 4:
            return w
        else:
            new = _generic(lw, rng)
        return new.capitalize() if w[:1].isupper() else new

    out = LATIN.sub(repl, text)
    toks, res, i = out.split(" "), [], 0
    while i < len(toks):
        t = toks[i]
        if (i + 1 < len(toks) and t.isalpha() and toks[i + 1].isalpha() and len(t) <= 4 and len(toks[i + 1]) <= 4
                and t.isascii() and toks[i + 1].isascii() and rng.random() < p_join):
            t, i = t + toks[i + 1], i + 1
        res.append(t)
        i += 1
    return " ".join(res)
