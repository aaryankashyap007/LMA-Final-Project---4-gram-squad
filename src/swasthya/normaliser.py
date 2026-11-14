"""Unicode and script handling for patient messages.

for_model(text)  conservative: what the gate and its cache see. NFKC, zero-width and control characters removed,
                 Devanagari / Arabic-Indic digits to ASCII, typographic quotes and dashes to ASCII, spaces collapsed
                 WITHIN each line. Line breaks are kept because the gate input uses them to separate speakers.
                 The danda and other punctuation are kept because G1 was trained on text that contains them.
                 D3 text is unchanged by it (tested), so G1 sees no distribution shift on D3-like input.
for_match(text)  aggressive, for lexicons and tagging: lower-case, nukta removed, chandrabindu to anusvara, danda and
                 punctuation to spaces, long letter repeats shortened, common romanised-Hindi spelling variants mapped
                 to one form ("nhi" -> "nahi", "sans" -> "saans").
detect_script(text)  deva | hinglish | en | mixed | unknown. Romanised Hindi versus English is decided by counting
                 function words, so very short messages ("ok", "11 am") come back "unknown".
"""
import re
import unicodedata

_ZERO_WIDTH = re.compile("[​-‍⁠﻿­]")
_CONTROL = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_TO_ASCII = str.maketrans({
    **{chr(0x966 + i): str(i) for i in range(10)},       # Devanagari digits
    **{chr(0x660 + i): str(i) for i in range(10)},       # Arabic-Indic digits
    **{chr(0x6F0 + i): str(i) for i in range(10)},       # Extended Arabic-Indic digits
    "‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-"})
_DEVA = re.compile("[ऀ-ॿ]")
_LATIN = re.compile("[A-Za-z]")

# Spelling variants of romanised Hindi mapped to one form (same variants as noise.py, inverted). Variants that
# are ordinary English words ("he", "me", "or") are left out on purpose.
CANON = {
    "nahin": "nahi", "nhi": "nahi", "bht": "bahut", "bohot": "bahut", "bahot": "bahut", "bhut": "bahut",
    "sans": "saans", "jyada": "zyada", "zyda": "zyada", "jada": "zyada", "tej": "tez", "sine": "seene",
    "sene": "seene", "chati": "chhati", "chhaati": "chhati", "khun": "khoon", "khon": "khoon", "behos": "behosh",
    "bahosh": "behosh", "pasina": "paseena", "pasena": "paseena", "chakar": "chakkar", "chakkr": "chakkar",
    "thora": "thoda", "bukhaar": "bukhar", "bukhr": "bukhar", "rha": "raha", "rhi": "rahi", "rhe": "rahe",
    "mjhe": "mujhe", "mujhey": "mujhe", "mri": "meri", "hy": "hai", "hen": "hain", "kia": "kya", "kyaa": "kya",
    "zahar": "zeher", "jahar": "zeher", "daura": "dora", "sanp": "saanp", "dheeli": "dheeli", "kl": "kal",
    "ultee": "ulti", "rahii": "rahi", "merii": "meri", "mera": "mera", "meraa": "mera"}

HI_WORDS = set("""hai hain hoon ho nahi nahin nhi mein mujhe mujhey meri mera mere ko ka ke ki se aur kya kyun kyu
bahut bohot bht zyada jyada raha rahi rahe rha rhi tha thi kal aaj abhi uska uski uske usko unka unki unko wo woh
vo ye yeh par pe lekin magar kuch koi bhi hum hamara apna apni aap aapko tum wala wali gaya gayi gaye liya liye
diya kiya karna karo kar dena chahiye chahta chahti dard bukhar ulti saans seene khoon behosh beti beta bacha bachi
bachcha bachche ghar toh ne h paani pani dawai dawa goli peshab sab bahar andar""".split())
EN_WORDS = set("""the is are was were my and has have had she he his her with for not been since of in on at very
cannot cant can i am it this that an from got started ago please would like you your we our they their there what
when how does did will be by as if so but about""".split())


def for_model(text):
    t = unicodedata.normalize("NFKC", text or "")
    t = _CONTROL.sub("", _ZERO_WIDTH.sub("", t)).translate(_TO_ASCII)
    lines = [" ".join(line.split()) for line in t.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    return "\n".join(line for line in lines if line)


def for_match(text, canon=True):
    t = for_model(text).lower()
    t = t.replace("़", "").replace("ँ", "ं")          # nukta off, chandrabindu -> anusvara
    t = t.replace("'", "")                                             # can't -> cant
    t = re.sub(r"[^\wऀ-ॿ]+", " ", t.replace("।", " ").replace("॥", " "))
    t = re.sub(r"([a-z])\1{2,}", r"\1\1", t)                           # saaaans -> saans
    words = t.split()
    if canon:
        words = [CANON.get(w, w) for w in words]
    return " ".join(words)


def detect_script(text):
    t = unicodedata.normalize("NFKC", text or "")
    d, l = len(_DEVA.findall(t)), len(_LATIN.findall(t))
    if d + l == 0:
        return "unknown"
    if d and l and min(d, l) / (d + l) >= 0.2:
        return "mixed"
    if d >= l:
        return "deva"
    words = re.findall(r"[a-z]+", t.lower().replace("'", ""))
    hi, en = sum(w in HI_WORDS for w in words), sum(w in EN_WORDS for w in words)
    if hi == 0 and en == 0:                 # no function words: "ok", "11 am" stay unknown, a phrase is English
        return "en" if len(words) >= 3 else "unknown"
    return "hinglish" if hi >= en else "en"
