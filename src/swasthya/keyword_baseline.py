"""Baseline B0: keyword lexicon (script-agnostic: English + Devanagari + Hinglish terms).
Priority: EMERGENCY > URGENT > ROUTINE > INSUFFICIENT (nothing matched).
Deliberately naive: no negation handling, so "no fever" still matches "fever".
"""
import re
import unicodedata

EMERGENCY = {
    "en": ["cant breathe", "cannot breathe", "can not breathe", "not breathing", "breathless",
           "difficulty breathing", "blue lips", "lips are turning blue", "unconscious",
           "not responding", "unresponsive", "fainted", "seizure", "seizures", "fits",
           "convulsion", "heavy bleeding", "bleeding heavily", "bleeding a lot", "poison",
           "poisoning", "overdose", "swallowed", "chest pain", "sweating", "face drooping",
           "slurred", "paralysis", "paralyzed", "end my life", "kill myself", "suicide",
           "not feeding", "limp", "stopped breathing", "choking", "snake bite", "snakebite"],
    "deva": ["सांस नहीं", "सांस लेने में तकलीफ", "सांस फूल", "होंठ नीले", "बेहोश", "जवाब नहीं दे",
             "दौरा", "झटके", "ज़्यादा ब्लीडिंग", "बहुत ब्लीडिंग", "बहुत खून", "खून बंद नहीं",
             "ज़हर", "ओवरडोज़", "सीने में दर्द", "छाती में दर्द", "पसीना", "चेहरा टेढ़ा",
             "बोल नहीं", "लकवा", "जान देना", "आत्महत्या", "मरना चाहता", "दूध नहीं पी",
             "ढीली पड़", "ढीला पड़", "सांप", "दम घुट"],
    "hinglish": ["saans nahi", "sans nahi", "saans lene mein", "saans lene me", "saans phool",
                 "honth neele", "hont neele", "behosh", "jawab nahi", "dora", "daura", "jhatke",
                 "bahut bleeding", "zyada bleeding", "bahut khoon", "khoon band nahi", "zeher",
                 "zahar", "overdose", "seene mein dard", "seene me dard", "seene mein bahut",
                 "chhati mein dard", "chhati me dard", "paseena", "pasina", "chehra tedha",
                 "bol nahi pa", "lakwa", "jaan dena", "jaan de dunga", "khudkushi", "marna chahta",
                 "doodh nahi pi", "dheeli pad", "dheela pad", "saanp", "sanp", "dam ghut"],
}
URGENT = {
    "en": ["fever", "vomit", "vomiting", "deep cut", "cut", "fracture", "broken", "swollen",
           "swelling", "burn", "burns", "diarrhoea", "diarrhea", "severe pain", "bite",
           "stitches", "infection", "pregnant", "pregnancy"],
    "deva": ["बुखार", "उल्टी", "गहरा कट", "कट गया", "फ्रैक्चर", "हड्डी टूट", "सूजन", "जल गया",
             "दस्त", "तेज़ दर्द", "गर्भवती"],
    "hinglish": ["bukhar", "bukhaar", "ulti", "gehra kat", "gehra cut", "chot", "fracture",
                 "haddi toot", "sujan", "jal gaya", "dast", "loose motion", "tez dard",
                 "kat gaya", "pregnant", "garbhwati"],
}
ROUTINE = {
    "en": ["mild", "cough", "cold", "checkup", "check up", "follow up", "followup", "refill",
           "report", "reports", "appointment", "routine", "sugar test", "bp check",
           "back pain", "headache"],
    "deva": ["हल्की", "खांसी", "जुकाम", "चेकअप", "रिपोर्ट", "फॉलो", "अपॉइंटमेंट", "अपॉइन्टमेंट",
             "पुरानी", "दवाई खत्म"],
    "hinglish": ["halki", "khansi", "khasi", "zukam", "sardi", "checkup", "check up", "report",
                 "reports", "follow up", "followup", "appointment", "appoinment", "purana",
                 "dawai khatam", "refill"],
}


def norm(text):
    t = unicodedata.normalize("NFC", text).lower()
    t = t.replace("़", "")          # drop nukta so ज़ == ज
    t = t.replace("ँ", "ं")    # chandrabindu -> anusvara
    t = t.replace("'", "").replace("’", "")   # can't -> cant
    t = re.sub(r"[.,!?;:\"()\[\]।\-]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _compile(lex):
    terms = sorted({norm(t) for ts in lex.values() for t in ts}, key=len, reverse=True)
    return re.compile(r"(?<![a-z0-9])(?:" + "|".join(re.escape(t) for t in terms) + r")(?![a-z0-9])")


_E, _U, _R = _compile(EMERGENCY), _compile(URGENT), _compile(ROUTINE)


def predict(text):
    t = norm(text)
    if _E.search(t):
        return "EMERGENCY"
    if _U.search(t):
        return "URGENT"
    if _R.search(t):
        return "ROUTINE"
    return "INSUFFICIENT"
