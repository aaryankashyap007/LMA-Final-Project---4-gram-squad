"""Rule-based verifier for the assistant's replies: no diagnosis, no medicine, no dose, no treatment advice.

This is a filter, not a medical adviser. It decides only whether a reply contains something the assistant must never
say (a named medicine, a dose or schedule, a diagnosis claim, a home-remedy recommendation) and, if so, swaps the
reply for a fixed text that sends the patient to a doctor or staff. It is deliberately conservative: a wrongly blocked
reply costs a less helpful answer, a missed one is medical advice from a booking assistant.

The word lists are written by the authors from general knowledge of common over-the-counter drugs and diseases. They
are NOT clinically reviewed and not complete. A reply that gives advice without using any listed word, for example
"take the usual fever tablet", is not caught: the evaluation in scripts/analyze_verifier.py reports those misses.

Words that appear in a plain refusal ("I cannot suggest any medicine or dose") are not flagged on their own: the
generic words medicine, dawai, tablet and dose are allowed. A hit needs a specific name, a number with a unit or a
frequency, a diagnosis phrase, or a recommendation of a home remedy.
"""
import re

# ---- specific medicines and drug classes (English, Hinglish, Devanagari) ----
MEDICINES = [
    "paracetamol", "acetaminophen", "crocin", "dolo", "calpol", "ibuprofen", "brufen", "combiflam", "aspirin", "disprin",
    "diclofenac", "voveran", "naproxen", "amoxicillin", "amoxycillin", "augmentin", "azithromycin", "azithral", "ciprofloxacin",
    "cipro", "doxycycline", "metronidazole", "flagyl", "cetirizine", "citrizine", "levocetirizine", "allegra", "montelukast",
    "pantoprazole", "pantop", "omeprazole", "rantac", "ranitidine", "eno", "digene", "loperamide", "imodium", "ors",
    "metformin", "glimepiride", "insulin", "amlodipine", "atenolol", "losartan", "telmisartan", "atorvastatin", "prednisolone",
    "dexamethasone", "salbutamol", "asthalin", "vicks", "benadryl", "corex", "ascoril", "otrivin", "betadine", "dettol",
    "antibiotic", "antibiotics", "painkiller", "painkillers", "antacid", "antihistamine", "steroid", "steroids", "antiviral",
    "cough syrup", "oral rehydration",
    "पैरासिटामोल", "क्रोसिन", "डोलो", "आइबुप्रोफेन", "इबुप्रोफेन", "एस्पिरिन", "डिस्प्रिन", "कॉम्बिफ्लाम", "एमोक्सिसिलिन",
    "एंटीबायोटिक", "एंटीबायोटिक्स", "पेनकिलर", "दर्द की गोली", "बुखार की गोली", "खाँसी का सिरप", "खांसी का सिरप",
    "ओआरएस", "इंसुलिन", "मेटफॉर्मिन", "सिट्रीज़िन", "सेटिरिज़िन", "एंटासिड", "स्टेरॉयड",
    "dard ki goli", "bukhar ki goli", "khansi ka syrup", "khasi ka syrup",
]
# ---- doses, quantities, schedules ----
DOSE_PATTERNS = [
    r"\b\d+(?:[.,]\d+)?\s?(?:mg|mcg|gm|g|ml|iu|units?)\b",
    r"\b\d+\s?(?:tablets?|tabs?|capsules?|caps|drops?|teaspoons?|tsp|tablespoons?|sachets?|puffs?)\b",
    r"\b(?:one|two|three|half|1|2|3)\s+(?:tablets?|tabs?|capsules?|spoons?|teaspoons?)\b",
    r"\b(?:once|twice|thrice)\s+(?:a|per|every)\s+(?:day|night|week)\b",
    r"\b(?:once|twice|thrice)\s+daily\b",
    r"\bevery\s+\d+\s+hours?\b",
    r"\b\d+\s+times\s+(?:a|per)\s+day\b",
    r"\bdin\s+(?:mein|me)\s+(?:ek|do|teen|\d+)\s+(?:baar|bar)\b",
    r"\b(?:ek|do|teen|aadhi)\s+(?:goli|tablet|chammach)\b",
    r"[०-९0-9]+\s?(?:मिग्रा|मि\.ग्रा\.|एमजी|मिलीग्राम|एमएल|मिलीलीटर|गोली|गोलियाँ|टैबलेट|चम्मच)",
    r"(?:एक|दो|तीन|आधी)\s+(?:गोली|टैबलेट|चम्मच)",
    r"दिन\s+में\s+(?:एक|दो|तीन|[०-९0-9]+)\s+बार",
    r"खुराक\s+(?:है|होगी|लें|लीजिए)",
]
# ---- diagnosis claims: a disease word near a "you have / this is" cue ----
DISEASES = [
    "dengue", "malaria", "typhoid", "diabetes", "diabetic", "hypertension", "high blood pressure", "asthma", "pneumonia",
    "tuberculosis", "tb", "migraine", "covid", "corona", "flu", "influenza", "viral fever", "viral infection",
    "bacterial infection", "infection", "food poisoning", "gastritis", "ulcer", "appendicitis", "anaemia", "anemia",
    "jaundice", "hepatitis", "arthritis", "thyroid", "cholesterol", "kidney stone", "uti", "allergy", "bronchitis",
    "डेंगू", "मलेरिया", "टाइफाइड", "डायबिटीज", "डायबिटीज़", "मधुमेह", "शुगर", "ब्लड प्रेशर", "बीपी", "अस्थमा", "दमा", "निमोनिया",
    "टीबी", "माइग्रेन", "कोविड", "कोरोना", "फ्लू", "वायरल", "इन्फेक्शन", "संक्रमण", "फूड पॉइज़निंग", "गैस्ट्राइटिस", "अल्सर",
    "एनीमिया", "पीलिया", "हेपेटाइटिस", "गठिया", "थायराइड", "कोलेस्ट्रॉल", "एलर्जी",
    "viral", "sugar", "bp",
]
CUE_BEFORE = (r"(?:you(?:\s+(?:probably|likely|may|might|could|seem\s+to|appear\s+to))?\s+have|you\s+(?:are|re)\s+(?:probably|likely)?"
              r"|this\s+(?:is|looks\s+like|sounds\s+like|could\s+be|might\s+be|may\s+be)|it\s+(?:is|looks\s+like|sounds\s+like|could\s+be|might\s+be|may\s+be)"
              r"|sounds\s+like|looks\s+like|diagnos\w+|suffering\s+from|sign\s+of|point(?:s|ing)?\s+to|indicat\w+|aapko|apko|tumhe|yeh|ye|lagta\s+hai|आपको|यह|ये|लगता\s+है|हो\s+सकता\s+है|हो\s+सकती\s+है)")
DIAGNOSIS_WINDOW = 45
# ---- recommendations of a remedy or a treatment ----
REMEDY_PATTERNS = [
    r"\b(?:try|take|drink|use|apply|have|eat|gargle\s+with|steam)\s+(?:some\s+|a\s+little\s+|warm\s+|hot\s+)?(?:turmeric|haldi|ginger|adrak|honey|shahad|kadha|tulsi|ajwain|steam|salt\s+water|lemon|ice|a\s+cold\s+compress)\b",
    r"\bhome\s+remed(?:y|ies)\b",
    r"\b(?:you\s+should|you\s+can|you\s+may|i\s+(?:suggest|recommend)|better\s+to|best\s+to)\s+(?:take|try|use|drink|apply|stop\s+taking|skip)\s+(?:the\s+|some\s+|a\s+|your\s+)?(?:tablet|tablets|pill|pills|syrup|capsule|ointment|cream|medicine|medication)\b",
    r"\b(?:stop|skip|reduce|increase|double)\s+(?:taking\s+)?(?:your|the)\s+(?:medicine|medication|tablets?|pills?|insulin|dose)\b",
    r"\b(?:kadha|haldi|adrak|tulsi|ajwain|shahad)\s+(?:piy?(?:jiye|yo|en|e)|lijiye|lo|len|khaiye)\b",
    r"(?:हल्दी|अदरक|तुलसी|अजवाइन|शहद|काढ़ा|काढा|गरारे|भाप)\s+(?:लें|लीजिए|पिएँ|पीजिए|पिएं|खाएँ|खाइए|कीजिए)",
    r"घरेलू\s+(?:नुस्खा|नुस्ख़े|उपाय|इलाज)",
    r"(?:दवा|दवाई|गोली)\s+(?:लें|लीजिए|खाएँ|खाइए|बंद\s+कर|छोड़\s+दें)",
]

_NAMES = "|".join(re.escape(m) for m in sorted(MEDICINES, key=len, reverse=True))
# word boundaries that also work for Devanagari (\b does not split on matras reliably)
_MED_RE = re.compile(rf"(?<![\wऀ-ॿ])(?:{_NAMES})(?![\wऀ-ॿ])", re.I)
_DOSE_RES = [re.compile(p, re.I) for p in DOSE_PATTERNS]
_REMEDY_RES = [re.compile(p, re.I) for p in REMEDY_PATTERNS]
_DIS = "|".join(re.escape(d) for d in sorted(DISEASES, key=len, reverse=True))
_DIS_RE = re.compile(rf"(?<![\wऀ-ॿ])(?:{_DIS})(?![\wऀ-ॿ])", re.I)
_CUE_RE = re.compile(rf"(?<![\wऀ-ॿ])(?:{CUE_BEFORE})(?![\wऀ-ॿ])", re.I)

# A refusal that repeats the patient's own word ("I cannot recommend antibiotics", "I cannot say whether it is dengue")
# contains a listed name but gives no advice. With refusal_aware=True such a mention is not a hit, provided the reply
# is a refusal, has no dose or schedule, and uses no verb that recommends or instructs (take, try, you can, lijiye...).
# This was added after the first live run, where 31 of 33 blocked replies were refusals echoing the patient's word.
REFUSAL_RE = re.compile(
    r"\b(?:can(?:'|’)t|cannot|unable to|not able to|won(?:'|’)t|not allowed to|do not|don(?:'|’)t|"
    r"nahi\s+(?:bata|de)\s+sak\w*|nahin\s+bata\w*)\b|नहीं\s+(?:दे|बता|कर)\s+सक|मैं\s+डॉक्टर\s+नहीं|"
    r"मेरे\s+पास\s+यह\s+जानकारी\s+नहीं",
    re.I)
RECOMMEND_RE = re.compile(
    r"\b(?:you\s+(?:can|should|may|could|might)|try|take|use|apply|drink|have\s+some)\b|"
    r"\b(?:lijiye|lijie|leejiye|le\s+lo|lena|piyen|piye)\b|लें|लीजिए|(?<![\wऀ-ॿ])लो(?![\wऀ-ॿ])|कीजिए|पिएँ|पिएं",
    re.I)

CONTRAST_RE = re.compile(
    r"\b(?:but|however|although|though|par|lekin|magar)\b|लेकिन|मगर|(?<![\wऀ-ॿ])पर(?![\wऀ-ॿ])",
    re.I)       # a "but" AFTER the refusal turns it round: "I cannot say for sure, but it sounds like dengue"

SAFE_TEXT_EN = ("I cannot give medical advice, medicines or doses. A doctor or our staff will help you with that. "
                "I can help you book an appointment.")
SAFE_TEXT_HI = ("मैं चिकित्सा सलाह, दवा या खुराक नहीं बता सकता। इसमें डॉक्टर या हमारा स्टाफ आपकी मदद करेगा। "
                "मैं अपॉइंटमेंट बुक करने में मदद कर सकता हूँ।")


class Verifier:
    """check(text) -> list of hits ("medicine:...", "dose:...", "diagnosis:...", "remedy:..."); callable(text) -> text.
    With enforce=False (shadow mode) it only reports and never changes a reply."""

    def __init__(self, enforce=True, refusal_aware=True):
        self.enforce = enforce
        self.refusal_aware = refusal_aware

    def check(self, text):
        t = text or ""
        hits = [f"medicine:{m.group(0).lower()}" for m in _MED_RE.finditer(t)]
        for r in _DOSE_RES:
            hits += [f"dose:{m.group(0).lower()}" for m in r.finditer(t)]
        for r in _REMEDY_RES:
            hits += [f"remedy:{m.group(0).lower()}" for m in r.finditer(t)]
        for m in _DIS_RE.finditer(t):
            before = t[max(0, m.start() - DIAGNOSIS_WINDOW):m.start()]
            after = t[m.end():m.end() + DIAGNOSIS_WINDOW]
            if _CUE_RE.search(before) or _CUE_RE.search(after):
                hits.append(f"diagnosis:{m.group(0).lower()}")
        refusal = REFUSAL_RE.search(t) if (self.refusal_aware and hits) else None
        if (refusal and not RECOMMEND_RE.search(t) and not CONTRAST_RE.search(t[refusal.end():])
                and not any(h.startswith("dose:") for h in hits)):
            return []                                           # a refusal that only repeats the patient's word
        return hits

    @staticmethod
    def safe_text(text):
        return SAFE_TEXT_HI if re.search("[ऀ-ॿ]", text or "") else SAFE_TEXT_EN

    def __call__(self, text):
        return self.safe_text(text) if (self.enforce and self.check(text)) else text
