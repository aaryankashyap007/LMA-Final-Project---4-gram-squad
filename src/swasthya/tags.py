"""Red-flag tags and intents for a patient message (deterministic, multilingual, no model).

This is the tagging half of the gate output. The urgency label comes from G1; the tags and intents come from this
tagger. It is a lexicon matcher with negation handling over `normaliser.for_match` text, so it works on Devanagari,
romanised Hindi and English, including code-mixed text. It is a draft: the red-flag list and every term need
clinician and native-speaker review, and it is validated only against tags derived from D3's cue lists
(scripts/eval_tagger.py). The intent rules have no labelled evaluation data.

Negation: a match is dropped when a negator ("no", "not", "without", "nahi", "नहीं", ...) sits within 3 tokens
before it, or a Hindi negator within 2 tokens after it ("seene mein dard nahi hai"). Terms that contain their own
negator ("saans nahi", "cannot breathe") are matched as phrases, so they are not cancelled by it.
"""
import re
from dataclasses import dataclass, field
from typing import List

from .normaliser import detect_script, for_match, for_model

RED_FLAGS = {
    "breathing_difficulty": {
        "en": ["cant breathe", "cannot breathe", "can not breathe", "not breathing", "stopped breathing",
               "difficulty breathing", "trouble breathing", "short of breath", "shortness of breath", "breathless",
               "breathlessness", "gasping", "suffocating", "suffocated", "choking", "blue lips",
               "lips are turning blue", "turning blue", "turning purple", "slow breathing",
               "struggling to breathe", "throat closing", "throat is closing", "choke", "choked", "breathing heavily", "breathing very heavily"],
        "deva": ["सांस नहीं", "सांस लेने में तकलीफ", "सांस लेने में दिक्कत", "सांस फूल", "सांस रुक", "दम घुट",
                 "होंठ नीले", "गला बंद", "सांस धीमी", "घुट रहा", "घुट रही", "सांसें ले रहे", "हवा के लिए तड़प"],
        "hinglish": ["saans nahi", "saans lene mein", "saans lene me", "saans phool", "saans ruk", "saans dheemi",
                     "dam ghut", "honth neele", "hont neele", "gala band", "choke ho", "saans ke liye tadap", "bhaari saansein"]},
    "unresponsive_or_collapse": {
        "en": ["unconscious", "unresponsive", "not responding", "not responsive", "not waking", "wont wake",
               "fainted", "passed out", "collapsed", "barely conscious", "lost consciousness",
               "losing consciousness", "blacked out", "blackout", "not waking up", "unarousable", "about to collapse", "collapse", "fainting", "lost all consciousness", "losing consciousness rapidly"],
        "deva": ["बेहोश", "जवाब नहीं दे", "होश नहीं", "नहीं उठ", "गिर पड़", "बेसुध", "बमुश्किल होश", "होश खो", "बेहोशी"],
        "hinglish": ["behosh", "jawab nahi", "hosh nahi", "nahi uth", "responsive nahi", "gir pada", "gir padi", "mushkil se hosh", "hosh kho", "consciousness kho", "behoshi"]},
    "seizure": {
        "en": ["seizure", "seizures", "convulsion", "convulsions", "fits", "frothing", "epileptic",
               "shaking uncontrollably"],
        "deva": ["दौरा", "दौरे", "झटके", "मिर्गी", "झाग"],
        "hinglish": ["dora", "jhatke", "mirgi", "jhaag"]},
    "severe_bleeding": {
        "en": ["heavy bleeding", "bleeding heavily", "bleeding a lot", "severe bleeding", "bleeding will not stop",
               "bleeding wont stop", "wont stop bleeding", "not stopping", "spurting", "lot of blood",
               "blood loss", "vomiting blood", "coughing blood", "heavy vaginal bleeding", "profuse bleeding", "bleeding profusely", "continuous bleeding", "blood clots", "gushing", "soaking pads", "pads soaked"],
        "deva": ["ज्यादा ब्लीडिंग", "बहुत ब्लीडिंग", "बहुत खून", "खून बंद नहीं", "खून रुक नहीं", "खून की उल्टी",
                 "खून बह", "तेज खून", "तेज ब्लीडिंग", "खून निकल", "खून के थक्के", "पैड पूरी तरह भीग"],
        "hinglish": ["bahut bleeding", "zyada bleeding", "bahut khoon", "khoon band nahi", "khoon ruk nahi",
                     "khoon ki ulti", "khoon beh", "khoon nikal", "profuse bleeding", "continuous bleeding", "khoon bah", "khoon ke thakke"]},
    "poisoning_overdose": {
        "en": ["poison", "poisoning", "overdose", "swallowed", "ingested", "pesticide", "rat poison", "kerosene",
               "bleach", "floor cleaner", "sleeping pills", "drank acid"],
        "deva": ["जहर", "ओवरडोज", "कीटनाशक", "मिट्टी का तेल", "तेजाब", "गोलियां खा", "गोलियाँ खा", "पत्ता खा"],
        "hinglish": ["zeher", "overdose", "keetnashak", "mitti ka tel", "tezaab", "goliyan kha", "goliyon ka"]},
    "chest_pain_cardiac": {
        "en": ["chest pain", "chest pressure", "chest tightness", "tightness in chest", "chest discomfort",
               "pain in chest", "chest squeezing", "crushing", "heart attack", "radiating", "pain in left arm",
               "left arm pain"],
        "deva": ["सीने में दर्द", "छाती में दर्द", "सीने में जकड़न", "सीने पर दबाव", "दिल का दौरा", "सीने में भारीपन",
                 "सीने में तेज"],
        "hinglish": ["seene mein dard", "seene me dard", "seene mein bahut", "seene mein tez", "chhati mein dard",
                     "chhati me dard", "seene mein jakdan", "dil ka dora", "seene pe dabav"]},
    "stroke_signs": {
        "en": ["face drooping", "drooping face", "face is drooping", "mouth is crooked", "crooked mouth", "slurred",
               "cannot speak", "cant speak", "difficulty speaking", "paralysis", "paralyzed", "paralysed",
               "weakness on one side", "left side is weak", "right side is weak", "left side weak",
               "right side weak", "sudden weakness", "sudden numbness", "sudden blindness", "loss of vision",
               "thunderclap", "worst headache", "stroke", "lost vision", "vision loss", "cannot utter", "single word", "droop", "paralyze", "cannot recognize", "excruciating headache", "lost all sensation"],
        "deva": ["चेहरा टेढ़ा", "मुंह टेढ़ा", "मुँह टेढ़ा", "बोल नहीं", "बोलने में दिक्कत", "लकवा", "एक तरफ कमजोर",
                 "हिस्सा कमजोर", "आवाज लड़खड़ा", "दिखाई नहीं", "एक शब्द", "लड़खड़ा", "रोशनी चली", "असहनीय सिरदर्द", "चेहरा एक तरफ"],
        "hinglish": ["chehra tedha", "munh tedha", "bol nahi", "bolne mein dikkat", "lakwa", "ek taraf kamzor",
                     "side kamzor", "jubaan lad", "dikhai nahi", "ek word", "ladkhada", "speech chali", "vision chali", "face ek taraf"]},
    "suicidal_intent": {
        "en": ["suicide", "suicidal", "kill myself", "end my life", "end it all", "want to die", "wanna die",
               "harm myself", "hurt myself", "no reason to live", "cant take it anymore", "cannot take it anymore", "kill himself", "kill herself", "kill themselves", "end his life", "end her life", "urge to jump", "jump off", "intentionally to end"],
        "deva": ["आत्महत्या", "जान देना", "जान दे दूं", "मरना चाहता", "मरना चाहती", "खुद को खत्म", "खुद को नुकसान", "जीवन लीला", "अपनी जान", "जान देने", "खुद की जान", "कूद जाने"],
        "hinglish": ["khudkushi", "jaan dena", "jaan de dunga", "jaan de dungi", "marna chahta", "marna chahti",
                     "khud ko khatam", "khud ko nuksan", "life end karne", "apni jaan", "jaan dene", "jump karne", "khud ki jaan"]},
    "pregnancy_danger": {
        "en": ["vaginal bleeding", "water broke", "waters broke", "labour pain", "labor pain", "contractions",
               "baby head", "baby is not moving", "baby not moving", "not feeling baby move", "baby is coming"],
        "deva": ["पानी की थैली फट", "प्रसव पीड़ा", "बच्चे का सिर", "बच्चा हिल नहीं", "बच्चा नहीं हिल", "योनि से खून"],
        "hinglish": ["paani ki thaili phat", "labour pain", "bachche ka sir", "baccha hil nahi", "baccha nahi hil",
                     "yoni se khoon"]},
    "dehydration_or_lethargy": {
        "en": ["not drinking", "not feeding", "not passing urine", "no urine", "not peed", "limp", "floppy",
               "very sleepy", "very drowsy", "hard to wake", "difficult to wake", "dark urine", "sunken eyes", "lethargic", "refusing feeds", "refusing all feeds", "ice cold", "dark brown urine", "black urine"],
        "deva": ["कुछ पी नहीं", "दूध नहीं पी", "पेशाब नहीं", "ढीला पड़", "ढीली पड़", "बहुत सो रहा", "बहुत सो रही",
                 "उठ नहीं रहा", "गहरे रंग का पेशाब", "काला पेशाब", "गहरे भूरे", "बहुत सुस्त"],
        "hinglish": ["kuch pi nahi", "doodh nahi pi", "peshab nahi", "dheela pad", "dheeli pad", "bahut so raha",
                     "bahut so rahi", "uth nahi raha", "peshab ka rang gehra", "feed nahi", "lethargic", "kala peshab", "bahut sust"]},
    "head_injury_trauma": {
        "en": ["head injury", "hit his head", "hit her head", "hit my head", "hit head", "fell from height",
               "fell from the roof", "fell from a height", "fall from height", "road accident", "accident",
               "spinal", "neck injury", "cannot feel legs", "cant feel legs", "cannot move legs", "crash", "bike accident", "machine accident", "high bed", "fell from high"],
        "deva": ["सिर में चोट", "सिर पर चोट", "ऊंचाई से गिर", "ऊँचाई से गिर", "दुर्घटना", "एक्सीडेंट", "रीढ़",
                 "पैर महसूस नहीं", "हादसा", "हादसे"],
        "hinglish": ["sir mein chot", "sir me chot", "sir par chot", "head injury", "accident", "uchai se gir",
                     "ped se gir", "chhat se gir", "reedh", "pair mehsoos nahi", "hadsa", "hadse", "takra gaya"]},
    "burn_or_deep_wound": {
        "en": ["third degree", "severe burn", "burnt", "scalded", "boiling", "deep cut", "deep wound", "stab",
               "stabbed", "severed", "amputated", "chemical splash", "acid", "cut open", "laceration", "cut off", "completely cut", "cut deeply", "skin tear", "skin tearing", "tearing skin", "bone sticking", "compound", "chemical", "burn", "burns"],
        "deva": ["जल गया", "जल गई", "जल गयी", "जला", "गहरा कट", "गहरा घाव", "चाकू", "उंगली कट", "ऊंगली कट",
                 "तेजाब", "तेज़ाब", "चमड़ी फट", "हड्डी बाहर", "केमिकल", "कट कर अलग", "जलन"],
        "hinglish": ["jal gaya", "jal gayi", "jal gaye", "gehra kat", "gehra cut", "gehra ghav", "chaku",
                     "ungli kat", "tezaab", "ubalte", "skin tear", "haddi bahar", "poori kat", "kat gayi", "chemical", "jalan", "burning pain"]},
    "animal_bite_or_sting": {
        "en": ["snake bite", "snakebite", "bitten by a snake", "snake", "scorpion", "bee sting", "stung",
               "dog bite", "dog bit", "monkey bite", "rabid", "bees", "wasp", "stings", "stray dog"],
        "deva": ["सांप", "बिच्छू", "मधुमक्खी", "कुत्ते ने काट", "बंदर ने काट", "ततैया", "ततैये", "मधुमक्खियों", "आवारा कुत्ते"],
        "hinglish": ["saanp", "bichhu", "bichchu", "madhumakkhi", "kutte ne kata", "kutte ne kat", "bandar ne kata", "bichhoo", "bees ne", "bee ne kaat", "dog ne"]},
    "allergic_reaction": {
        "en": ["allergic reaction", "swollen tongue", "tongue is swollen", "swelling of face", "face swelling",
               "face is swollen", "hives", "rash all over", "tongue swollen", "throat feels like it is closing"],
        "deva": ["एलर्जी", "जीभ सूज", "चेहरा सूज", "चकत्ते", "गला पूरी तरह बंद"],
        "hinglish": ["allergy", "jeebh sooj", "chehra sooj", "chakatte", "hives", "gala band ho"]},
}

PREGNANCY = {"en": ["pregnant", "pregnancy", "delivered", "delivery"],
             "deva": ["गर्भवती", "प्रेग्नेंट", "डिलीवरी", "प्रसव"],
             "hinglish": ["pregnant", "garbhwati", "garbh", "delivery", "prasav"]}
PREGNANCY_WITH = ["severe_bleeding", "seizure"]       # pregnancy + these tags also raises pregnancy_danger
PREGNANCY_SEVERE = {"en": ["severe abdominal pain", "severe pain", "blurred vision", "swelling"],
                    "deva": ["पेट में तेज दर्द", "धुंधला"], "hinglish": ["pet mein tez dard", "dhundhla"]}

INTENTS = {
    "book_appointment": {"strong": {
        "en": ["book", "booking", "slot"], "deva": ["बुक", "समय चाहिए", "डॉक्टर से मिल"],
        "hinglish": ["book", "slot", "time chahiye", "doctor se mil", "doctor ko dikha"]},
        "weak": {"en": ["appointment", "schedule", "consultation", "consult", "see the doctor", "see a doctor"],
                 "deva": ["अपॉइंटमेंट", "अपॉइन्टमेंट"], "hinglish": ["appointment", "appoinment"]}},
    "cancel_appointment": {"strong": {
        "en": ["cancel", "call off"], "deva": ["रद्द", "कैंसिल"], "hinglish": ["cancel", "radd", "rad kar"]}},
    "reschedule": {"strong": {
        "en": ["reschedule", "postpone", "change the time", "change my appointment", "another slot",
               "different time", "move my appointment", "another day", "different day", "other day"],
        "deva": ["समय बदल", "दूसरा समय", "दूसरी तारीख", "आगे बढ़ा", "किसी और दिन", "दूसरे दिन"],
        "hinglish": ["reschedule", "time badal", "dusra time", "dusra slot", "aage badha", "postpone", "dusre din",
                     "dusra din", "kisi aur din"]}},
    "appointment_info": {"strong": {
        "en": ["my appointment", "my appointments", "my upcoming appointments", "upcoming appointments",
               "when is my", "what time is my", "appointment status", "appointment details"],
        "deva": ["मेरा अपॉइंटमेंट", "मेरी अपॉइंटमेंट", "मेरा अपॉइन्टमेंट", "मेरी अपॉइन्टमेंट", "मेरे अपॉइंटमेंट",
                 "आने वाले अपॉइंटमेंट"],
        "hinglish": ["mera appointment", "meri appointment", "mera slot", "appointment kab",
                     "appointment kitne baje", "mere appointments", "mere upcoming appointments",
                     "upcoming appointments"]}},
    "clinic_info": {"strong": {
        "en": ["opening hours", "open on", "opening time", "closing time", "what time does the clinic",
               "when does the clinic", "clinic open", "clinic closed", "where is the clinic", "clinic address",
               "address", "what to bring", "documents", "fees", "fee", "charges", "how much does", "cost",
               "directions", "parking"],
        "deva": ["खुलता", "खुला", "बंद होता", "क्लिनिक का पता", "पता क्या", "फीस", "कितना खर्च", "कितने पैसे",
                 "क्या लाना", "कौन से कागज", "कागजात"],
        "hinglish": ["kab khulta", "kab tak khula", "kab band", "baje khulta", "khulta hai", "khula rehta",
                     "khula hai", "baje khul", "timing", "address", "fees", "fee", "kitne paise",
                     "kitna kharcha", "kya lana", "kaun se kagaz", "kagaz"]}},
    "medical_advice": {"strong": {
        "en": ["what medicine", "which medicine", "what tablet", "which tablet", "what drug", "what should i take",
               "should i take", "can i take", "is it safe to take", "dose", "dosage", "how many tablets",
               "paracetamol", "ibuprofen", "antibiotic", "antibiotics", "painkiller", "what is wrong with me",
               "diagnose", "diagnosis", "home remedy", "cure"],
        "deva": ["कौन सी दवा", "कौन सी दवाई", "कौन सा दवा", "क्या दवा", "क्या दवाई", "कितनी गोली", "कितनी खुराक",
                 "खुराक", "पैरासिटामोल", "एंटीबायोटिक", "दर्द की दवा", "क्या हुआ है मुझे", "घरेलू इलाज",
                 "इलाज बताइए", "क्या खाऊं", "क्या लूं"],
        "hinglish": ["kaun si dawai", "kaun si dawa", "kaunsi dawai", "kaunsi dawa", "kya dawai", "kya dawa",
                     "kitni goli", "kitna dose", "dose kitna", "paracetamol", "antibiotic", "dard ki dawai",
                     "mujhe kya hua hai", "gharelu ilaj", "ilaj batao", "kya khaun", "kya lun", "dawai batao",
                     "dawa batao"]}},
    "symptom_report": {"strong": {
        "en": ["pain", "ache", "fever", "cough", "cold", "vomit", "vomiting", "nausea", "dizzy", "dizziness", "rash",
               "swelling", "swollen", "bleeding", "diarrhoea", "diarrhea", "headache", "sore", "weak", "weakness",
               "itching", "injury", "unwell", "sick", "not feeling well"],
        "deva": ["दर्द", "बुखार", "खांसी", "उल्टी", "चक्कर", "सूजन", "दस्त", "कमजोरी", "खुजली", "चोट", "तबीयत",
                 "बीमार", "जुकाम", "घबराहट"],
        "hinglish": ["dard", "bukhar", "khansi", "ulti", "chakkar", "sujan", "dast", "kamzori", "kamzor", "khujli",
                     "chot", "tabiyat", "bimar", "zukam", "ghabrahat", "loose motion", "sardi"]}},
}

# Proximity rules: a tag also fires when a word from A and a word from B occur within `window` tokens (either
# order) and the pair is not negated. They cover phrasing that exact phrases miss ("breathing very heavily",
# "bitten on the leg by a stray dog"). Words go through for_match, so spelling variants are handled.
PROXIMITY = {
    "breathing_difficulty": [
        (["breathing", "breathe", "breath", "saans", "saansein", "saanse", "सांस", "सांसें", "सांसे"],
         ["problem", "issue", "issues", "trouble", "difficulty", "heavily", "heavy", "fast", "slow", "struggling",
          "dikkat", "taklif", "tadap", "bhaari", "bhari", "dheemi", "तकलीफ", "दिक्कत", "धीमी", "भारी", "तड़प"], 4),
        (["saans", "saansein", "सांस", "सांसें"], ["nahi", "नहीं"], 2),     # "saans bhi nahi aa rahi"
        (["throat", "gala", "गला"], ["closing", "band", "बंद"], 4),
        (["skin", "face", "lips", "त्वचा", "चेहरा", "होंठ", "honth", "hont"],
         ["blue", "purple", "neela", "neeli", "neele", "नीला", "नीली", "नीले", "बैंगनी"], 3)],
    "unresponsive_or_collapse": [
        (["consciousness", "hosh", "होश"], ["kho", "khoraha", "losing", "lost", "खो", "mushkil", "बमुश्किल"], 3)],
    "severe_bleeding": [
        (["bleeding", "ब्लीडिंग", "khoon", "खून", "blood", "रक्त"],
         ["profuse", "profusely", "continuous", "gushing", "heavy", "heavily", "severe", "tez", "तेज", "bahut",
          "बहुत", "zyada", "ज्यादा", "nikal", "निकल", "beh", "bah", "बह", "clots", "थक्के", "thakke"], 3)],
    "chest_pain_cardiac": [
        (["chest", "seene", "सीने", "छाती", "chhati"],
         ["pain", "tight", "tightness", "pressure", "squeezing", "squeeze", "discomfort", "crushing", "stabbing",
          "heavy", "dard", "दर्द", "दबाव", "जकड़न", "jakdan", "भारी", "bhaari"], 5)],
    "stroke_signs": [
        (["face", "chehra", "चेहरा", "munh", "मुंह", "मुँह"],
         ["droop", "drooping", "droops", "crooked", "tedha", "टेढ़ा", "लटक", "lataka"], 4),
        (["speech", "boli", "बोली"], ["slurred", "ladkhada", "लड़खड़ा", "gone", "chali", "lost"], 4),
        (["vision", "रोशनी", "eye", "aankh", "आंख"], ["lost", "gone", "chali", "चली"], 5),
        (["weakness", "kamzori", "कमजोरी", "paralyze", "paralyzed"],
         ["left", "right", "side", "बाएं", "दाएं", "दाहिने", "taraf", "तरफ"], 4),
        (["headache", "सिरदर्द"], ["worst", "thunderclap", "excruciating", "असहनीय"], 4)],
    "suicidal_intent": [
        (["end", "ending", "khatam", "समाप्त", "खत्म"],
         ["life", "jeevan", "जीवन", "zindagi", "जिंदगी"], 3),
        (["kill"], ["myself", "himself", "herself", "themselves"], 2),
        (["jump", "कूद", "kood"], ["balcony", "ledge", "edge", "urge", "बालकनी", "किनारे"], 6)],
    "dehydration_or_lethargy": [
        (["urine", "peshab", "पेशाब"], ["dark", "brown", "black", "kala", "काला", "काली", "गहरे"], 4)],
    "head_injury_trauma": [
        (["head", "sir", "सिर"], ["injury", "hit", "bump", "struck", "chot", "चोट", "takra", "टकरा"], 4)],
    "burn_or_deep_wound": [
        (["cut", "kat", "कट"], ["deep", "deeply", "gehra", "गहरा", "completely", "poori", "पूरी", "off", "अलग"], 3),
        (["burn", "burning", "burnt", "jalan", "जलन"], ["severe", "bahut", "बहुत", "bhayankar", "भयंकर", "chemical"], 4)],
    "animal_bite_or_sting": [
        (["dog", "kutte", "कुत्ते", "कुत्ता", "monkey", "bandar", "बंदर", "bee", "bees", "bichhoo", "bichhu",
          "scorpion", "snake", "saanp", "सांप", "wasp", "ततैये", "ततैया"],
         ["bit", "bitten", "bite", "kaat", "kaata", "kata", "काट", "sting", "stung", "stings", "डंक"], 6)],
    "allergic_reaction": [
        (["tongue", "jeebh", "जीभ"], ["swollen", "sooj", "सूज", "swelling"], 4)],
}

NEG_BEFORE = {"no", "not", "never", "without", "nothing", "none", "dont", "doesnt", "didnt", "isnt", "wasnt",
              "nahi", "bina", "koi", "na", "नहीं", "बिना", "कोई", "ना", "न"}
NEG_AFTER = {"nahi", "nahin", "नहीं", "na", "ना", "न"}
_WORD = r"[\wऀ-ॿ]"


def _compile(terms):
    """One regex for a list of terms. Latin terms need a word boundary on both sides; Devanagari terms only on the
    left, because Hindi suffixes attach to the word (बेहोश -> बेहोशी)."""
    pats = []
    for t in sorted({for_match(x) for x in terms if x.strip()}, key=len, reverse=True):
        deva = bool(re.search("[ऀ-ॿ]", t))
        pats.append(f"(?<!{_WORD}){re.escape(t)}" + ("" if deva else f"(?!{_WORD})"))
    return re.compile("|".join(pats))


def _all_terms(d):
    return [t for ts in d.values() for t in ts]


_RED = {tag: _compile(_all_terms(lex)) for tag, lex in RED_FLAGS.items()}
_PREG = _compile(_all_terms(PREGNANCY))
_PREG_SEVERE = _compile(_all_terms(PREGNANCY_SEVERE))
_INT = {name: {kind: _compile(_all_terms(lex)) for kind, lex in kinds.items()} for name, kinds in INTENTS.items()}


def _negated(toks, start, end):
    """A negator within 3 tokens before toks[start:end], or a Hindi negator within 2 tokens after it."""
    return bool(NEG_BEFORE & set(toks[max(0, start - 3):start]) or NEG_AFTER & set(toks[end:end + 2]))


def _hits(rx, text, negation=True):
    """True if the regex has at least one match that is not negated. A matched phrase that carries its own negator
    ("saans nahi", "koi jawab nahi de rahe") is never cancelled by a neighbouring one."""
    toks = text.split()
    for m in rx.finditer(text):
        if not negation or (NEG_BEFORE | NEG_AFTER) & set(m.group(0).split()):
            return True
        start, end = len(text[:m.start()].split()), len(text[:m.end()].split())
        if not _negated(toks, start, end):
            return True
    return False


def _words(ws):
    return {for_match(w) for w in ws}


_PROX = {tag: [(_words(a), _words(b), w) for a, b, w in rules] for tag, rules in PROXIMITY.items()}


def _prox_hit(rules, toks):
    for A, B, w in rules:
        ia = [i for i, t in enumerate(toks) if t in A]
        ib = [i for i, t in enumerate(toks) if t in B]
        for i in ia:
            for j in ib:
                if i != j and abs(i - j) <= w and not _negated(toks, min(i, j), max(i, j) + 1):
                    return True
    return False


def red_flags(text):
    """Sorted red-flag tags found in a message (any language mix)."""
    t = for_match(text)
    toks = t.split()
    found = {tag for tag, rx in _RED.items() if _hits(rx, t) or _prox_hit(_PROX.get(tag, []), toks)}
    if _hits(_PREG, t) and (found & set(PREGNANCY_WITH) or _hits(_PREG_SEVERE, t)):
        found.add("pregnancy_danger")
    return sorted(found)


def intents(text, flags=None):
    """Sorted intents of a message; ["other"] if none apply. Weak booking words ("appointment") only count when no
    cancel, reschedule or info intent is present."""
    t = for_match(text)
    found = {name for name, kinds in _INT.items() if _hits(kinds["strong"], t, negation=name == "symptom_report")}
    weak = _INT["book_appointment"]["weak"]
    if "book_appointment" not in found and _hits(weak, t, negation=False) and not (
            found & {"cancel_appointment", "reschedule", "appointment_info"}):
        found.add("book_appointment")
    flags = red_flags(text) if flags is None else flags
    if flags:
        found.add("symptom_report")
    return sorted(found) or ["other"]


# ---- administrative-only messages -------------------------------------------------------------------------
# G1 was trained on symptom messages, so it calls plain administrative requests ("cancel my appointment", "clinic
# hours") INSUFFICIENT: 79% of them as a first message on the probe, every Devanagari one. `administrative_only`
# identifies those messages so the orchestrator can treat an INSUFFICIENT label on them as ROUTINE. It is deliberately
# strict, because D3 contains genuinely vague requests that look administrative ("Appointment for heart issue",
# "Depression doctor appointment", "Doctor appointment required") and those must keep their INSUFFICIENT label:
#   - only administrative intents, no red flag, no symptom, no medical-advice request;
#   - no clinical vocabulary at all (an area of medicine, "problem", "worried", ...);
#   - a booking needs concrete specifics (a named doctor, a date or a time); cancel, reschedule, my-appointments
#     and clinic-information requests do not seek care, so they need none.
ADMIN_INTENTS = {"book_appointment", "cancel_appointment", "reschedule", "appointment_info", "clinic_info"}
NOT_SEEKING_CARE = {"cancel_appointment", "reschedule", "appointment_info", "clinic_info"}
CLINICAL_AREA = {
    "en": ["heart", "cardiac", "cardio", "maternity", "pregnant", "pregnancy", "depression", "depressed", "psychiatrist",
           "psychiatric", "mental", "cancer", "diabetes", "diabetic", "kidney", "liver", "lung", "asthma", "epilepsy",
           "fracture", "skin", "eye", "ear", "throat", "stomach", "issue", "issues", "problem", "problems", "worried",
           "worry", "sick", "unwell", "strange", "wrong", "symptom", "symptoms", "disease", "illness", "infection",
           "blood pressure", "bp", "sugar", "baby", "child", "newborn", "feel", "feeling", "treatment", "medicine"],
    "deva": ["दिल", "हृदय", "हार्ट", "मातृत्व", "गर्भ", "अवसाद", "डिप्रेशन", "मनोचिकित्सक", "मानसिक", "कैंसर", "मधुमेह",
             "शुगर", "किडनी", "फेफड़े", "अस्थमा", "हड्डी", "त्वचा", "आंख", "कान", "गला", "पेट", "समस्या", "परेशानी",
             "चिंता", "बीमार", "बीमारी", "लक्षण", "संक्रमण", "बच्चा", "बच्चे", "इलाज", "दवा", "दवाई"],
    "hinglish": ["dil", "heart", "bimari", "bimar", "problem", "pareshani", "chinta", "ghabrahat", "sugar", "bp",
                 "cancer", "kidney", "pet", "aankh", "kaan", "gala", "skin", "depression", "maternity", "tabiyat",
                 "lakshan", "bachcha", "bachche", "ilaj", "dawa", "dawai"]}
_CLINICAL = _compile(_all_terms(CLINICAL_AREA))
_DAYS = ("monday tuesday wednesday thursday friday saturday sunday tomorrow today tonight morning afternoon evening "
         "kal aaj subah dopahar shaam parso somvar mangalvar budhvar guruvar shukravar shanivar ravivar "
         "january february march april may june july august september october november december").split()
_DEVA_DAYS = ["कल", "आज", "सुबह", "दोपहर", "शाम", "परसों", "सोमवार", "मंगलवार", "बुधवार", "गुरुवार", "शुक्रवार", "शनिवार",
              "रविवार", "जनवरी", "फरवरी", "मार्च", "अप्रैल", "मई", "जून", "जुलाई", "अगस्त", "सितंबर", "अक्टूबर",
              "नवंबर", "दिसंबर"]
_WHEN = {for_match(w) for w in _DAYS + _DEVA_DAYS}
_TITLES = {"dr", "डॉ", "डा"}


def _has_specifics(t):
    """A named doctor ("Dr. Rao"), a date or part of the day, or any number."""
    toks = t.split()
    return (any(w in _WHEN or any(c.isdigit() for c in w) for w in toks)
            or any(w in _TITLES and i + 1 < len(toks) for i, w in enumerate(toks)))


def administrative_only(text, flags=None, ints=None):
    flags = red_flags(text) if flags is None else flags
    ints = set(intents(text, flags) if ints is None else ints)
    if flags or not ints or not ints <= ADMIN_INTENTS:        # any symptom, advice request or other intent blocks it
        return False
    t = for_match(text)
    if _hits(_CLINICAL, t, negation=False):
        return False
    return bool(ints & NOT_SEEKING_CARE) or _has_specifics(t)


@dataclass
class MessageInfo:
    script: str
    red_flags: List[str] = field(default_factory=list)
    intents: List[str] = field(default_factory=list)
    for_model: str = ""
    administrative: bool = False


def analyse_message(text):
    flags = red_flags(text)
    ints = intents(text, flags)
    return MessageInfo(detect_script(text), flags, ints, for_model(text), administrative_only(text, flags, ints))
