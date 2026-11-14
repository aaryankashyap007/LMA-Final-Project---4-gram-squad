import pytest

from swasthya.tags import analyse_message, intents, red_flags


@pytest.mark.parametrize("text,tag", [
    ("I cannot breathe", "breathing_difficulty"),
    ("my father is breathing very heavily", "breathing_difficulty"),         # proximity rule
    ("saans nhi aa rahi", "breathing_difficulty"),                           # spelling variant
    ("sans nhi aa rhi", "breathing_difficulty"),
    ("सांस नहीं आ रही", "breathing_difficulty"),
    ("होंठ नीले पड़ गए हैं", "breathing_difficulty"),
    ("he is unconscious", "unresponsive_or_collapse"),
    ("wo koi jawab nahi de rahe", "unresponsive_or_collapse"),               # "koi" must not cancel a phrase with its own nahi
    ("वो बेहोश हो गई", "unresponsive_or_collapse"),
    ("बे​होश हो गई", "unresponsive_or_collapse"),                       # zero-width character inside the word
    ("she is having a seizure", "seizure"), ("bacche ko dora pad raha hai", "seizure"),
    ("bahut khoon beh raha hai", "severe_bleeding"), ("the cut is bleeding profusely", "severe_bleeding"),
    ("उसने ज़हर पी लिया", "poisoning_overdose"), ("उसने जहर पी लिया", "poisoning_overdose"),
    ("he swallowed a whole strip of tablets", "poisoning_overdose"),
    ("I have chest pain", "chest_pain_cardiac"), ("mere father ko chest pain hai", "chest_pain_cardiac"),
    ("seene mein dard ho raha hai", "chest_pain_cardiac"), ("सीने में दर्द है", "chest_pain_cardiac"),
    ("face is drooping and speech is slurred", "stroke_signs"),
    ("my mother cannot utter a single word", "stroke_signs"),
    ("he says he wants to kill himself", "suicidal_intent"),
    ("I have an urge to jump from the balcony", "suicidal_intent"),
    ("mujhe khudkushi ke vichar aa rahe hain", "suicidal_intent"),
    ("my baby is not feeding", "dehydration_or_lethargy"),
    ("he fell and hit his head", "head_injury_trauma"), ("sir mein chot lagi hai", "head_injury_trauma"),
    ("he got a deep cut on his hand", "burn_or_deep_wound"), ("boiling oil se jal gaya", "burn_or_deep_wound"),
    ("a snake bit my son", "animal_bite_or_sting"), ("bitten on the leg by a stray dog", "animal_bite_or_sting"),
    ("her tongue is swollen after eating", "allergic_reaction")])
def test_red_flags_found(text, tag):
    assert tag in red_flags(text), text


@pytest.mark.parametrize("text", [
    "I have no chest pain", "she is not breathless", "seene mein dard nahi hai", "सीने में दर्द नहीं है",
    "no fever and no vomiting", "there is no bleeding", "saans mein koi dikkat nahi",
    "I have a mild cough and need a check-up", "mujhe halki khansi hai, checkup ke liye appointment chahiye",
    "I want to book an appointment", "", "ok thanks", "11 am", "haan ji"])
def test_no_red_flags_for_negated_or_routine_text(text):
    assert red_flags(text) == [], text


def test_pregnancy_danger_needs_pregnancy_plus_a_danger_sign():
    assert "pregnancy_danger" in red_flags("my wife is 8 months pregnant and has heavy bleeding")
    assert "pregnancy_danger" in red_flags("meri patni garbhwati hai aur uska paani ki thaili phat gayi")
    assert "pregnancy_danger" in red_flags("her water broke and labour pain has started")
    assert "pregnancy_danger" not in red_flags("I am 5 months pregnant and want a check-up")
    assert "pregnancy_danger" not in red_flags("heavy bleeding after an injury")


def test_returns_sorted_unique_tags_and_handles_odd_input():
    tags = red_flags("chest pain and cannot breathe, chest pain again")
    assert tags == sorted(set(tags)) and {"chest_pain_cardiac", "breathing_difficulty"} <= set(tags)
    assert red_flags(None) == [] and red_flags("   ") == [] and red_flags("​‍") == []


@pytest.mark.parametrize("text,want,absent", [
    ("I want to book an appointment for tomorrow", {"book_appointment"}, {"cancel_appointment"}),
    ("Mujhe appointment cancel karna hai", {"cancel_appointment"}, {"book_appointment"}),
    ("मुझे अपॉइंटमेंट रद्द करना है", {"cancel_appointment"}, {"book_appointment"}),
    ("Can I reschedule my appointment to Friday?", {"reschedule"}, {"book_appointment"}),
    ("what time is my appointment", {"appointment_info"}, {"book_appointment", "cancel_appointment"}),
    ("What time does the clinic open on Sunday?", {"clinic_info"}, {"medical_advice"}),
    ("Which tablet should I take for fever?", {"medical_advice", "symptom_report"}, {"clinic_info"}),
    ("kaun si dawai lun bukhar ke liye", {"medical_advice", "symptom_report"}, set()),
    ("क्या दवा लूं बुखार के लिए", {"medical_advice", "symptom_report"}, set()),
    ("I have chest pain", {"symptom_report"}, {"medical_advice"}),
    ("book Dr Rao for my knee pain", {"book_appointment", "symptom_report"}, set()),
    ("I have no fever", set(), {"symptom_report"}),
    ("ok thanks", {"other"}, {"book_appointment"}), ("haan ji", {"other"}, set())])
def test_intents(text, want, absent):
    got = set(intents(text))
    assert want <= got and not (absent & got), (text, got)


def test_analyse_message_bundles_script_tags_and_clean_text():
    m = analyse_message("seene mein dard aur paseena​")
    assert m.script == "hinglish" and "chest_pain_cardiac" in m.red_flags and "symptom_report" in m.intents
    assert m.for_model == "seene mein dard aur paseena"
    assert analyse_message("").intents == ["other"] and analyse_message("").script == "unknown"
