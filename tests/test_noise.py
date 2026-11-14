import random

from swasthya.noise import NEGATION, VARIANTS, noisify

TEXT = "meri beti ko bahut tez bukhar hai aur saans nahi aa rahi, 2 din se khoon bhi aa raha hai"


def run(text=TEXT, seed=1, **kw):
    return noisify(text, random.Random(seed), **kw)


def test_deterministic_for_a_seed_and_varies_across_seeds():
    assert run(seed=5) == run(seed=5)
    assert len({run(seed=s) for s in range(20)}) > 5


def test_zero_probabilities_are_the_identity():
    assert run(p_word=0.0, p_join=0.0) == TEXT


def test_full_noise_changes_the_text_but_keeps_digits_and_word_count_without_joins():
    out = run(p_word=1.0, p_join=0.0)
    assert out != TEXT and "2" in out.split()
    assert len(out.split()) == len(TEXT.split())


def test_digits_and_non_latin_text_are_never_touched():
    deva = "मेरी बेटी को बहुत तेज़ बुखार है 102"
    assert run(deva, p_word=1.0, p_join=1.0) == deva
    mixed = noisify("bukhar 102 degree hai", random.Random(0), p_word=1.0, p_join=0.0)
    assert "102" in mixed.split()


def test_negation_words_only_take_known_spellings():
    for seed in range(60):
        w = noisify("nahi", random.Random(seed), p_word=1.0, p_join=0.0)
        assert w in {"nahi"} | set(VARIANTS["nahi"]) and w != "kuchh"
    for neg in ("mat", "na", "bina", "koi"):
        assert all(noisify(neg, random.Random(s), p_word=1.0, p_join=0.0) == neg for s in range(30))
    assert {"nahi", "mat", "bina"} <= NEGATION


def test_at_most_one_change_per_word_and_output_stays_latin_text():
    for seed in range(40):
        out = noisify("zyada bahut chhoti udaas kharab", random.Random(seed), p_word=1.0, p_join=0.0)
        assert out.isascii() and out.replace(" ", "").isalpha()
        for before, after in zip("zyada bahut chhoti udaas kharab".split(), out.split()):
            assert abs(len(before) - len(after)) <= 1


def test_clinically_important_words_survive_in_a_known_form():
    seen = {noisify("saans", random.Random(s), p_word=1.0, p_join=0.0) for s in range(50)}
    assert seen <= {"saans"} | set(VARIANTS["saans"])


def test_variants_never_collide_with_a_different_word():
    flat = {v for vs in VARIANTS.values() for v in vs}
    assert "saas" not in flat and "main" not in flat                       # saas = mother-in-law, main = I
    for seed in range(100):                                                # English words are not rewritten with w/v
        assert "responsiwe" not in noisify("responsive", random.Random(seed), p_word=1.0, p_join=0.0)


def test_empty_and_punctuation_only_text():
    assert noisify("", random.Random(0)) == ""
    assert noisify("?!", random.Random(0), p_word=1.0) == "?!"
