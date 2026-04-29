import json
import pytest
from pathlib import Path

_PHRASES_PATH = Path("data/training_phrases.json")
_skip_if_missing = pytest.mark.skipif(
    not _PHRASES_PATH.exists(),
    reason="data/training_phrases.json not present (untracked); generate via scripts/generate_phrases.py",
)


def _load():
    with open(_PHRASES_PATH) as f:
        return json.load(f)


@_skip_if_missing
def test_dataset_has_required_keys():
    data = _load()
    assert "training" in data
    assert "held_out" in data


@_skip_if_missing
def test_dataset_sizes():
    data = _load()
    assert len(data["training"]) >= 30, "Need at least 30 training phrases"
    assert len(data["held_out"]) >= 5, "Need at least 5 held-out phrases"


@_skip_if_missing
def test_no_overlap_between_splits():
    data = _load()
    overlap = set(data["training"]) & set(data["held_out"])
    assert not overlap, f"Phrases in both splits: {overlap}"


@_skip_if_missing
def test_excluded_phrases_absent():
    data = _load()
    excluded = {"pizza", "birthday party", "basketball", "cooking dinner", "feeling nostalgic"}
    found = excluded & set(data["training"])
    assert not found, f"Excluded phrases in training set: {found}"


@_skip_if_missing
def test_all_phrases_nonempty_strings():
    data = _load()
    for phrase in data["training"] + data["held_out"]:
        assert isinstance(phrase, str) and phrase.strip(), f"Bad phrase: {phrase!r}"


def test_early_stop_triggers_on_plateau():
    best = float("-inf")
    no_improve = 0
    should_stop = False
    rewards = [0.30, 0.35, 0.36, 0.36, 0.36]
    for r in rewards:
        if r > best + 0.02:
            best = r
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= 3:
            should_stop = True
            break
    assert should_stop, "Expected early stop after 3 non-improving evals"
    assert no_improve == 3


def test_early_stop_does_not_trigger_on_improvement():
    best = float("-inf")
    no_improve = 0
    should_stop = False
    rewards = [0.20, 0.25, 0.30, 0.35, 0.40]
    for r in rewards:
        if r > best + 0.02:
            best = r
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= 3:
            should_stop = True
            break
    assert not should_stop
    assert no_improve == 0
