"""Shared NTU-60 metadata and prompt builders.

This is the single source of truth for the class labels, zero-shot split
presets, cross-subject protocol, and the MQA prompt used by both the annotation
builders (``tools/``) and the evaluation scripts (``eval/``). Keeping it in one
place avoids the three-way drift that existed in the original code dump.

The MQA prompt here is the label-only variant (the model answers with the class
label directly). The released DrAction candidate weights were produced with this
label-only formulation; see the README for how it relates to the paper prompt.
"""
from __future__ import annotations

import re
from typing import Iterable, List, Sequence

# 60 NTU RGB+D action classes, 1-indexed by position (id = index + 1).
NTU60_CLASSES: List[str] = [
    "drink water", "eat food", "brush teeth", "brush hair", "drop something",
    "pick up", "throw", "sit down", "stand up", "clapping", "reading", "writing",
    "tear up paper", "put on jacket", "take off jacket", "put on shoe", "take off shoe",
    "put on glasses", "take off glasses", "put on hat/cap", "take off hat/cap",
    "cheer (raise arms)", "hand waving", "kicking the air/something", "reach into pocket",
    "hopping", "jump up", "phone call", "play with phone/tablet", "type on keyboard",
    "point at something/directions", "take a selfie", "check time (from watch)",
    "rub two hands", "bow (bend forward)", "shake head", "wipe face", "salute",
    "pray with hands together", "cross hands in front", "sneezing/cough", "staggering",
    "falling down", "headache", "chest pain", "back pain", "neck pain",
    "nausea/vomiting", "fan self", "punch/slap someone", "kicking someone",
    "pushing someone", "pat someone on the back", "point at someone", "hug each other",
    "giving object", "touch someone's pocket", "shaking hands",
    "walking towards each other", "walking apart from each other",
]

# NTU-60 zero-shot splits. The number in the split name is the count of
# seen/training classes; the listed ids are the unseen/test classes. The
# 55/5 and 48/12 splits follow SynSE; the 40/20 and 30/30 follow PURLS.
NTU60_55_CS_UNSEEN = [11, 12, 20, 27, 57]
NTU60_48_CS_UNSEEN = [4, 6, 10, 13, 16, 41, 43, 48, 52, 57, 59, 60]
NTU60_40_CS_UNSEEN = [1, 13, 14, 15, 16, 17, 18, 23, 24, 27, 30, 31, 32, 36, 37, 43, 44, 49, 57, 58]
NTU60_30_CS_UNSEEN = [1, 2, 3, 7, 8, 9, 11, 13, 14, 16, 17, 19, 21, 22, 26, 27, 28, 32, 33, 34,
                      40, 43, 46, 48, 49, 52, 53, 56, 59, 60]

NTU60_SPLIT_PRESETS = {
    "ntu60_55_cs": NTU60_55_CS_UNSEEN,
    "ntu60_48_cs": NTU60_48_CS_UNSEEN,
    "ntu60_40_cs": NTU60_40_CS_UNSEEN,
    "ntu60_30_cs": NTU60_30_CS_UNSEEN,
    # Closed-set sanity check: no held-out classes (all 60 are "seen").
    "ntu60_all_cs": [],
}

# Standard NTU-60 cross-subject training subjects. Test samples are subjects
# NOT in this set. Files with setup id >= 18 belong to NTU-120 and are dropped.
NTU60_CS_TRAIN_SUBJECTS = {
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38,
}

SAMPLE_RE = re.compile(
    r"S(?P<setup>\d{3})C(?P<camera>\d{3})P(?P<subject>\d{3})R(?P<repeat>\d{3})A(?P<action>\d{3})"
)


def parse_sample_id(sample_id: str) -> dict | None:
    """Parse an NTU ``S###C###P###R###A###`` stem into its integer fields."""
    match = SAMPLE_RE.match(sample_id)
    if match is None:
        return None
    return {k: int(v) for k, v in match.groupdict().items()}


def parse_int_csv(text: str | None) -> List[int]:
    if not text:
        return []
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def unseen_action_ids(split: str) -> List[int]:
    """Held-out (excluded-from-training) action ids for a split; [] for all_cs."""
    if split not in NTU60_SPLIT_PRESETS:
        raise KeyError(f"unknown split '{split}'; choices: {sorted(NTU60_SPLIT_PRESETS)}")
    return sorted(set(NTU60_SPLIT_PRESETS[split]))


def seen_action_ids(split: str) -> List[int]:
    """Seen/training action ids for a split (all 60 minus the held-out set)."""
    unseen = set(unseen_action_ids(split))
    return [i for i in range(1, 61) if i not in unseen]


def test_action_ids(split: str, action_ids_csv: str | None = None) -> List[int]:
    """Candidate/test action ids used at evaluation time. Explicit CSV overrides.

    For zero-shot splits these are the held-out (unseen) classes; for the
    closed-set ``ntu60_all_cs`` sanity check they are all 60 classes.
    """
    if action_ids_csv:
        return sorted(set(parse_int_csv(action_ids_csv)))
    unseen = unseen_action_ids(split)
    return unseen if unseen else list(range(1, 61))


def is_ntu60_cs_train_file(sample_id: str) -> bool:
    """True if the file is an NTU-60 (setup < 18) cross-subject training sample."""
    ids = parse_sample_id(sample_id)
    if ids is None or ids["setup"] >= 18:
        return False
    return ids["subject"] in NTU60_CS_TRAIN_SUBJECTS


def is_ntu60_cs_test_file(sample_id: str, action_ids: Iterable[int]) -> bool:
    """True if the file is an NTU-60 cross-subject TEST sample for these actions."""
    ids = parse_sample_id(sample_id)
    if ids is None or ids["setup"] >= 18:
        return False
    if ids["subject"] in NTU60_CS_TRAIN_SUBJECTS:
        return False
    return ids["action"] in set(action_ids)


def build_mqa_question(action_ids: Sequence[int], num_frames: int, simplified: bool = True) -> str:
    """Label-only multiple-choice question over the given candidate action ids.

    Thin wrapper over :func:`skeletonllm.data.prompts.build_mqa_question` that
    maps NTU-60 action ids to class label strings. ``simplified=True`` (default)
    is the label-only prompt used to produce the released weights.
    """
    from skeletonllm.data.prompts import build_mqa_question as _build

    labels = [NTU60_CLASSES[i - 1] for i in action_ids]
    return _build(labels, num_frames, simplified=simplified)
