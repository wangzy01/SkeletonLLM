"""Prompt templates for the SkeletonLLM training tasks.

Single source of truth for the natural-language prompts used by MQA (Stages 1 &
4), Disc-FT (Stage 2), and CR-Distill (Stage 3). The wording of Disc-FT and
CR-Distill matches the templates in the paper appendix (Fig. "Prompt templates
for MQA, Disc-FT, and CR-Distill"). The MQA prompt is the label-only variant
that the released DrAction candidate weights were produced with; see the README
for how it relates to the paper's reasoning-style MQA prompt.

The frame placeholder is ``<image>`` (one per rendered frame). At training time
the InternVL conversation template supplies ``SYSTEM_PROMPT`` separately, so it
is not embedded in the human turn; it is, however, sent as the system message
when querying the teacher model for CR-Distill.
"""
from __future__ import annotations

from typing import Sequence

SYSTEM_PROMPT = "You are an expert in video categorization and human action recognition."

# Shared description of the DrAction pseudo-image modality.
PSEUDO_IMAGE_DESC = (
    "The clip you see is rendered from 3D skeleton data into abstract pseudo-images. "
    "Each frame encodes joint positions, bone connections, depth, and motion cues, "
    "and may look different from natural RGB videos. Treat these frames as a faithful "
    "visualization of human body kinematics."
)


def frame_block(num_frames: int) -> str:
    """``Frame1: <image>\\n ... FrameN: <image>\\n`` — one placeholder per frame."""
    return "".join(f"Frame{i + 1}: <image>\n" for i in range(num_frames))


def build_mqa_question(labels: Sequence[str], num_frames: int, simplified: bool = True) -> str:
    """Multiple-choice question over candidate ``labels``.

    ``simplified=True`` (default) is the label-only prompt used to produce the
    released weights. ``simplified=False`` is the paper's reasoning-style prompt
    (think, then output a final ``Label: <action>`` line).
    """
    labels_block = "\n".join(labels)
    if simplified:
        body = (
            "This clip is from the NTU RGB+D dataset for human action recognition. "
            "It is rendered from skeleton data. "
            "Choose exactly one action label from the list below and answer with the label only:\n"
            + labels_block
        )
    else:
        body = (
            PSEUDO_IMAGE_DESC
            + " You need to identify the action in the skeleton video and select one from the "
            "action labels below as the classification result. You are required to output your "
            "thought process, and finally, provide the specific classification label on a "
            "separate line in the format 'Label: <action>'.\n"
            + labels_block
        )
    return frame_block(num_frames) + body


def build_discft_question(action_label: str, num_frames: int) -> str:
    """Binary judgment prompt for Disc-FT (answer is a single ``YES``/``NO`` token)."""
    return (
        frame_block(num_frames)
        + PSEUDO_IMAGE_DESC
        + f"\nIs the action in this video clip '{action_label}'? "
        + 'Answer with ONLY the word "YES" or "NO".'
    )


def build_cr_student_question(num_frames: int) -> str:
    """Student prompt for CR-Distill (no ground-truth label given to the student)."""
    return (
        frame_block(num_frames)
        + PSEUDO_IMAGE_DESC
        + "\nPlease describe what the person does in the skeleton video, infer the likely "
        "activity, and end with exactly one classification label on a separate line "
        "formatted as 'Label: <action>'."
    )


def build_cr_teacher_prompt(action_label: str, num_frames: int, dataset_name: str = "NTU RGB+D") -> str:
    """Label-conditioned teacher prompt for CR-Distill rationale generation.

    Sent to a strong MLLM teacher (e.g. GPT-4o) together with ``SYSTEM_PROMPT``
    and the rendered frames. The teacher is given the ground-truth label and
    asked for a summary, a body-part-grounded causal chain, and a final label.
    """
    return (
        frame_block(num_frames)
        + f"This clip is from the {dataset_name} dataset for human action recognition. "
        "It is rendered from 3D skeleton data into abstract motion-aware images.\n"
        f"Ground-truth action label: {action_label}\n"
        "Please:\n"
        "1) Briefly summarize what the person is doing in this skeleton video.\n"
        "2) Provide a step-by-step causal chain that explains the action over time, "
        "explicitly referring to body parts and their motion (e.g., which joints move "
        "first, how limbs follow, how the pose evolves).\n"
        "3) Conclude with the final classification label.\n"
        "End your response with exactly one line:\n"
        f"Label: {action_label}"
    )
