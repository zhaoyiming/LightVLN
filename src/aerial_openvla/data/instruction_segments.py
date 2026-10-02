"""Deterministic semantic-clause segmentation for navigation instructions."""

from __future__ import annotations

import re
from dataclasses import dataclass


NAVIGATION_PREDICATES = (
    "take off", "ascend", "descend", "fly", "move", "go", "head", "turn",
    "rotate", "continue", "approach", "pass", "cross", "follow", "reach",
    "land", "stop", "proceed", "travel", "veer", "circle", "hover",
    "ascending", "descending", "moving", "going", "flying", "landing",
    "lift", "rise", "drop", "fall", "set down", "face",
    "look", "lower", "climb", "float", "drift", "advance", "back up",
    "strafe", "tilt", "keep",
    "dive", "taking off", "pivot", "elevate", "take", "make", "levitate",
    "lay", "turning", "turns", "slow", "bear", "place", "navigate",
    "rotating", "zoom", "zooming", "decrease", "increase", "get down",
    "get up", "stay", "stopping", "passing", "pan", "visit", "focus",
    "park", "avoid", "curve", "flies", "moved",
)
_PREDICATE_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(item) for item in NAVIGATION_PREDICATES) + r")\b",
    re.IGNORECASE,
)
_STRONG_RE = re.compile(
    r"(?:[.;]+\s*|\b(?:and\s+then|then|next|after\s+that|afterwards|subsequently|finally)\b\s*[,;:]?\s*)",
    re.IGNORECASE,
)
_CONDITIONAL_RE = re.compile(
    # Subordinate connectors (until/when/before/after/once) stay attached to
    # their governing action. Treating them as boundaries can mistake nouns
    # such as "park" for navigation predicates and create dangling clauses.
    r",\s*", re.IGNORECASE
)
_CONTENT_RE = re.compile(r"[A-Za-z0-9]+")


@dataclass(frozen=True)
class InstructionSegment:
    text: str
    char_start: int
    char_end: int
    boundary_strength: int = 4

    def as_dict(self) -> dict[str, object]:
        return {
            "text": self.text,
            "char_start": self.char_start,
            "char_end": self.char_end,
        }


def _has_predicate(text: str) -> bool:
    return bool(_PREDICATE_RE.search(text))


def _trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _candidate_boundaries(text: str) -> list[tuple[int, int]]:
    """Return (position, confidence); position starts the next segment."""
    candidates: dict[int, int] = {}
    for match in _STRONG_RE.finditer(text):
        # Keep punctuation with the prior clause; keep sequencing words with
        # the following clause so fragments never dangle on "and then".
        position = match.end() if re.search(r"[.;]", match.group()) else match.start()
        # Sentence/semicolon and explicit sequencing are the strongest cuts.
        strength = 4 if re.search(r"[.;]", match.group()) else 3
        candidates[position] = max(candidates.get(position, 0), strength)
    for match in _CONDITIONAL_RE.finditer(text):
        suffix = text[match.end():]
        if re.match(
            r"\s*(?:and|until|when|before|after|once)\b", suffix, re.IGNORECASE
        ):
            # Keep coordinated predicates and subordinate clauses with the
            # governing action instead of creating connector-led fragments.
            continue
        predicate = _PREDICATE_RE.search(suffix)
        prefix_words = _CONTENT_RE.findall(suffix[:predicate.start()]) if predicate else []
        if predicate is not None and len(prefix_words) <= 3:
            position = match.end() if "," in match.group() else match.start()
            # Plain "and" and comma are deliberately the weakest boundaries.
            token = match.group().strip(" ,").lower()
            strength = 1 if token in {"", ",", "and"} else 2
            candidates[position] = max(candidates.get(position, 0), strength)
    return sorted(candidates.items())


def _merge_pair(segments: list[InstructionSegment], left: int) -> None:
    first, second = segments[left], segments[left + 1]
    segments[left:left + 2] = [InstructionSegment(
        text="", char_start=first.char_start, char_end=second.char_end,
        boundary_strength=second.boundary_strength,
    )]


def _merge_weak_fragment(
    text: str, segments: list[InstructionSegment], index: int
) -> int:
    """Merge a modifier fragment across the weaker adjacent boundary."""
    if index == 0:
        left = 0
    elif index == len(segments) - 1:
        left = index - 1
    else:
        left_strength = segments[index - 1].boundary_strength
        right_strength = segments[index].boundary_strength
        # Prefer the preceding intent for ties (e.g. "on its left side").
        left = index - 1 if left_strength <= right_strength else index
    _merge_pair(segments, left)
    return max(left - 1, 0)


def _materialize(text: str, spans: list[InstructionSegment]) -> list[InstructionSegment]:
    return [InstructionSegment(
        text=text[item.char_start:item.char_end],
        char_start=item.char_start,
        char_end=item.char_end,
        boundary_strength=item.boundary_strength,
    ) for item in spans]


def segment_instruction(instruction: str, max_segments: int = 10) -> list[InstructionSegment]:
    """Split an instruction into 1..max_segments ordered, exhaustive spans."""
    text = instruction.strip()
    if not text:
        raise ValueError("Instruction must not be empty")
    if max_segments < 1:
        raise ValueError("max_segments must be positive")

    boundaries = _candidate_boundaries(text)
    starts = [0, *[position for position, _ in boundaries]]
    ends = [*[position for position, _ in boundaries], len(text)]
    strengths = [*[strength for _, strength in boundaries], 4]
    spans: list[InstructionSegment] = []
    for start, end, strength in zip(starts, ends, strengths):
        start, end = _trim_span(text, start, end)
        if start < end:
            spans.append(InstructionSegment("", start, end, strength))
    if not spans:
        spans = [InstructionSegment("", 0, len(text), 4)]

    # Merge fragments that cannot stand as local navigation intents. Prefer
    # attaching modifiers/STOP-only fragments to their predecessor.
    index = 0
    while len(spans) > 1 and index < len(spans):
        fragment = text[spans[index].char_start:spans[index].char_end]
        content = _CONTENT_RE.findall(fragment)
        normalized = fragment.strip(" ,.;:").lower()
        has_predicate = _has_predicate(fragment)
        starts_with_and = bool(re.match(r"^and\b", normalized, re.IGNORECASE))
        weak = (
            not has_predicate
            or normalized in {"and", "then", "next", "finally", "stop"}
            or (len(content) < 3 and not has_predicate)
        )
        if starts_with_and and index > 0:
            _merge_pair(spans, index - 1)
            index = max(index - 2, 0)
        elif weak:
            index = _merge_weak_fragment(text, spans, index)
        else:
            index += 1

    # Remove the weakest remaining boundaries until K <= Kmax. The stored
    # strength belongs to the boundary after each segment.
    # Reduce only when K actually exceeds Kmax. Always remove the weakest
    # remaining boundary first: comma/subordinate < then < sentence/semicolon.
    # Therefore a strong boundary is crossed only if strong clauses alone
    # exceed the fixed model budget.
    while len(spans) > max_segments:
        left = min(range(len(spans) - 1), key=lambda idx: spans[idx].boundary_strength)
        _merge_pair(spans, left)

    spans = _materialize(text, spans)
    validate_segments(text, spans, max_segments=max_segments)
    return spans


def validate_segments(
    instruction: str,
    segments: list[InstructionSegment],
    *,
    max_segments: int = 10,
) -> None:
    text = instruction.strip()
    if not 1 <= len(segments) <= max_segments:
        raise ValueError(f"Expected 1..{max_segments} segments, got {len(segments)}")
    previous_end = 0
    for index, segment in enumerate(segments):
        if not 0 <= segment.char_start < segment.char_end <= len(text):
            raise ValueError(f"Invalid segment span at index {index}")
        if segment.char_start < previous_end:
            raise ValueError("Segment spans overlap or are out of order")
        # Only whitespace may be left between trimmed segment spans.
        if text[previous_end:segment.char_start].strip():
            raise ValueError("Segment spans leave uncovered instruction content")
        if segment.text != text[segment.char_start:segment.char_end]:
            raise ValueError("Segment text does not match its character span")
        previous_end = segment.char_end
    if text[previous_end:].strip():
        raise ValueError("Segment spans do not cover the instruction tail")


def segments_as_metadata(instruction: str, max_segments: int = 10) -> dict[str, object]:
    segments = segment_instruction(instruction, max_segments=max_segments)
    return {
        "instruction": instruction,
        "segments": [segment.as_dict() for segment in segments],
        "num_segments": len(segments),
    }
