"""LoCoMo dataset loader.

Downloads and parses the LoCoMo-10 dataset from snap-research/locomo.
Provides structured access to conversations, sessions, turns, and QA pairs.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmarks.locomo.config import CATEGORY_MAP, DATASET_URL

logger = logging.getLogger(__name__)


@dataclass
class Turn:
    """A single dialogue turn in a conversation session."""

    speaker: str
    text: str
    dia_id: str
    img_url: list[str] | None = None
    blip_caption: str | None = None
    query: str | None = None

    def format_for_memory(self) -> str:
        """Format this turn for ingestion into a memory system.

        Returns "{Speaker}: {text}" with image context appended if present.
        """
        parts = [f"{self.speaker}: {self.text}"]
        if self.blip_caption:
            img_context = f"[Image: {self.blip_caption}"
            if self.query:
                img_context += f" (search: {self.query})"
            img_context += "]"
            parts.append(img_context)
        return " ".join(parts)


@dataclass
class Session:
    """A conversation session (one continuous chat between speakers)."""

    index: int  # 1-based session number
    date_time: str  # e.g., "1:56 pm on 8 May, 2023"
    turns: list[Turn]


@dataclass
class Question:
    """A QA evaluation question."""

    question: str
    answer: str
    category: int  # 1-5
    evidence: list[str]
    adversarial_answer: str | None = None

    @property
    def category_name(self) -> str:
        return CATEGORY_MAP.get(self.category, "unknown")

    @property
    def is_adversarial(self) -> bool:
        return self.category == 5


@dataclass
class Conversation:
    """A complete LoCoMo conversation with sessions and QA pairs."""

    index: int  # 0-based index in the dataset
    speaker_a: str
    speaker_b: str
    sessions: list[Session]
    questions: list[Question]

    @property
    def total_turns(self) -> int:
        return sum(len(s.turns) for s in self.sessions)

    def get_questions(self, categories: list[int] | None = None) -> list[Question]:
        """Get questions filtered by category."""
        if categories is None:
            return self.questions
        return [q for q in self.questions if q.category in categories]


@dataclass
class LoCoMoDataset:
    """The complete LoCoMo-10 dataset."""

    conversations: list[Conversation]

    def get_conversations(self, indices: list[int] | None = None) -> list[Conversation]:
        """Get conversations by index (0-based). None = all."""
        if indices is None:
            return self.conversations
        return [self.conversations[i] for i in indices]

    def count_questions(self, categories: list[int] | None = None) -> int:
        """Count total questions across all conversations."""
        return sum(len(c.get_questions(categories)) for c in self.conversations)

    def summary(self, categories: list[int] | None = None) -> str:
        """Return a human-readable summary of the dataset."""
        lines = [
            f"LoCoMo Dataset: {len(self.conversations)} conversations",
        ]
        total_turns = 0
        total_questions = 0
        cat_counts: dict[str, int] = {}
        for conv in self.conversations:
            total_turns += conv.total_turns
            for q in conv.get_questions(categories):
                total_questions += 1
                name = q.category_name
                cat_counts[name] = cat_counts.get(name, 0) + 1

        lines.append(f"Total turns: {total_turns}")
        lines.append(f"Total questions: {total_questions}")
        if cat_counts:
            lines.append("Questions by category:")
            for name, count in sorted(cat_counts.items()):
                lines.append(f"  {name}: {count}")
        return "\n".join(lines)


def download_dataset(data_dir: Path, url: str = DATASET_URL) -> Path:
    """Download the LoCoMo dataset if not already present.

    Returns the path to the downloaded file.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    dest = data_dir / "locomo10.json"

    if dest.exists():
        logger.info("Dataset already exists: %s", dest)
        return dest

    logger.info("Downloading LoCoMo dataset from %s ...", url)
    urllib.request.urlretrieve(url, dest)
    logger.info("Saved to %s", dest)
    return dest


def _parse_sessions(conversation: dict[str, Any]) -> list[Session]:
    """Extract sessions from a conversation dict.

    Sessions are stored as session_1, session_2, ... with corresponding
    session_1_date_time, session_2_date_time, ... keys.
    """
    sessions = []
    # Find all session keys (session_1, session_2, ...)
    session_keys = sorted(
        [k for k in conversation if re.match(r"^session_\d+$", k)],
        key=lambda k: int(k.split("_")[1]),
    )

    for key in session_keys:
        idx = int(key.split("_")[1])
        date_key = f"{key}_date_time"
        date_time = conversation.get(date_key, "")

        turns = []
        for turn_data in conversation[key]:
            if not isinstance(turn_data, dict):
                continue
            text = turn_data.get("text", "")
            if not text:
                continue
            turns.append(
                Turn(
                    speaker=turn_data.get("speaker", "Unknown"),
                    text=text,
                    dia_id=turn_data.get("dia_id", ""),
                    img_url=turn_data.get("img_url"),
                    blip_caption=turn_data.get("blip_caption"),
                    query=turn_data.get("query"),
                )
            )

        sessions.append(Session(index=idx, date_time=date_time, turns=turns))

    return sessions


def _parse_questions(qa_list: list[dict[str, Any]]) -> list[Question]:
    """Parse QA items into Question objects."""
    questions = []
    for qa in qa_list:
        # Some answers are integers (e.g., years), normalize to string
        answer = qa.get("answer", "")
        if answer is None:
            answer = ""
        answer = str(answer)

        questions.append(
            Question(
                question=qa.get("question", ""),
                answer=answer,
                category=qa.get("category", 0),
                evidence=qa.get("evidence", []),
                adversarial_answer=qa.get("adversarial_answer"),
            )
        )
    return questions


def load_dataset(dataset_path: Path) -> LoCoMoDataset:
    """Load and parse the LoCoMo dataset from a JSON file.

    Args:
        dataset_path: Path to locomo10.json.

    Returns:
        Parsed LoCoMoDataset with all conversations, sessions, and questions.
    """
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}\n"
            "Run 'python -m benchmarks.locomo download' first."
        )

    logger.info("Loading dataset from %s", dataset_path)
    with open(dataset_path) as f:
        raw_data = json.load(f)

    conversations = []
    for idx, item in enumerate(raw_data):
        conv_data = item.get("conversation", {})
        sessions = _parse_sessions(conv_data)
        questions = _parse_questions(item.get("qa", []))

        conversations.append(
            Conversation(
                index=idx,
                speaker_a=conv_data.get("speaker_a", "Speaker A"),
                speaker_b=conv_data.get("speaker_b", "Speaker B"),
                sessions=sessions,
                questions=questions,
            )
        )

    dataset = LoCoMoDataset(conversations=conversations)
    logger.info("Loaded %s", dataset.summary())
    return dataset
