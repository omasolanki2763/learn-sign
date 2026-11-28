"""The 20 signs used by the Task 8 verifier spike, chosen to include minimal pairs.

A minimal pair is two signs that differ in one part only. They are the hard case for a
verifier: a random wrong sign is easy to reject, MOM vs DAD (same handshape, chin vs forehead)
is not. Glosses use the GISLR / PopSign spelling. `validate_glosses` checks them against the
competition's 250-gloss list at run time.

Part labels below are the Planner's reading of the standard citation forms; the notebook
cross-checks each pair against ASL-LEX 2.0 phonology before results are reported.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Final

SIGNS_JSON_PATH: Final[Path] = Path(__file__).parent / "step0_signs.json"

# The complete 250-gloss vocabulary of the GISLR competition.
GISLR_250_GLOSSES: Final[tuple[str, ...]] = (
    "tv", "after", "airplane", "all", "alligator", "animal", "another", "any", "apple", "arm",
    "aunt", "awake", "backyard", "bad", "balloon", "bath", "because", "bed", "bedroom", "bee",
    "before", "beside", "better", "bird", "black", "blow", "blue", "boat", "book", "boy",
    "brother", "brown", "bug", "bye", "callonphone", "can", "car", "carrot", "cat", "cereal",
    "chair", "cheek", "child", "chin", "chocolate", "clean", "close", "closet", "cloud", "clown",
    "cow", "cowboy", "cry", "cut", "cute", "dad", "dance", "dirty", "dog", "doll", "donkey",
    "down", "drawer", "drink", "drop", "dry", "dryer", "duck", "ear", "elephant", "empty",
    "every", "eye", "face", "fall", "farm", "fast", "feet", "find", "fine", "finger", "finish",
    "fireman", "first", "fish", "flag", "flower", "food", "for", "frenchfries", "frog", "garbage",
    "gift", "giraffe", "girl", "give", "glasswindow", "go", "goose", "grandma", "grandpa",
    "grass", "green", "gum", "hair", "happy", "hat", "hate", "have", "haveto", "head", "hear",
    "helicopter", "hello", "hen", "hesheit", "hide", "high", "home", "horse", "hot", "hungry",
    "icecream", "if", "into", "jacket", "jeans", "jump", "kiss", "kitty", "lamp", "later", "like",
    "lion", "lips", "listen", "look", "loud", "mad", "make", "man", "many", "milk", "minemy",
    "mitten", "mom", "moon", "morning", "mouse", "mouth", "nap", "napkin", "night", "no", "noisy",
    "nose", "not", "now", "nuts", "old", "on", "open", "orange", "outside", "owie", "owl",
    "pajamas", "pen", "pencil", "penny", "person", "pig", "pizza", "please", "police", "pool",
    "potty", "pretend", "pretty", "puppy", "puzzle", "quiet", "radio", "rain", "read", "red",
    "refrigerator", "ride", "room", "sad", "same", "say", "scissors", "see", "shhh", "shirt",
    "shoe", "shower", "sick", "sleep", "sleepy", "smile", "snack", "snow", "stairs", "stay",
    "sticky", "store", "story", "stuck", "sun", "table", "talk", "taste", "thankyou", "that",
    "there", "think", "thirsty", "tiger", "time", "tomorrow", "tongue", "tooth", "toothbrush",
    "touch", "toy", "tree", "uncle", "underwear", "up", "vacuum", "wait", "wake", "water",
    "wet", "weus", "where", "white", "who", "why", "will", "wolf", "yellow", "yes", "yesterday",
    "yourself", "yucky", "zebra", "zipper",
)


def _load_embedded_or_json() -> dict[str, Any]:
    if SIGNS_JSON_PATH.exists():
        return json.loads(SIGNS_JSON_PATH.read_text(encoding="utf-8"))
    return {
        "signs": [
            "eye", "nose", "ear", "mouth", "tooth", "mom", "dad", "think", "hat", "sleep",
            "book", "table", "car", "shoe", "yes", "no", "milk", "water", "dog", "cat",
        ],
        "minimal_pairs": [
            {"signs": ["mom", "dad"], "differs_in": "location"},
            {"signs": ["eye", "nose"], "differs_in": "location"},
            {"signs": ["nose", "mouth"], "differs_in": "location"},
            {"signs": ["mouth", "tooth"], "differs_in": "location"},
            {"signs": ["yes", "no"], "differs_in": "handshape"},
        ],
        "two_handed": ["book", "table", "car", "shoe"],
    }


STEP0_SIGNS: dict[str, Any] = _load_embedded_or_json()


def load_step0_signs() -> dict[str, Any]:
    """Return the Task 8 sign list (a copy, so callers cannot change the module constant)."""
    return {
        "signs": list(STEP0_SIGNS["signs"]),
        "minimal_pairs": [dict(p) for p in STEP0_SIGNS["minimal_pairs"]],
        "two_handed": list(STEP0_SIGNS["two_handed"]),
    }


def validate_glosses(known_glosses: Iterable[str]) -> list[str]:
    """Return the Task 8 signs missing from `known_glosses` (empty list = all present)."""
    known = {g.lower() for g in known_glosses}
    return [g for g in STEP0_SIGNS["signs"] if g.lower() not in known]
