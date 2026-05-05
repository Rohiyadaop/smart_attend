"""
Shared college configuration and branch helpers for the B.Tech workflow.
"""

from __future__ import annotations

import re

PROGRAM_NAME = "B.Tech"
YEAR_OPTIONS = [1, 2, 3, 4]
BTECH_BRANCHES = [
    {"code": "CSE", "name": "Computer Science and Engineering"},
    {"code": "ME", "name": "Mechanical Engineering"},
    {"code": "CE", "name": "Civil Engineering"},
    {"code": "EE", "name": "Electrical Engineering"},
    {"code": "ECE", "name": "Electronics and Communication Engineering"},
]
BRANCH_CODES = [item["code"] for item in BTECH_BRANCHES]
BRANCH_NAME_MAP = {item["code"]: item["name"] for item in BTECH_BRANCHES}

_BRANCH_ALIASES = {
    "cse": "CSE",
    "computer science": "CSE",
    "computer science engineering": "CSE",
    "computer science and engineering": "CSE",
    "cs": "CSE",
    "me": "ME",
    "mechanical": "ME",
    "mechanical engineering": "ME",
    "ce": "CE",
    "civil": "CE",
    "civil engineering": "CE",
    "ee": "EE",
    "electrical": "EE",
    "electrical engineering": "EE",
    "ece": "ECE",
    "electronics": "ECE",
    "electronics communication": "ECE",
    "electronics and communication": "ECE",
    "electronics and communication engineering": "ECE",
    "electronics communication engineering": "ECE",
}


def _clean_branch(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (value or "").strip().lower()).strip()


def normalize_branch(value: str) -> str:
    if not value:
        return ""
    cleaned = _clean_branch(value)
    if cleaned in _BRANCH_ALIASES:
        return _BRANCH_ALIASES[cleaned]
    return value.strip().upper()


def is_valid_branch(value: str) -> bool:
    return normalize_branch(value) in BRANCH_CODES


def get_branch_name(value: str) -> str:
    code = normalize_branch(value)
    return BRANCH_NAME_MAP.get(code, code)


def format_batch_label(branch: str, year: int, section: str = "") -> str:
    code = normalize_branch(branch) or "NA"
    label = f"{code} Year {year}"
    if section:
        label = f"{label} {section.strip().upper()}"
    return label


def format_session_label(program: str, branch: str, year: int, room: str = "",
                         subject: str = "", section: str = "") -> str:
    parts = [program or PROGRAM_NAME, format_batch_label(branch, year, section)]
    if room:
        parts.append(f"Room {room.strip().upper()}")
    if subject:
        parts.append(subject.strip())
    return " - ".join(parts)
