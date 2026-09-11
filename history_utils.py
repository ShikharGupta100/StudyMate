"""
history_utils.py
Simple JSON-file-backed storage for per-book chat history, so past
questions/answers survive a Streamlit restart.
"""

import json
import os
from datetime import datetime

from rag_utils import BASE_DIR

HISTORY_PATH = os.path.join(BASE_DIR, "history.json")


def _ensure_file():
    os.makedirs(BASE_DIR, exist_ok=True)
    if not os.path.isfile(HISTORY_PATH):
        with open(HISTORY_PATH, "w") as f:
            json.dump({}, f)


def load_history() -> dict:
    """
    Returns: { book_id: {"name": str, "uploaded_at": str, "chats": [ {q, a, time} ] } }
    """
    _ensure_file()
    with open(HISTORY_PATH, "r") as f:
        return json.load(f)


def save_history(history: dict):
    _ensure_file()
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)


def register_book(book_id: str, display_name: str):
    history = load_history()
    if book_id not in history:
        history[book_id] = {
            "name": display_name,
            "uploaded_at": datetime.now().isoformat(timespec="seconds"),
            "chats": [],
        }
        save_history(history)
    return history


def add_chat(book_id: str, question: str, answer: str):
    history = load_history()
    if book_id not in history:
        history[book_id] = {"name": book_id, "uploaded_at": "", "chats": []}
    history[book_id]["chats"].append(
        {
            "q": question,
            "a": answer,
            "time": datetime.now().isoformat(timespec="seconds"),
        }
    )
    save_history(history)


def clear_book_chats(book_id: str):
    history = load_history()
    if book_id in history:
        history[book_id]["chats"] = []
        save_history(history)


def remove_book(book_id: str):
    history = load_history()
    if book_id in history:
        del history[book_id]
        save_history(history)