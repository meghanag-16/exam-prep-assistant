"""Key terms of a piece of course text, used by the anti-generic check (an answer must
use the words of the passages it cites) and by the notes check (the notes for a page
must use that page's words).

A term's weight is how often it appears in the text times how rare it is across the
subject's pages (IDF), so words that appear on almost every page (the course name, "slide")
never count as key terms. Tokens are the same as keyword search uses (index.tokens):
lower case, stop words dropped, a plural "s" removed.
"""

from __future__ import annotations

import math
import sqlite3
from collections import Counter

from examprep.search import chunks as chunking
from examprep.search import index


class TermStats:
    """Document frequencies over a subject's content pages."""

    def __init__(self, con: sqlite3.Connection | None = None, texts: list[str] | None = None):
        if texts is None:
            pages = chunking.load_pages(con) if con is not None else []
            texts = [p.text for p in pages if not p.skip]
        self.n = max(len(texts), 1)
        self.df: Counter[str] = Counter()
        for t in texts:
            self.df.update(set(index.tokens(t)))

    def idf(self, term: str) -> float:
        return math.log(1 + (self.n - self.df.get(term, 0) + 0.5) / (self.df.get(term, 0) + 0.5))

    def key_terms(self, text: str, top_n: int = 10) -> list[str]:
        """The top_n most telling terms of text. Numbers, very short tokens and terms found on
        more than half of the pages are left out."""
        counts = Counter(
            t
            for t in index.tokens(text)
            if len(t) > 2 and not t.isdigit() and (self.n < 4 or self.df.get(t, 0) <= self.n / 2)
        )
        ranked = sorted(counts, key=lambda t: (-(1 + math.log(counts[t])) * self.idf(t), t))
        return ranked[:top_n]

    def details(self, text: str, top_n: int = 8) -> list[str]:
        """Specific details of text that notes must keep: numbers of two or more digits and
        words found on at most two of the subject's pages (names, rare terms). The rarest come
        first (numbers before words when equally rare), so a page full of common numbers (a
        timing table) still has its unusual values checked."""
        seen: dict[str, int] = {}
        for t in index.tokens(text):
            numeric = t.isdigit() and len(t) >= 2
            rare = t.isalpha() and len(t) > 3 and self.df.get(t, 0) <= 2
            if (numeric or rare) and t not in seen:
                seen[t] = len(seen)
        ranked = sorted(seen, key=lambda t: (self.df.get(t, 0), not t.isdigit(), seen[t]))
        return ranked[:top_n]


def present(terms: list[str], text: str) -> list[str]:
    """The terms that appear in text (compared as tokens, so "Deadlocks" matches "deadlock")."""
    have = set(index.tokens(text))
    return [t for t in terms if t in have]


def share_present(terms: list[str], text: str) -> float:
    return len(present(terms, text)) / len(terms) if terms else 1.0
