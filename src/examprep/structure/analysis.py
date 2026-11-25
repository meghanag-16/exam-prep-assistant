"""Page-structure analysis: headings, near-duplicate pages, deck boundaries in
combined PDFs, and contiguous topic segments. Pure functions over page data,
so they are easy to test."""

from __future__ import annotations

import re
import statistics
from collections import Counter

import numpy as np

from examprep import config
from examprep.structure.store import Page

NUM_LINE = re.compile(r"^(?:slide\s*|page\s*|p\.\s*)?(\d{1,3})(?:\s*(?:/|of)\s*\d{1,3})?$", re.I)


# ------------------------------------------------------------ headings


def _sizes(page: Page) -> list[float]:
    return [float(x[1]) for x in page.layout if len(x) > 1 and x[1]]


def heading(page: Page, file_median: float | None) -> str | None:
    """Slide title for PPTX/TXT; for PDFs the line with the largest font if it
    clearly stands out from body text."""
    if page.kind != "pdf":
        return (page.title or "").strip() or None
    sizes = _sizes(page)
    if not sizes or not file_median:
        return None
    best = max(page.layout, key=lambda x: float(x[1] or 0))
    text = str(best[0]).strip()
    if float(best[1]) >= 1.2 * file_median and 2 < len(text) <= 120 and not NUM_LINE.match(text):
        return text
    return None


def file_medians(pages: list[Page]) -> dict[int, float]:
    by_file: dict[int, list[float]] = {}
    for p in pages:
        by_file.setdefault(p.file_id, []).extend(_sizes(p))
    return {f: statistics.median(s) for f, s in by_file.items() if s}


# ------------------------------------------------------------ duplicates


def _shingles(text: str, n: int) -> set[str]:
    words = re.findall(r"\w+", text.lower())
    if len(words) < n:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def near_duplicates(pages: list[Page]) -> dict[int, int]:
    """Map page id -> id of the page it duplicates. The kept page is the one
    with more text (so "same slide plus one bullet" keeps the fuller slide),
    or the earlier one when equal. Duplicates stay in the ledger for coverage."""
    cfg = config.structure()["duplicates"]
    n, min_words = int(cfg["shingle_words"]), int(cfg["min_words"])
    sim_t, cont_t = float(cfg["similarity"]), float(cfg["containment"])
    sh = []
    for p in pages:
        words = len(re.findall(r"\w+", p.text))
        sh.append(_shingles(p.text, n) if words >= min_words else set())
    dup: dict[int, int] = {}
    # Index shingles so only pages sharing text are compared.
    index: dict[str, list[int]] = {}
    for i, s in enumerate(sh):
        for g in s:
            index.setdefault(g, []).append(i)
    # Ignore word sequences found on very many pages (footers, college name):
    # they say nothing about whether two slides are the same.
    too_common = max(20, len(pages) // 10)
    boiler = {g for g, ids in index.items() if len(ids) > too_common}
    index = {g: ids for g, ids in index.items() if len(ids) <= too_common}
    sh = [x - boiler for x in sh]
    for i, s in enumerate(sh):
        if not s or pages[i].id in dup:
            continue
        shared = Counter(j for g in s for j in index.get(g, ()) if j != i)
        for j, common in shared.items():
            if pages[j].id in dup or pages[i].id in dup:
                continue
            union = len(s) + len(sh[j]) - common
            jacc = common / union if union else 0.0
            small, big = (i, j) if len(s) <= len(sh[j]) else (j, i)
            contain = common / len(sh[small]) if sh[small] else 0.0
            if jacc >= sim_t or contain >= cont_t:
                if len(sh[small]) == len(sh[big]):
                    small, big = (max(i, j), min(i, j))  # keep the earlier page
                dup[pages[small].id] = pages[big].id
    return dup


# ------------------------------------------------------------ deck boundaries


def _printed_number(page: Page) -> int | None:
    candidates = [str(x[0]).strip() for x in page.layout if len(str(x[0]).strip()) <= 12]
    for text in reversed(candidates):
        m = NUM_LINE.match(text)
        if m:
            return int(m.group(1))
    return None


def _boilerplate_signatures(pages: list[Page]) -> list[frozenset[str]]:
    def norm(t: str) -> str:
        return re.sub(r"\d+", "#", t.strip().lower())

    counts = Counter()
    per_page = []
    for p in pages:
        lines = {norm(str(x[0])) for x in p.layout if 3 <= len(str(x[0]).strip()) <= 80}
        per_page.append(lines)
        counts.update(lines)
    common = {line for line, c in counts.items() if c >= 3 and c <= 0.95 * len(pages) + 1}
    return [frozenset(lines & common) for lines in per_page]


def deck_starts(pages: list[Page], vectors: np.ndarray, file_median: float | None) -> tuple[list[int], list[dict]]:
    """For one PDF's pages (in order) return the 0-based indices where decks
    start (always includes 0) and the evidence for each detected start."""
    cfg = config.structure()["boundaries"]
    w = cfg["weights"]
    n = len(pages)
    min_deck = int(cfg["min_deck_pages"])
    if n < 2 * min_deck:
        return [0], [{"page": pages[0].page_no, "reasons": ["start of file"]}] if pages else []
    nums = [_printed_number(p) for p in pages]
    sigs = _boilerplate_signatures(pages)
    starts, evidence = [0], [{"page": pages[0].page_no, "reasons": ["start of file"]}]
    for i in range(1, n):
        score, reasons = 0.0, []
        p = pages[i]
        sizes = _sizes(p)
        if (
            file_median
            and sizes
            and len(p.layout) <= int(cfg["title_max_lines"])
            and max(sizes) >= 1.5 * file_median
            and len(p.text) < 400
        ):
            score += w["title_slide"]
            reasons.append("title slide")
        if nums[i] is not None and (nums[i] == 1 or (nums[i - 1] is not None and nums[i] < nums[i - 1])):
            score += w["number_restart"]
            reasons.append(f"slide number restarts ({nums[i - 1]} -> {nums[i]})")
        nxt = sigs[i + 1] if i + 1 < n else None
        if sigs[i] != sigs[i - 1] and nxt is not None and sigs[i] == nxt and (sigs[i] or sigs[i - 1]):
            score += w["template_change"]
            reasons.append("header/footer changes")
        if vectors.shape[0] == n and 2 <= i <= n - 2:
            before = vectors[max(0, i - 2) : i].mean(axis=0)
            after = vectors[i : i + 2].mean(axis=0)
            denom = (np.linalg.norm(before) * np.linalg.norm(after)) or 1.0
            if float(before @ after) / denom < float(cfg["topic_shift_similarity"]):
                score += w["topic_shift"]
                reasons.append("topic shift")
        if score >= float(cfg["threshold"]) and i - starts[-1] >= min_deck:
            starts.append(i)
            evidence.append({"page": p.page_no, "score": round(score, 2), "reasons": reasons})
    if len(starts) > 1 and n - starts[-1] < min_deck:
        starts.pop()
        evidence.pop()
    return starts, evidence


def boundary_accuracy(predicted: dict[str, list[int]], truth: dict[str, list[int]], tolerance: int = 1) -> dict:
    """Precision/recall of detected deck starts against hand-marked starts
    (page numbers). Page 1 of each file is ignored since it is always a start."""
    tp = fp = fn = 0
    for name, true_starts in truth.items():
        t = sorted(s for s in true_starts if s != 1)
        pr = sorted(s for s in predicted.get(name, []) if s != 1)
        matched = set()
        for s in pr:
            hit = next((x for x in t if abs(x - s) <= tolerance and x not in matched), None)
            if hit is None:
                fp += 1
            else:
                matched.add(hit)
                tp += 1
        fn += len(t) - len(matched)
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": tp,
        "false_positive": fp,
        "missed": fn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
    }


# ------------------------------------------------------------ topic segments


def segment(vectors: np.ndarray, headings: list[str | None]) -> list[tuple[int, int]]:
    """Split one deck (pages in order) into contiguous topic segments.
    Returns (start, end) index pairs, end inclusive."""
    cfg = config.structure()["topics"]
    n = vectors.shape[0]
    target, min_p, max_p = int(cfg["target_pages_per_topic"]), int(cfg["min_pages"]), int(cfg["max_pages"])
    k = max(1, min(round(n / target), n // max(min_p, 1)))
    if k <= 1 or n <= min_p * 2:
        cuts: list[int] = []
    else:
        from scipy.sparse import diags
        from sklearn.cluster import AgglomerativeClustering

        conn = diags([1.0, 1.0], [-1, 1], shape=(n, n))
        x = vectors if np.any(vectors) else np.eye(n)
        labels = AgglomerativeClustering(n_clusters=k, connectivity=conn, linkage="ward").fit_predict(x)
        cuts = [i for i in range(1, n) if labels[i] != labels[i - 1]]
        # Snap cuts to a nearby new heading: slide decks usually change topic at a heading.
        changes = {
            i for i in range(1, n) if headings[i] and (headings[i] or "").lower() != (headings[i - 1] or "").lower()
        }
        snapped = []
        for c in cuts:
            near = [h for h in (c - 1, c + 1) if h in changes and c not in changes]
            snapped.append(near[0] if near else c)
        cuts = sorted(set(snapped))
    # Enforce minimum size by merging small segments into their neighbour.
    bounds = [0, *cuts, n]
    segs = [(bounds[i], bounds[i + 1] - 1) for i in range(len(bounds) - 1)]
    merged: list[tuple[int, int]] = []
    for s in segs:
        if merged and (s[1] - s[0] + 1) < min_p:
            merged[-1] = (merged[-1][0], s[1])
        else:
            merged.append(s)
    if len(merged) > 1 and merged[0][1] - merged[0][0] + 1 < min_p:
        merged[1] = (merged[0][0], merged[1][1])
        merged.pop(0)
    # Enforce maximum size by splitting evenly.
    out = []
    for a, b in merged:
        length = b - a + 1
        parts = -(-length // max_p)
        step = -(-length // parts)
        for start in range(a, b + 1, step):
            out.append((start, min(b, start + step - 1)))
    return out
