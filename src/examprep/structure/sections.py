"""Turn slides and notes into sections and topics the way a student would read them.

Works on any college's files, without knowing their template in advance:

1. Repeated lines are ignored: course name, instructor, department, college,
   email, "Thank you", book headers on every other page. A line counts as
   repeated when it appears in at least half of the decks (with 3 or more
   decks), or on nearly every page (or every other page) of one long file.
   The ledger keeps the full text; this only affects topics.
2. Headings are found from several clues, each judged against the file's own
   body text: a bigger font, bold text when the body is not bold, numbering
   such as "2.3", a short title-like line, and the slide title box. A heading
   split over two lines ("DATABASES AND" / "DATABASE USERS") is joined.
3. Each page gets its main heading; the other headings on the page are kept as
   its points, so nothing is dropped.
4. Cover, contents, outline and closing pages are "filler" and are attached to
   a neighbouring section, so they never become a topic.
5. Consecutive pages with the same heading form a section ("X (cont.)" and
   "X: part" count as X).
6. Topics: when a file has chapter or unit headings (a clearly bigger style
   used on a few pages) or numbered headings (1.1, 1.2, 2.1 ...), each chapter
   is a topic. Otherwise small sections are merged with their most similar
   neighbour into topics of a sensible size.
"""

from __future__ import annotations

import os
import re
import statistics
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from examprep import config
from examprep.structure.store import Page

ALWAYS_SKIP = re.compile(
    r"^(thank\s*you\.?!?|thanks\.?|any questions\??|questions\??|\S+@\S+\.\S+|https?://\S+|www\.\S+|\d{1,3}|"
    r"\d{1,3}\s*(/|of)\s*\d{1,3}|(slide|page)\s*\d{1,3}|"
    r"(the\s+)?end(\s+of\s+(the\s+)?(chapter|unit|lecture|module|presentation|session)(\s+[\w.]+)?)?[.!]?)$",
    re.I,
)
# Pages with these headings are front or back matter, not topics.
FRONT_MATTER = re.compile(
    r"^(contents?|table of contents|index|outline|lecture outline|course outline|agenda|"
    r"references?|bibliography|acknowledge?ments?|reading list|further reading)\s*:?$",
    re.I,
)
# Headings that are fine as sections but say nothing on their own as a topic name.
GENERIC = re.compile(
    r"^(introduction|overview|summary|lecture summary|conclusions?|recap|review|examples?|"
    r"learning (goals|objectives|outcomes)|objectives|syllabus|exercises?|problems?)\s*:?$",
    re.I,
)
NUMBERED = re.compile(r"^((?:\d{1,2}\.){0,3}\d{1,2})\.?\s+(?=[A-Za-z(\[])")
CONTINUED = re.compile(
    r"\s*[-–:,]?\s*[\(\[]?\s*(cont(inued|'?d)?|contd)\.?\s*[\)\]]?\s*$|"
    r"\s*[-–:,]?\s*\(?\s*(part|pt\.?)\s*([ivx]+|\d+)\s*\)?\s*$|"
    r"\s*[-–]\s*([ivx]+|\d+)\s*$|\s*\((\d+|[ivx]+)\)\s*$",
    re.I,
)
# Labels inside notes that are points, never a page's main heading.
POINT_LABEL = re.compile(
    r"^(notes?|examples?|solution|answer|important\s+points?|points?|steps?|case|result|proof|definition|"
    r"advantages?|disadvantages?|limitations?|properties|syntax|output|input)\b\s*[\divx]{0,3}\s*[:.\-–)]?\s*$",
    re.I,
)
CAPTION = re.compile(
    r"^(fig(ure)?\.?\s*[\d.:]|fig(ure)?\s*:|table\s*\d|example\s*\d*\s*:|note\s*:|\(?\d{1,2}\)|[a-z]\))", re.I
)
# "Chapter 15: Query Processing", "Unit 2 - Joins", "Lecture 3." -> the label part.
LABEL = re.compile(
    r"^(chapter|unit|module|lecture|lec|topic|part|week|session|section)\s*[\divxlc]+\s*[:.\-–]\s*", re.I
)
BULLETS = " \t•▪●◦‣∙·-–—*|>»■□❖➢✓"


def norm(line: str) -> str:
    t = re.sub(r"\d+", "#", line.lower())
    t = re.sub(r"[^\w#]+", " ", t)
    return " ".join(t.split())


def page_lines(page: Page) -> list[str]:
    return [ln.strip(BULLETS).strip() for ln in page.text.split("\n") if ln.strip(BULLETS).strip()]


def _keys(lines: list[str]) -> set[str]:
    return {norm(x) for x in lines} | {norm(a + " " + b) for a, b in zip(lines, lines[1:], strict=False)}


def boilerplate(units: list[list[Page]]) -> set[str]:
    """Normalised lines that repeat across decks, or across nearly every page
    (or every other page, as book headers do) of one long file or deck."""
    cfg = config.structure().get("sections", {})
    share_units = float(cfg.get("repeated_in_decks", 0.5))
    share_pages = float(cfg.get("repeated_on_pages", 0.9))
    min_pages = int(cfg.get("single_deck_min_pages", 8))
    per_unit = Counter()
    boiler: set[str] = set()
    for unit in units:
        seen = [Counter(), Counter(), Counter()]  # all pages, odd pages, even pages
        for k, p in enumerate(unit):
            layout = [str(e[0]).strip(BULLETS).strip() for e in p.layout if str(e[0]).strip(BULLETS).strip()]
            for ln in _keys(page_lines(p)) | _keys(layout):
                if ln:
                    seen[0][ln] += 1
                    seen[1 + k % 2][ln] += 1
        per_unit.update(seen[0].keys())
        if len(unit) >= min_pages:
            boiler |= {ln for ln, c in seen[0].items() if c >= share_pages * len(unit)}
            for side, count in ((1, (len(unit) + 1) // 2), (2, len(unit) // 2)):
                if count >= 4:
                    boiler |= {ln for ln, c in seen[side].items() if c >= share_pages * count}
    if len(units) >= 3:
        boiler |= {ln for ln, c in per_unit.items() if c >= max(2, share_units * len(units))}
    return boiler


def _skip(line: str, boiler: set[str]) -> bool:
    return norm(line) in boiler or ALWAYS_SKIP.match(line) is not None


def clean_lines(page: Page, boiler: set[str]) -> list[str]:
    lines = page_lines(page)
    keep = []
    for i, ln in enumerate(lines):
        if _skip(ln, boiler):
            continue
        # A repeated line split over two lines on some slides ("DATABASE" / "TECHNOLOGIES").
        if i + 1 < len(lines) and norm(ln + " " + lines[i + 1]) in boiler:
            continue
        if i > 0 and norm(lines[i - 1] + " " + ln) in boiler:
            continue
        keep.append(ln)
    return keep


def _looks_like_heading(line: str) -> bool:
    words = line.split()
    return 2 < len(line) <= 90 and 1 <= len(words) <= 12 and re.search(r"[A-Za-z]{3}", line) is not None


def _title_like(line: str) -> bool:
    """Short, not a sentence, does not start in lower case (unless numbered)."""
    ellipsis = line.rstrip().endswith(("...", "…"))
    if not _looks_like_heading(line) or (line.rstrip().endswith((".", ",", ";")) and not ellipsis):
        return False
    words = re.findall(r"[A-Za-z]+", line)
    if not words or sum(map(len, words)) / len(words) < 2.5:
        return False  # "R B C ABC": table headers and formulas
    letters = sum(ch.isalpha() or ch.isspace() for ch in line)
    if letters < 0.7 * len(line) or line.count("|") or line.count("[") > 1:
        return False  # table rows, formulas, OCR noise
    first = line.lstrip("([")[:1]
    return not first.islower()


def number_depth(text: str) -> int:
    """'2' -> 1, '2.3 Joins' -> 2, '2.3.1 Hash join' -> 3, no numbering -> 0."""
    m = NUMBERED.match(text)
    return m.group(1).count(".") + 1 if m else 0


def strip_number(text: str) -> str:
    return NUMBERED.sub("", text, count=1).strip() if NUMBERED.match(text) else text.strip()


# ------------------------------------------------------------ file styles


@dataclass
class FileStyle:
    body_size: float | None  # typical size of body text in this file
    body_bold: bool  # True when most body text is bold (then bold says nothing)
    notes: bool  # dense, document-like pages (typed notes, book chapters) rather than slides
    bullets: frozenset[str] = frozenset()  # OCR misreads of bullet symbols in this file ("4", "e", "O")


def file_styles(pages: list[Page]) -> dict[int, FileStyle]:
    cfg = config.structure().get("sections", {})
    notes_words = int(cfg.get("notes_words_per_page", 170))
    by_file: dict[int, list[Page]] = {}
    for p in pages:
        by_file.setdefault(p.file_id, []).append(p)
    out = {}
    for fid, fpages in by_file.items():
        sized: list[tuple[float, float, bool]] = []  # (size, weight, bold)
        for p in fpages:
            for entry in p.layout:
                t, size = str(entry[0]).strip(), entry[1] if len(entry) > 1 else None
                bold = bool(entry[2]) if len(entry) > 2 else False
                if size and len(t) >= 20:
                    sized.append((float(size), float(len(t)), bold))
        body = bold_share = None
        if sized:
            sized.sort()
            half, acc = sum(w for _, w, _ in sized) / 2, 0.0
            for s, w, _ in sized:
                acc += w
                if acc >= half:
                    body = s
                    break
            bold_share = sum(w for _, w, b in sized if b) / sum(w for _, w, _ in sized)
        # On scans, bullet symbols are read as a stray character ("4 Databases ...",
        # "e Flat file model"). A short token starting many OCR lines is such a bullet.
        firsts = Counter()
        ocr_lines = 0
        for p in fpages:
            for entry in p.layout:
                if len(entry) > 2 and entry[2] is None:
                    ocr_lines += 1
                    tok = str(entry[0]).split(" ", 1)[0]
                    if len(tok) <= 2 and not re.fullmatch(r"\d{1,2}\.", tok):
                        firsts[tok] += 1
                    elif re.match(r"^\d[A-Z]", tok):
                        firsts[tok[0]] += 1
        min_count = max(5, 0.08 * ocr_lines)
        bullets = frozenset(t for t, c in firsts.items() if c >= min_count and t not in {"A", "I", "a"})
        words = [len(p.text.split()) for p in fpages if p.text.strip()]
        dense = bool(words) and statistics.median(words) >= notes_words and fpages[0].kind != "pptx"
        out[fid] = FileStyle(body, bool(bold_share and bold_share > 0.5), dense, bullets)
    return out


# ------------------------------------------------------------ headings


@dataclass
class Head:
    text: str
    size: float  # 0 when unknown
    bold: bool
    pos: int  # line position on the page
    depth: int  # numbering depth, 0 when not numbered


def _layout_lines(page: Page, boiler: set[str], bullets: frozenset[str] = frozenset()) -> list:
    out = []
    for entry in page.layout:
        t = str(entry[0]).strip(BULLETS).strip()
        if bullets and len(entry) > 2 and entry[2] is None:
            for b in bullets:
                if t.startswith(b) and (t[len(b) : len(b) + 1] == " " or t[len(b) : len(b) + 1].isupper()):
                    t = t[len(b) :].strip()
                    break
        size = float(entry[1] or 0) if len(entry) > 1 else 0.0
        bold = entry[2] if len(entry) > 2 else None
        if t and not _skip(t, boiler):
            out.append((t, size, bold if bold is None else bool(bold)))
    return out


def heading_lines(page: Page, style: FileStyle | None, boiler: set[str]) -> list[Head]:
    """Every heading-like line on a page (from the text layer or OCR line sizes)."""
    if not page.layout or not style or not style.body_size:
        return []
    lines = _layout_lines(page, boiler, style.bullets)
    ocr = any(b is None for _, _, b in lines)
    big = 1.2 if ocr else 1.12  # OCR sizes are estimates, so ask for a clearer difference
    tol = 0.12 if ocr else 0.03

    def strong(size: float, bold: bool | None) -> bool:
        ratio = size / style.body_size
        return ratio >= big or (bool(bold) and not style.body_bold and ratio >= 0.95)

    # Join a heading split over two lines with the same style.
    joined: list[tuple[str, float, bool | None, int]] = []
    i = 0
    while i < len(lines):
        t, s, b = lines[i]
        j = i
        while j + 1 < len(lines):
            t2, s2, b2 = lines[j + 1]
            connector = re.search(r"\b(and|of|the|for|in|to|on|with|&|a|an|vs\.?)$|[,&\-–]$", t, re.I)
            same = s and s2 and b == b2 and abs(s - s2) <= (0.25 if connector else tol) * max(s, s2)
            ends_open = not t.rstrip().endswith((".", ":", ";", "!", "?"))
            follows = t2[:1].isupper() or connector
            if (
                same
                and strong(s, b)
                and strong(s2, b2)
                and ends_open
                and follows
                and not number_depth(t2)
                and len(t) + len(t2) <= 90
                and len(t.split()) <= 6
            ):
                t = f"{t} {t2}"
                j += 1
            else:
                break
        joined.append((t, s, b, i))
        i = j + 1

    heads = []
    for t, s, b, pos in joined:
        if not s or not _title_like(t) or CAPTION.match(t) or _skip(t, boiler):
            continue
        depth = number_depth(t)
        if ocr and depth == 1 and not re.match(r"^\d{1,2}\.\s", t):
            depth = 0  # on scans "4 Collection ..." is usually a misread bullet, not a number
            if s / style.body_size < big:
                continue
        if strong(s, b) or (depth and s >= style.body_size * (1.05 if ocr else 0.98) and (b or ocr)):
            if ocr and pos > 2 and not depth:
                continue  # OCR size noise: on scans only trust lines near the top, or numbered ones
            heads.append(Head(t, s, bool(b), pos, depth))
    return heads


def headings(
    page: Page, lines: list[str], style: FileStyle | None, boiler: set[str]
) -> tuple[str | None, list[str], list[Head]]:
    """(main heading, points, all heading lines) for one page."""
    heads = heading_lines(page, style, boiler)
    head = None
    if page.kind == "pptx" and page.title and not _skip(page.title, boiler) and _looks_like_heading(page.title):
        head = " ".join(page.title.split())
    elif heads:
        main = [h for h in heads if not POINT_LABEL.match(strip_number(h.text))]
        if style and style.notes and style.body_size:
            # In notes, bold lines at body size are points; a page without a bigger
            # heading continues the section before it.
            main = [h for h in main if h.size >= 1.12 * style.body_size]
        numbered = [h for h in main if h.depth]
        if numbered:  # in numbered notes the numbered line is the section, unless a much bigger title exists
            top = max(h.size for h in numbered)
            main = [h for h in main if not h.depth and h.size >= 1.3 * top] or numbered
        if main:
            head = max(main, key=lambda h: (round(h.size), h.bold, -h.pos)).text
    elif not page.layout and lines and _title_like(lines[0]) and not lines[0].rstrip().endswith(":"):
        head = lines[0]
    elif page.kind == "pptx" and lines and _title_like(lines[0]):
        head = lines[0]
    points = [h.text for h in heads if h.text != head]
    if head and not points:
        # The first line under the heading when it reads like a sub-heading ("Shared-Disk Machines").
        rest = [ln for ln in lines if ln != head and not norm(ln).startswith(norm(head)) and norm(ln) not in norm(head)]
        if rest:
            cand = rest[0].split(":", 1)[0].strip() if ":" in rest[0] else rest[0].strip()
            words = cand.split()
            capitalised = sum(1 for w in words if w[:1].isupper() or not w[:1].isalpha())
            title_like = len(words) <= 3 or capitalised >= 0.6 * len(words) or rest[0].rstrip().endswith(":")
            if _title_like(cand) and len(words) <= 8 and title_like:
                points = [cand]
    return head, list(dict.fromkeys(points)), heads


def is_front_matter(head: str | None) -> bool:
    return bool(head) and FRONT_MATTER.match(strip_number(head)) is not None


def is_generic(head: str | None) -> bool:
    return not head or GENERIC.match(strip_number(head)) is not None or is_front_matter(head)


# ------------------------------------------------------------ continuation slides


def base_headings(heads: list[str | None]) -> list[str | None]:
    """'Hash-Join (Cont.)' -> 'Hash-Join'; 'Other Operations : Aggregation' ->
    'Other Operations' when 'Other Operations' is itself a heading in the file or
    starts several headings."""
    stripped = [CONTINUED.sub("", h).strip() if h else h for h in heads]
    known = {norm(h) for h in stripped if h}
    prefixes = Counter(norm(h.split(":", 1)[0]) for h in {h for h in stripped if h and ":" in h})
    out = []
    for h in stripped:
        if h and ":" in h:
            left = h.split(":", 1)[0].strip()
            if (
                len(left) >= 4
                and not is_generic(left)
                and not LABEL.match(left + ":")
                and (norm(left) in known or prefixes[norm(left)] >= 2)
            ):
                h = left
        out.append(h or None)
    return out


# ------------------------------------------------------------ sections


@dataclass
class Section:
    start: int  # index into the deck's page list
    end: int  # inclusive
    heading: str | None
    subheadings: list[str] = field(default_factory=list)

    @property
    def size(self) -> int:
        return self.end - self.start + 1


def build_sections(heads: list[str | None], subs: list[list[str] | str | None], filler: list[bool]) -> list[Section]:
    """Runs of pages with the same heading. Filler pages and pages without a
    heading join the section before them (or the first section at the start)."""
    bases = base_headings(heads)

    def points(i: int) -> list[str]:
        s = subs[i]
        found = [s] if isinstance(s, str) else list(s or [])
        if heads[i] and bases[i] and norm(heads[i]) != norm(bases[i]):
            extra = heads[i][len(bases[i]) :].strip(" :-–")  # "Other Operations : Aggregation" -> "Aggregation"
            extra = CONTINUED.sub("", extra).strip()
            if extra and _looks_like_heading(extra):
                found.insert(0, extra)
        return found

    def add_points(sec: Section, i: int) -> None:
        for s in points(i):
            if s not in sec.subheadings and (not sec.heading or norm(s) != norm(sec.heading)):
                sec.subheadings.append(s)

    sections: list[Section] = []
    pending_start: int | None = None
    for i, (h, f) in enumerate(zip(bases, filler, strict=True)):
        if f or not h:
            if sections:
                sections[-1].end = i
                if not f:
                    add_points(sections[-1], i)
            elif pending_start is None:
                pending_start = i
            continue
        if sections and sections[-1].heading and norm(sections[-1].heading) == norm(h):
            sections[-1].end = i
            if sections[-1].heading != h:  # "... - Part 1" then "... - Part 2": keep the shared part
                prefix = os.path.commonprefix([sections[-1].heading, h])
                prefix = (prefix.rsplit(" - ", 1)[0] if " - " in prefix else prefix).rstrip(" -–:0123456789")
                if len(prefix) >= 8:
                    sections[-1].heading = prefix
        else:
            start = pending_start if pending_start is not None and not sections else i
            sections.append(Section(start, i, h))
            pending_start = None
        add_points(sections[-1], i)
    if not sections:
        return [Section(0, len(heads) - 1, None)] if heads else []
    return sections


def group_sections(sections: list[Section], vectors: np.ndarray) -> list[list[Section]]:
    """Merge small sections into neighbours (by similarity) until topics reach
    a minimum size; very long sections are later split by size."""
    cfg = config.structure()["topics"]
    min_pages, max_pages = int(cfg.get("min_pages", 3)), int(cfg.get("max_pages", 25))
    groups = [[s] for s in sections]

    def size(g: list[Section]) -> int:
        return g[-1].end - g[0].start + 1

    def centroid(g: list[Section]) -> np.ndarray:
        v = vectors[g[0].start : g[-1].end + 1].mean(axis=0)
        n = np.linalg.norm(v)
        return v / n if n else v

    while len(groups) > 1:
        small = [i for i, g in enumerate(groups) if size(g) < min_pages]
        if not small:
            break
        i = min(small, key=lambda k: (size(groups[k]), k))
        options = [j for j in (i - 1, i + 1) if 0 <= j < len(groups)]
        fits = [j for j in options if size(groups[i]) + size(groups[j]) <= max_pages] or options
        c = centroid(groups[i])
        j = max(fits, key=lambda k: (float(c @ centroid(groups[k])), -size(groups[k])))
        a, b = sorted((i, j))
        groups[a] = groups[a] + groups[b]
        del groups[b]
    return groups


# ------------------------------------------------------------ chapters


@dataclass
class Chapter:
    start: int  # index into the file's (or deck's) page list
    title: str | None


def _valid_starts(starts: list[int], n: int, min_gap: int, max_share: float) -> bool:
    if len(starts) < 2 or len(starts) > max(2, max_share * n):
        return False
    bounds = [*starts, n]
    gaps = [bounds[k + 1] - bounds[k] for k in range(len(starts))]
    return statistics.mean(gaps) >= min_gap


def chapters(
    heads_per_page: list[list[Head]], main: list[str | None], filler: list[bool], notes: bool
) -> tuple[list[Chapter], str | None]:
    """Chapter or unit starts in one file and how they were found, or ([], None)
    when there is no such level.

    1. Style (notes only): a heading style clearly bigger than the usual page
       heading, used on a few pages ("DATABASE DESIGN", "ER MODEL", ...).
    2. Numbering: headings numbered 1.1, 1.2, 2.1 ... -> chapter 1, 2, ...
    """
    cfg = config.structure()["topics"]
    min_gap = int(cfg.get("min_pages", 3))
    n = len(heads_per_page)
    max_share = 0.35

    if notes:
        main_sizes = [
            round(next((h.size for h in hs if h.text == m), 0))
            for hs, m, f in zip(heads_per_page, main, filler, strict=True)
            if m and not f
        ]
        main_sizes = [s for s in main_sizes if s]
        if main_sizes:
            usual = Counter(main_sizes).most_common(1)[0][0]
            sizes = sorted({round(h.size) for hs in heads_per_page for h in hs if h.size >= 1.15 * usual})
            best: list[int] | None = None
            for size in sizes:  # smallest qualifying size first = the finest chapter level
                starts = [
                    k
                    for k, hs in enumerate(heads_per_page)
                    if not filler[k] and any(size <= round(h.size) <= size * 1.1 + 0.5 for h in hs)
                ]
                if _valid_starts(starts, n, min_gap, max_share):
                    best = starts
                    break
            if best:
                out = []
                for k in best:
                    top = max(
                        (h for h in heads_per_page[k] if size <= round(h.size) <= size * 1.1 + 0.5),
                        key=lambda h: (round(h.size), -h.pos),
                    )
                    same = [h.text for h in heads_per_page[k] if round(h.size) == round(top.size)]
                    out.append(Chapter(k, " ".join(same)[:90]))
                return out, "chapter headings"

    # Numbering: the first component of 2nd/3rd-level numbers names the chapter.
    keyed = []
    for k, hs in enumerate(heads_per_page):
        if filler[k]:
            continue
        deep = [h for h in hs if h.depth >= 2]
        level2 = min((h.size for h in deep), default=None)
        top = [h for h in hs if h.depth == 1 and (h.text == main[k] if level2 is None else h.size >= level2 * 0.95)]
        first = sorted(deep + top, key=lambda h: h.pos)
        if first:
            keyed.append((k, NUMBERED.match(first[0].text).group(1).split(".")[0], first[0]))
    if sum(1 for hs in heads_per_page if any(h.depth >= 2 for h in hs)) < 3:
        return [], None
    starts: list[Chapter] = []
    current = None
    for k, key, h in keyed:
        if key != current:
            title = strip_number(h.text) if h.depth == 1 else None
            starts.append(Chapter(k, title))
            current = key
    if len({key for _, key, _ in keyed}) < 2 or not _valid_starts([c.start for c in starts], n, min_gap, 0.5):
        return [], None
    return starts, "numbered headings"
