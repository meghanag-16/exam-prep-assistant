"""Quick revision for a topic or a whole subject: keywords, short points, comparison
tables and a formula table.

Keywords, short points and comparisons are drawn from the topic's notes: the model refers
to note points by number, so every item traces back to pages (an item that names no valid
point is dropped). The formula table lists the formulas found on the topic's pages
(examprep-formulas) with their meaning, symbols and source page; the model adds units,
when to use the formula and a worked example, and the example's result is computed with
SymPy, not by the model. Units, "when to use" and examples are marked as generated.

Stored in the ledger (table revision) and made again only when the notes, the formulas
or the settings change.
"""

from __future__ import annotations

import hashlib
import json
import keyword
import re
import sqlite3
from collections.abc import Callable

from examprep import config, llm
from examprep.ingest import ledger
from examprep.notes import build

Log = Callable[[str], None]
PROMPT_VERSION = "revision-1"

SCHEMA = """
CREATE TABLE IF NOT EXISTS revision (
    topic_id    INTEGER PRIMARY KEY,
    fingerprint TEXT NOT NULL,
    data        TEXT NOT NULL,
    created     TEXT
);
"""


def migrate(con: sqlite3.Connection) -> None:
    build.migrate(con)
    con.executescript(SCHEMA)
    con.commit()


# ------------------------------------------------------------ SymPy worked examples

_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]*")


def _functions() -> dict:
    import sympy

    return {
        "sqrt": sympy.sqrt,
        "log": sympy.log,
        "ln": sympy.log,
        "exp": sympy.exp,
        "sin": sympy.sin,
        "cos": sympy.cos,
        "tan": sympy.tan,
        "asin": sympy.asin,
        "acos": sympy.acos,
        "atan": sympy.atan,
        "abs": sympy.Abs,
        "Abs": sympy.Abs,
        "floor": sympy.floor,
        "ceiling": sympy.ceiling,
        "factorial": sympy.factorial,
        "binomial": sympy.binomial,
        "Min": sympy.Min,
        "Max": sympy.Max,
        "pi": sympy.pi,
        "E": sympy.E,
    }


def _parse(expr: str, names: set[str]):
    """Parse an arithmetic expression written by the model, allowing only numbers, the given
    variable names, + - * / ^ ( ) and a fixed list of maths functions. Anything else
    (attribute access, other names, strings) is refused before SymPy sees it."""
    import sympy
    from sympy.parsing.sympy_parser import parse_expr

    expr = expr.strip()
    if not expr or len(expr) > 300 or not re.fullmatch(r"[A-Za-z0-9_+\-*/^().,\s]+", expr) or "__" in expr:
        raise ValueError("not a plain arithmetic expression")
    if re.search(r"\.(?!\d)|(?<![\d\s(+\-*/^,])\.", expr):
        raise ValueError("a dot outside a number")
    funcs = _functions()
    unknown = {n for n in _NAME.findall(expr) if n not in names and n not in funcs}
    # Exponents such as 1e-3 contain an "e" that is part of a number, not a name.
    unknown = {n for n in unknown if not re.fullmatch(r"e\d*", n) or n in names}
    if unknown:
        raise ValueError(f"unknown names: {', '.join(sorted(unknown))}")
    local = funcs | {n: sympy.Symbol(n) for n in names}  # a variable named E or pi is the variable
    glob = {"__builtins__": {}, "Integer": sympy.Integer, "Float": sympy.Float, "Rational": sympy.Rational}
    glob["Symbol"] = sympy.Symbol
    return parse_expr(expr.replace("^", "**"), local_dict=local, global_dict=glob)


def solve_example(equation: str, given: dict, find: str) -> str | None:
    """The value of `find` from "lhs = rhs" with the given values, computed by SymPy; None if it fails."""
    import sympy

    if not isinstance(given, dict) or not isinstance(find, str) or equation.count("=") != 1:
        return None
    names = {str(k) for k in given} | {find}
    if any(not _NAME.fullmatch(n) or keyword.iskeyword(n) for n in names):
        return None
    try:
        lhs, rhs = (_parse(side, names) for side in equation.split("="))
        values = {sympy.Symbol(str(k)): sympy.nsimplify(float(v)) for k, v in given.items()}
        sols = sympy.solve(sympy.Eq(lhs, rhs).subs(values), sympy.Symbol(find))
    except Exception:  # noqa: BLE001 (a bad example is dropped, never shown wrong)
        return None
    real = []
    for s in sols:
        try:
            v = complex(sympy.N(s))
        except (TypeError, ValueError):
            continue
        if abs(v.imag) < 1e-9:
            real.append(v.real)
    if not real:
        return None
    x = sorted(real, key=lambda x: (x < 0, abs(x)))[0]  # a positive value first (lengths, times, counts)
    return str(int(round(x))) if abs(x - round(x)) < 1e-9 and abs(x) < 1e15 else f"{x:.6g}"


# ------------------------------------------------------------ inputs


def _formulas(con: sqlite3.Connection, page_ids: list[int]) -> list[dict]:
    if (
        not page_ids
        or not con.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'formulas'").fetchone()
    ):
        return []
    marks = ",".join("?" * len(page_ids))
    rows = con.execute(
        "SELECT fo.id, fo.latex, fo.name, fo.meaning, fo.symbols, fo.norm, fo.page_no, f.name AS file"
        f" FROM formulas fo JOIN files f ON f.id = fo.file_id WHERE fo.valid = 1 AND fo.page_id IN ({marks})"
        " ORDER BY f.name, fo.page_no, fo.id",
        page_ids,
    ).fetchall()
    out, seen = [], set()
    for r in rows:
        if r["norm"] in seen:
            continue
        seen.add(r["norm"])
        try:
            symbols = json.loads(r["symbols"] or "[]")
        except ValueError:
            symbols = []
        out.append(
            {
                "id": r["id"],
                "latex": r["latex"],
                "name": r["name"],
                "meaning": r["meaning"],
                "symbols": symbols if isinstance(symbols, list) else [],
                "source": f"{r['file'].replace(chr(92), '/').rsplit('/', 1)[-1]} p.{r['page_no']}",
            }
        )
    return out


def _ask(rules: list[str], user: str) -> dict | None:
    try:
        reply = llm.chat_json(
            [
                {"role": "system", "content": llm.SYSTEM_GUARD + "\n\n" + "\n".join(f"- {r}" for r in rules)},
                {"role": "user", "content": user},
            ],
            max_tokens=4096,
            trace_name="write-revision",
        )
    except llm.LLMError:
        return None
    return reply if isinstance(reply, dict) else None


def _summary(topic: str, points: list[dict]) -> dict:
    cfg = config.answer()["revision"]
    rules = [
        "Make a quick-revision sheet for one topic from the numbered note points in <material>.",
        f"keywords: up to {cfg.get('keywords', 12)} key terms, each with a one-line meaning taken from the points.",
        f"short_points: up to {cfg.get('short_points', 10)} one-line facts a student must remember.",
        "comparisons: a table only when the points compare two or more things (methods, types, algorithms); "
        "otherwise an empty list.",
        "Every item lists in points the numbers of the note points it comes from. Use only what the points say.",
        'Reply with JSON only: {"keywords": [{"term": "...", "meaning": "...", "points": [1]}], '
        '"short_points": [{"text": "...", "points": [2]}], '
        '"comparisons": [{"title": "...", "columns": ["Aspect", "A", "B"], "rows": [["...", "...", "..."]], '
        '"points": [3, 4]}]}',
    ]
    listing = "\n".join(f"{n}. {p['text']}" for n, p in enumerate(points, start=1))
    reply = _ask(rules, f"Topic: {topic}\n\n<material>\n{listing}\n</material>") or {}

    def refs(item) -> list[int]:
        nums = []
        for x in item.get("points") or []:
            try:
                k = int(x)
            except (TypeError, ValueError):
                continue
            if 1 <= k <= len(points) and k not in nums:
                nums.append(k)
        return nums

    def pages(nums) -> list[int]:
        return sorted({i for k in nums for i in points[k - 1]["page_ids"]})

    out: dict = {"keywords": [], "short_points": [], "comparisons": [], "dropped": 0}
    for it in reply.get("keywords") or []:
        nums = refs(it) if isinstance(it, dict) else []
        if nums and it.get("term") and it.get("meaning"):
            out["keywords"].append({"term": str(it["term"]), "meaning": str(it["meaning"]), "page_ids": pages(nums)})
        else:
            out["dropped"] += 1
    for it in reply.get("short_points") or []:
        nums = refs(it) if isinstance(it, dict) else []
        if nums and it.get("text"):
            out["short_points"].append({"text": str(it["text"]), "page_ids": pages(nums)})
        else:
            out["dropped"] += 1
    for it in reply.get("comparisons") or []:
        nums = refs(it) if isinstance(it, dict) else []
        cols = [str(c) for c in (it.get("columns") or [])] if isinstance(it, dict) else []
        rows = [[str(c) for c in r] for r in (it.get("rows") or []) if isinstance(r, list)] if cols else []
        rows = [r[: len(cols)] + [""] * (len(cols) - len(r)) for r in rows]
        if nums and cols and rows:
            out["comparisons"].append(
                {"title": str(it.get("title") or ""), "columns": cols, "rows": rows, "page_ids": pages(nums)}
            )
        else:
            out["dropped"] += 1
    return out


def _formula_table(topic: str, formulas: list[dict]) -> list[dict]:
    if not formulas:
        return []
    rules = [
        "For each formula in <material>, give: units (the units of the result, or null when it has none or "
        "they are not clear), when_to_use (one sentence), and a small worked example.",
        "The example gives values for every symbol except one, the symbol to find, and the equation in "
        "plain arithmetic: lhs = rhs, using only the symbol names in given and find, numbers, + - * / ^ ( ) "
        "and sqrt, log, exp, sin, cos, tan. Symbol names are letters, digits and _ only (write lambda as lam). "
        "Do not compute the answer.",
        "If a formula cannot be turned into a numeric example (a definition, a set expression), example is null.",
        'Reply with JSON only: {"formulas": [{"id": "F1", "units": "...", "when_to_use": "...", '
        '"example": {"story": "one line", "given": {"n": 4, "p": 0.5}, "find": "S", '
        '"equation": "S = 1/((1 - p) + p/n)"}}]}',
    ]
    listing = "\n".join(
        f"[F{n}] {f['latex']}"
        + (f" ({f['name']})" if f.get("name") else "")
        + (f": {f['meaning']}" if f.get("meaning") else "")
        + (
            " Symbols: "
            + "; ".join(f"{s.get('symbol')} = {s.get('meaning')}" for s in f["symbols"] if isinstance(s, dict))
            if f["symbols"]
            else ""
        )
        for n, f in enumerate(formulas, start=1)
    )
    reply = _ask(rules, f"Topic: {topic}\n\n<material>\n{listing}\n</material>") or {}
    extra = {}
    for it in reply.get("formulas") or []:
        if isinstance(it, dict) and re.fullmatch(r"F\d+", str(it.get("id", ""))):
            extra[int(str(it["id"])[1:])] = it
    table = []
    for n, f in enumerate(formulas, start=1):
        e = extra.get(n, {})
        row = {**f, "units": e.get("units") or None, "when_to_use": e.get("when_to_use") or None, "example": None}
        ex = e.get("example")
        if isinstance(ex, dict):
            result = solve_example(str(ex.get("equation") or ""), ex.get("given") or {}, str(ex.get("find") or ""))
            if result is not None:
                row["example"] = {
                    "story": str(ex.get("story") or ""),
                    "given": {str(k): v for k, v in ex["given"].items()},
                    "find": str(ex["find"]),
                    "equation": str(ex["equation"]),
                    "result": result,
                }
        table.append(row)
    return table


# ------------------------------------------------------------ entry points


def topic_revision(subject: str, topic_id: int, redo: bool = False, log: Log = print, con=None) -> dict:
    own = con is None
    con = con or ledger.connect(ledger.safe_subject(subject))
    try:
        migrate(con)
        notes = build.topic_notes(subject, topic_id, False, log, con)
        points = [p for sec in notes["sections"] for p in sec["points"]]
        page_ids = sorted({i for p in points for i in p["page_ids"]})
        formulas = _formulas(con, [r[0] for r in con.execute("SELECT id FROM pages WHERE topic_id = ?", (topic_id,))])
        fp = hashlib.sha256(
            json.dumps(
                [PROMPT_VERSION, notes.get("fingerprint"), formulas, config.answer()["revision"]], default=str
            ).encode()
        ).hexdigest()[:24]
        row = con.execute("SELECT fingerprint, data FROM revision WHERE topic_id = ?", (topic_id,)).fetchone()
        if row and row[0] == fp and not redo:
            return json.loads(row[1])
        log(f"  revision: {notes['topic']}")
        with llm.track_usage() as usage:
            data = {
                "topic_id": topic_id,
                "topic": notes["topic"],
                "pages": notes["pages"],
                **(
                    _summary(notes["topic"], points)
                    if points
                    else {"keywords": [], "short_points": [], "comparisons": []}
                ),
                "formulas": _formula_table(notes["topic"], formulas),
                "page_ids": page_ids,
                "llm_calls": 0,
                "created": ledger.now(),
            }
        data["llm_calls"] = usage["calls"]
        con.execute(
            "INSERT OR REPLACE INTO revision (topic_id, fingerprint, data, created) VALUES (?,?,?,?)",
            (topic_id, fp, json.dumps(data, ensure_ascii=False), data["created"]),
        )
        con.commit()
        return data
    finally:
        if own:
            con.close()


def subject_revision(subject: str, wanted: list[str] | None = None, redo: bool = False, log: Log = print) -> list[dict]:
    subject = ledger.safe_subject(subject)
    ids = build.topic_ids(subject, wanted)
    con = ledger.connect(subject)
    try:
        return [topic_revision(subject, i, redo, log, con) for i in ids]
    finally:
        con.close()
