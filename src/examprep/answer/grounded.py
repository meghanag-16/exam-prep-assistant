"""Grounded answers: the model receives only the passages search kept for the question,
must cite them, and code checks the result before it is shown.

The flow is a LangGraph state graph:

  retrieve ──► syllabus ──(nothing relevant in the slides)──────────────► outside
                   │
                   ├──(extra explanations)──► trusted ──┐
                   ▼                                    ▼
                generate ◄──(a check failed, tries left)── check ──► self_check ──► finish
                   │                                                  (optional)
                   └──(the model says the passages do not answer)──► outside

- retrieve: search with the subject's cuts (reranker floor and relative cut, see
  search.py), so only passages that really match the question are given to the model.
  Nothing left means the question is not in the slides.
- syllabus: when a syllabus was loaded (examprep-topics --syllabus), the closest syllabus
  points are found by similarity and the model judges whether the question is covered;
  an answer to a question outside the syllabus says so.
- generate: a marks-based template from config/exam_pattern.yaml (word budget and
  structure) and an example answer from config/answer_examples.yaml.
- check (code, no model): every citation must name a passage that was given; every
  sentence or point that states something must carry a citation; the answer must use
  the key terms of the passages it cites (anti-generic check); the length must be near
  the word budget. A failed check sends the problems back to the model for another try.
- trusted (extra explanations only): passages from the allowlisted sites that pass the
  relevance gate (answer/trusted.py). The answer keeps "From your slides" and "Beyond
  your slides" apart; a point beyond the slides is kept only with a valid site link.
- self_check: the model compares its draft with the passages and removes contradictions
  and filler; the revision is used only if it still passes every check.
- outside: the answer starts with "This is not in your slides" (and "not in your syllabus"
  when that is known), then answers from the trusted sites with their links, or, when
  they have nothing either, from the model's general knowledge, labelled as unchecked
  (config/answer.yaml outside_answer).

Answers are cached in the ledger (table answers) while the index, the search settings
and the answer settings stay the same, so a repeated question costs nothing.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from typing import Any, TypedDict

import numpy as np

from examprep import config, llm, tracing
from examprep.answer import terms as termlib
from examprep.answer import trusted
from examprep.ingest import ledger
from examprep.search import search as searching

REFUSAL = "Not found in your course material."
NOT_IN_SLIDES = "This is not in your slides."
NOT_IN_SYLLABUS = "This is not in your syllabus."
PROMPT_VERSION = "answer-2"
LAYERS = ("intuition", "example", "common_mistakes", "exam_tips")
LAYER_TITLES = {
    "intuition": "Intuition",
    "example": "Example",
    "common_mistakes": "Common mistakes",
    "exam_tips": "How it is asked in exams",
}
_LABEL = re.compile(r"\[((?:\s*[SW]\d+\s*[,;]?)+)\]")
_ONE = re.compile(r"[SW]\d+")
_SPACED_LABEL = re.compile(r"[ \t]*" + _LABEL.pattern)

SCHEMA = """
CREATE TABLE IF NOT EXISTS answers (
    id        INTEGER PRIMARY KEY,
    key       TEXT UNIQUE NOT NULL,     -- question, marks, mode and settings (see _cache_key)
    question  TEXT NOT NULL,
    marks     INTEGER,
    deeper    INTEGER NOT NULL,
    result    TEXT NOT NULL,            -- JSON returned by answer()
    created   TEXT
);
CREATE TABLE IF NOT EXISTS feedback (
    id        INTEGER PRIMARY KEY,
    answer_id INTEGER REFERENCES answers(id) ON DELETE SET NULL,
    question  TEXT,
    rating    INTEGER NOT NULL,         -- 1 = thumbs up, -1 = thumbs down
    comment   TEXT,
    created   TEXT
);
"""


def migrate(con: sqlite3.Connection) -> None:
    con.executescript(SCHEMA)
    con.commit()


class State(TypedDict, total=False):
    question: str
    marks: int
    deeper: bool
    sources: list[dict]
    web: list[dict]
    web_info: dict
    reply: dict
    attempts: int
    problems: list[str]
    checked: dict
    final: dict
    refusal: str
    search_info: dict
    syllabus: dict
    outside: dict


# ------------------------------------------------------------ helpers


def marks_spec(marks: int) -> dict:
    table = config.exam_pattern().get("marks") or {}
    if not table:
        return {"word_budget": [100, 250], "structure": []}
    keys = sorted(int(k) for k in table)
    pick = min(keys, key=lambda k: (abs(k - marks), k))
    spec = dict(table.get(pick) or table.get(str(pick)) or {})
    spec["marks"] = pick
    return spec


def labels_in(text: str) -> list[str]:
    """Citation labels in order of first use: "[S1, S2] ... [W1]" -> ["S1", "S2", "W1"]."""
    out: list[str] = []
    for m in _LABEL.finditer(text or ""):
        for lab in _ONE.findall(m.group(1)):
            if lab not in out:
                out.append(lab)
    return out


def strip_labels(text: str) -> str:
    return re.sub(r"[ \t]+([.,;:])", r"\1", _SPACED_LABEL.sub("", text or "")).strip()


def _content_lines(text: str, min_words: int) -> list[str]:
    """Lines that state something: not headings, separators or short labels."""
    out = []
    for raw in (text or "").split("\n"):
        line = raw.strip()
        if not line or line.startswith("#") or set(line) <= set("|-: "):
            continue
        if len(strip_labels(line).split()) < min_words:
            continue
        out.append(line)
    return out


def _sources(hits: list[searching.Hit]) -> list[dict]:
    return [
        {
            "label": f"S{n}",
            "where": searching.where(h.pages),
            "pages": [{"file": p["file"], "page_no": p["page_no"], "page_id": p["page_id"]} for p in h.pages],
            "topic": h.topic,
            "heading": h.heading,
            "text": h.text,
            "rerank_score": None if h.rerank_score is None else round(h.rerank_score, 4),
        }
        for n, h in enumerate(hits, start=1)
    ]


def _material(sources: list[dict], web: list[dict]) -> str:
    parts = [f"[{s['label']}] ({s['where']})\n{s['text']}" for s in sources]
    parts += [f"[{w['label']}] ({w['site']}: {w['title']})\n{w['text']}" for w in web]
    return "<material>\n" + "\n\n".join(parts) + "\n</material>"


def _example(marks: int) -> str | None:
    """The example answer for this mark level, or for the nearest level that has one."""
    ex = {int(k): v for k, v in (config.answer_examples() or {}).items() if str(k).isdigit() and v}
    if not ex:
        return None
    pick = min(ex, key=lambda k: (abs(k - marks), k))
    return str(ex[pick]).strip()


def _max_tokens(marks: int) -> int:
    return 8192 if marks >= 8 else 4096


# ------------------------------------------------------------ the answerer


class Answerer:
    """Keeps the search index, models and term statistics open, so many questions are cheap."""

    def __init__(self, subject: str, searcher: searching.Searcher | None = None):
        self.subject = ledger.safe_subject(subject)
        self._own = searcher is None
        self.searcher = searcher or searching.Searcher(self.subject)
        self.con = self.searcher.con
        migrate(self.con)
        self.stats = termlib.TermStats(self.con)
        self.topic_names = [r[0] for r in self.con.execute("SELECT name FROM topics ORDER BY topic_no")]
        self.syllabus = [
            (r[0], r[1])
            for r in self.con.execute("SELECT unit, text FROM syllabus_points ORDER BY id")
            if r[1] and r[1].strip()
        ]
        self.cfg = config.answer()["answer"]
        self.graph = self._graph()

    def close(self) -> None:
        if self._own:
            self.searcher.close()

    # ---------------------------------------------------- graph nodes

    def _retrieve(self, s: State) -> State:
        passages = self.cfg.get("passages") or {}
        k = int(passages.get(s["marks"], passages.get(str(s["marks"]), 5)))
        hits, info = self.searcher.search(s["question"], k=k, cut=True)
        out: State = {"sources": _sources(hits), "search_info": info, "attempts": 0, "web": [], "web_info": {}}
        if not hits:
            out["refusal"] = "no passage of your material matched the question closely enough"
        return out

    def _syllabus(self, s: State) -> State:
        if not self.syllabus or not self.cfg.get("syllabus_check", True):
            return {"syllabus": {"checked": False}}
        from examprep.structure import embed

        texts = [f"{u}: {t}" if u else t for u, t in self.syllabus]
        a, b = embed.embed_pair(texts, [s["question"]])
        a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-9)
        b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-9)
        order = list(np.argsort(-(a @ b.T).ravel()))[: int(self.cfg.get("syllabus_candidates", 12))]
        listing = "\n".join(f"{n}. {texts[i]}" for n, i in enumerate(order, start=1))
        rules = [
            "Decide whether a university syllabus covers an exam question.",
            "The question is covered when the concept it asks about is one of the syllabus points or clearly "
            "part of one (a method, property, example or step of a listed topic).",
            'Reply with JSON only: {"covered": true, "point": 3} or {"covered": false, "point": null}',
        ]
        try:
            reply = llm.chat_json(
                [
                    {"role": "system", "content": llm.SYSTEM_GUARD + "\n\n" + "\n".join(f"- {r}" for r in rules)},
                    {"role": "user", "content": f"Question: {s['question']}\n\n<material>\n{listing}\n</material>"},
                ],
                max_tokens=1024,
                trace_name="judge-syllabus",
            )
        except llm.LLMError:
            return {"syllabus": {"checked": False, "error": "the syllabus check could not be run"}}
        if not isinstance(reply, dict) or not isinstance(reply.get("covered"), bool):
            return {"syllabus": {"checked": False}}
        point = None
        try:
            k = int(reply.get("point") or 0)
            point = texts[order[k - 1]] if 1 <= k <= len(order) else None
        except (TypeError, ValueError):
            point = None
        return {
            "syllabus": {"checked": True, "in_syllabus": reply["covered"], "point": point if reply["covered"] else None}
        }

    def _outside(self, s: State) -> State:
        """The slides do not cover the question: answer from the trusted sites, else from general knowledge."""
        mode = str(self.cfg.get("outside_answer", "trusted_then_general"))
        info: dict[str, Any] = {"source": None, "reason": s.get("refusal")}
        if mode == "none":
            return {"outside": info}
        res = trusted.passages(s["question"], self.subject, [], self.searcher.reranker())
        web = [p.as_dict() for p in res["kept"]]
        info["web_info"] = {
            "candidates": res["candidates"],
            "rejected": res["rejected"],
            "kept": len(web),
            "provider": res.get("provider"),
            **({"error": res["error"]} if res["error"] else {}),
        }
        spec = marks_spec(s["marks"])
        lo, hi = (spec.get("word_budget") or [100, 250])[:2]
        shape = f"This is a {spec['marks']}-mark question: write {lo} to {hi} words in Markdown, structured as: "
        shape += ", ".join(spec.get("structure") or []) + "."
        if web:
            rules = [
                "The student's course slides do not cover this question. Answer it using ONLY the passages from "
                "trusted sites in <material>.",
                "Cite the passage label in square brackets, such as [W1], at the end of every sentence or point "
                "that states a fact. Use only the labels given.",
                shape,
                'If the passages do not answer the question, reply {"found": false}.',
                'Reply with JSON only: {"found": true, "answer": "<Markdown with [W] citations>"}',
            ]
            user = f"Question: {s['question']}\n\n{_material([], web)}"
            for _ in range(2):
                try:
                    reply = llm.chat_json(
                        [
                            {
                                "role": "system",
                                "content": llm.SYSTEM_GUARD + "\n\n" + "\n".join(f"- {r}" for r in rules),
                            },
                            {"role": "user", "content": user},
                        ],
                        temperature=0.2,
                        max_tokens=_max_tokens(spec["marks"]),
                        trace_name="write-trusted-answer",
                    )
                except llm.LLMError:
                    break
                answer = str(reply.get("answer") or "") if isinstance(reply, dict) else ""
                if not (isinstance(reply, dict) and reply.get("found", True) and answer.strip()):
                    break
                labels = {w["label"] for w in web}
                used = labels_in(answer)
                if used and all(x in labels for x in used):
                    info.update(source="trusted", answer=answer, web=web, used=used)
                    return {"outside": info}
                user += "\n\nYour last answer cited labels that were not given or none at all. Cite only " + ", ".join(
                    sorted(labels)
                )
        if mode != "trusted_then_general":
            info["web"] = web
            return {"outside": info}
        rules = [
            "The student's course slides and the trusted sites do not cover this question. Answer it from your "
            "general knowledge, accurately and plainly. Say so briefly if the question is unclear.",
            shape,
            'Reply with JSON only: {"answer": "<Markdown answer>"}',
        ]
        try:
            reply = llm.chat_json(
                [
                    {"role": "system", "content": "\n".join(f"- {r}" for r in rules)},
                    {"role": "user", "content": f"Question: {s['question']}"},
                ],
                temperature=0.2,
                max_tokens=_max_tokens(spec["marks"]),
                trace_name="write-general-answer",
            )
        except llm.LLMError as e:
            info["error"] = str(e)
            return {"outside": info}
        answer = str(reply.get("answer") or "") if isinstance(reply, dict) else ""
        if answer.strip():
            info.update(source="general", answer=strip_labels(answer))
        return {"outside": info}

    def _trusted(self, s: State) -> State:
        res = trusted.passages(s["question"], self.subject, [x["text"] for x in s["sources"]], self.searcher.reranker())
        info = {"candidates": res["candidates"], "rejected": res["rejected"], "kept": len(res["kept"])}
        if res["error"]:
            info["error"] = res["error"]
        return {"web": [p.as_dict() for p in res["kept"]], "web_info": info}

    def _messages(self, s: State) -> list[dict]:
        spec = marks_spec(s["marks"])
        lo, hi = (spec.get("word_budget") or [100, 250])[:2]
        structure = ", ".join(spec.get("structure") or [])
        rules = [
            "Answer the student's exam question using ONLY the passages in <material>. Do not add facts from "
            "your own knowledge, even if you know them.",
            "Cite the passage label in square brackets, such as [S2], at the end of every sentence, point or "
            "table row that states a fact. Use only the labels given. Never invent a label or a page number.",
            f"This is a {spec['marks']}-mark question: write {lo} to {hi} words in Markdown, structured as: "
            f"{structure}. Skip a part of the structure the passages say nothing about.",
            "Use the technical terms of the passages.",
            'If the passages do not answer the question, reply {"found": false}.',
        ]
        shape = '{"found": true, "answer": "<Markdown answer with [S] citations>"}'
        if s.get("deeper"):
            rules += [
                "The answer field uses only the [S] passages (the student's slides).",
                "Then add extra explanation points under layers: intuition, example (worked or real-world), "
                "common_mistakes and exam_tips. Each point lists the labels that support it in cite: [S] labels "
                "for points the slides support, [W] labels for points from the trusted sites. A point that no "
                "label supports must be left out.",
                "related_topics: up to 4 names copied exactly from <topics> that are closely related.",
            ]
            shape = (
                '{"found": true, "answer": "<Markdown with [S] citations>", "layers": {"intuition": '
                '[{"text": "...", "cite": ["S1"]}], "example": [...], "common_mistakes": [...], '
                '"exam_tips": [...]}, "related_topics": ["..."]}'
            )
        rules.append("Reply with JSON only, in this shape: " + shape)
        system = llm.SYSTEM_GUARD + "\n\n" + "\n".join(f"- {r}" for r in rules)
        example = _example(spec["marks"])
        if example:
            system += f"\n\nExample of the expected shape and citation style (a different question):\n{example}"
        user = f"Question: {s['question']}\n\n{_material(s['sources'], s.get('web') or [])}"
        if s.get("deeper") and self.topic_names:
            user += "\n\n<topics>\n" + "\n".join(self.topic_names) + "\n</topics>"
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def _generate(self, s: State) -> State:
        messages = self._messages(s)
        if s.get("problems") and s.get("reply"):
            messages += [
                {"role": "assistant", "content": json.dumps(s["reply"], ensure_ascii=False)},
                {
                    "role": "user",
                    "content": "Your answer failed these checks. Fix them and reply with the full JSON again:\n- "
                    + "\n- ".join(s["problems"]),
                },
            ]
        try:
            reply = llm.chat_json(
                messages, temperature=0.2, max_tokens=_max_tokens(s["marks"]), trace_name="write-answer"
            )
        except llm.LLMError as e:
            return {"refusal": f"the answer could not be written ({e})", "attempts": s.get("attempts", 0) + 1}
        if not isinstance(reply, dict):
            reply = {"found": False}
        out: State = {"reply": reply, "attempts": s.get("attempts", 0) + 1}
        if not reply.get("found", True) or not str(reply.get("answer") or "").strip():
            out["refusal"] = "the passages found do not answer the question"
        return out

    def check(self, reply: dict, sources: list[dict], web: list[dict], marks: int) -> dict:
        """Code checks of one reply. Returns {"problems": [...], "citations_valid", ...}."""
        answer = str(reply.get("answer") or "")
        src = {x["label"]: x for x in sources}
        webl = {x["label"] for x in web}
        used = labels_in(answer)
        problems = []
        bad = [lab for lab in used if lab not in src]
        if bad:
            problems.append(
                f"The answer cites {', '.join(bad)}, which "
                + ("are web passages or were not given" if any(b.startswith("W") for b in bad) else "were not given")
                + f". Cite only {', '.join(src)} in the answer."
            )
        min_words = int(self.cfg.get("min_line_words", 6))
        uncited = [line for line in _content_lines(answer, min_words) if not labels_in(line)]
        if uncited:
            problems.append(
                "These lines state facts without a citation: "
                + " | ".join(strip_labels(u)[:80] for u in uncited[:3])
                + ". Add the label of the passage that says it, or remove the line."
            )
        cited_text = "\n".join(src[lab]["text"] for lab in used if lab in src) or "\n".join(x["text"] for x in sources)
        key = self.stats.key_terms(cited_text, int(self.cfg.get("key_terms", 10)))
        plain = strip_labels(answer)
        share = termlib.share_present(key, plain)
        if share < float(self.cfg.get("min_term_share", 0.3)):
            missing = [t for t in key if t not in termlib.present(key, plain)]
            problems.append(
                "The answer is too generic: it does not use the words of the passages. Use terms such as: "
                + ", ".join(missing[:6])
                + "."
            )
        spec = marks_spec(marks)
        lo, hi = (spec.get("word_budget") or [0, 10**6])[:2]
        tol = float(self.cfg.get("length_tolerance", 0.5))
        words = len(plain.split())
        if words < lo * (1 - tol) or words > hi * (1 + tol):
            problems.append(f"The answer has {words} words; a {spec['marks']}-mark answer needs {lo} to {hi}.")
        layers, beyond_total, beyond_valid, dropped = self._layers(reply, src, webl)
        return {
            "problems": problems,
            "citations_valid": not bad and not uncited,
            "uncited_lines": len(uncited),
            "invalid_labels": bad,
            "key_terms": key,
            "term_share": round(share, 3),
            "words": words,
            "layers": layers,
            "beyond_points": beyond_total,
            "beyond_with_valid_link": beyond_valid,
            "dropped_points": dropped,
        }

    def _layers(self, reply: dict, src: dict, webl: set[str]) -> tuple[dict, int, int, int]:
        """Extra explanation points split into from-slides and beyond-slides, keeping only points
        with a valid citation. Returns (layers, beyond points, beyond with a valid link, dropped)."""
        raw = reply.get("layers")
        if not isinstance(raw, dict):
            raw = {}
        out: dict[str, list[dict]] = {}
        beyond_total = beyond_valid = dropped = 0
        for name in LAYERS:
            for item in raw.get(name) or []:
                if isinstance(item, str):
                    item = {"text": item, "cite": labels_in(item)}
                if not isinstance(item, dict) or not str(item.get("text") or "").strip():
                    continue
                cite = [c for c in (item.get("cite") or []) if isinstance(c, str)] + labels_in(str(item["text"]))
                cite = list(dict.fromkeys(c.strip() for c in cite))
                s_ok = [c for c in cite if c in src]
                w_ok = [c for c in cite if c in webl]
                is_beyond = not s_ok
                if is_beyond:
                    beyond_total += 1
                    beyond_valid += bool(w_ok)
                if not s_ok and not w_ok:
                    dropped += 1
                    continue
                out.setdefault(name, []).append(
                    {"text": strip_labels(str(item["text"])), "slides": s_ok, "web": w_ok, "beyond": is_beyond}
                )
        return out, beyond_total, beyond_valid, dropped

    def _check(self, s: State) -> State:
        return {"checked": self.check(s["reply"], s["sources"], s.get("web") or [], s["marks"]), "problems": []}

    def _after_check(self, s: State) -> str:
        if s["checked"]["problems"] and s.get("attempts", 0) <= int(self.cfg.get("max_attempts", 2)):
            return "retry"
        mode = str(self.cfg.get("self_check", "deeper"))
        if mode == "always" or (mode == "deeper" and s.get("deeper")):
            return "self_check"
        return "finish"

    def _retry(self, s: State) -> State:
        return {"problems": s["checked"]["problems"]}

    def _self_check(self, s: State) -> State:
        messages = self._messages(s) + [
            {"role": "assistant", "content": json.dumps(s["reply"], ensure_ascii=False)},
            {
                "role": "user",
                "content": "Check your draft against the passages. Remove any statement that a cited passage does "
                "not support or that contradicts the slides ([S] passages), remove filler and repetition, and "
                "keep every citation on the statements that remain. Reply with the full JSON in the same shape.",
            },
        ]
        try:
            revised = llm.chat_json(
                messages, temperature=0.1, max_tokens=_max_tokens(s["marks"]), trace_name="self-check-answer"
            )
        except llm.LLMError:
            return {}
        if not isinstance(revised, dict) or not str(revised.get("answer") or "").strip():
            return {}
        checked = self.check(revised, s["sources"], s.get("web") or [], s["marks"])
        if checked["problems"] and not s["checked"]["problems"]:
            return {}  # the revision broke a check that the draft passed: keep the draft
        return {"reply": revised, "checked": {**checked, "self_checked": True}}

    def _finish(self, s: State) -> State:
        reply, checked = s["reply"], s["checked"]
        used = [lab for lab in labels_in(str(reply.get("answer") or "")) if lab.startswith("S")]
        sources = [x for x in s["sources"] if x["label"] in used] or []
        number = {x["label"]: n for n, x in enumerate(sources, start=1)}

        def show(text: str) -> str:
            def sub(m: re.Match) -> str:
                nums = [number[lab] for lab in _ONE.findall(m.group(1)) if lab in number]
                return "[" + ", ".join(str(n) for n in sorted(set(nums))) + "]" if nums else ""

            return re.sub(r"\s+([.,;:])", r"\1", _LABEL.sub(sub, text)).strip()

        web_by = {w["label"]: w for w in s.get("web") or []}
        layers = {}
        for name, items in (checked.get("layers") or {}).items():
            layers[name] = [
                {
                    "text": it["text"],
                    "beyond": it["beyond"],
                    "slides": [x["where"] for x in s["sources"] if x["label"] in it["slides"]],
                    "links": [{"site": web_by[w]["site"], "url": web_by[w]["url"]} for w in it["web"]],
                }
                for it in items
            ]
        related = [t for t in (reply.get("related_topics") or []) if isinstance(t, str) and t in self.topic_names]
        syl = s.get("syllabus") or {}
        return {
            "final": {
                "in_slides": True,
                "in_syllabus": syl.get("in_syllabus") if syl.get("checked") else None,
                "syllabus_point": syl.get("point"),
                "notice": NOT_IN_SYLLABUS if syl.get("checked") and not syl.get("in_syllabus") else None,
                "outside_source": None,
                "answer": show(str(reply.get("answer") or "")),
                "answer_with_labels": str(reply.get("answer") or ""),
                "sources": [{"n": number[x["label"]], **x} for x in sources],
                "retrieved": s["sources"],
                "layers": layers,
                "related_topics": related[:4],
                "web": s.get("web") or [],
                "web_info": s.get("web_info") or {},
                "checks": {k: v for k, v in checked.items() if k != "layers"},
                "attempts": s.get("attempts", 1),
            }
        }

    def _finish_outside(self, s: State) -> State:
        out = s.get("outside") or {}
        syl = s.get("syllabus") or {}
        notice = NOT_IN_SLIDES
        if syl.get("checked") and not syl.get("in_syllabus"):
            notice += " " + NOT_IN_SYLLABUS
        source = out.get("source")
        web = out.get("web") or []
        sources: list[dict] = []
        answer = REFUSAL
        if source == "trusted":
            used = [w for w in web if w["label"] in out.get("used", [])]
            number = {w["label"]: n for n, w in enumerate(used, start=1)}

            def sub(m: re.Match) -> str:
                nums = [number[lab] for lab in _ONE.findall(m.group(1)) if lab in number]
                return "[" + ", ".join(str(n) for n in sorted(set(nums))) + "]" if nums else ""

            answer = re.sub(r"\s+([.,;:])", r"\1", _LABEL.sub(sub, out["answer"])).strip()
            sources = [{"n": number[w["label"]], **w} for w in used]
            notice += " This answer is from trusted sites:"
        elif source == "general":
            answer = out["answer"]
            notice += " This answer is from general knowledge, not from your slides or a trusted site, so check it:"
        return {
            "final": {
                "in_slides": False,
                "in_syllabus": syl.get("in_syllabus") if syl.get("checked") else None,
                "syllabus_point": syl.get("point"),
                "notice": notice,
                "outside_source": source,
                "reason": out.get("reason") or s.get("refusal"),
                "answer": answer,
                "sources": sources,
                "retrieved": s.get("sources") or [],
                "layers": {},
                "related_topics": [],
                "web": web,
                "web_info": out.get("web_info") or {},
                "checks": {},
                "attempts": s.get("attempts", 0),
            }
        }

    # Graph steps shown as their own observations in the trace (name, type, what to show as
    # output). Steps not listed are already visible through what they contain: the search
    # (retrieve), the LLM generations (generate, self_check) or nothing worth a span.
    TRACED_STEPS: dict[str, tuple[str, str, Any]] = {
        "syllabus": ("check-syllabus", "evaluator", lambda out: out.get("syllabus")),
        "trusted": ("add-trusted-passages", "retriever", lambda out: out.get("web_info")),
        "check": (
            "check-answer",
            "guardrail",
            lambda out: {
                k: (out.get("checked") or {}).get(k)
                for k in ("problems", "citations_valid", "uncited", "term_share", "words")
            },
        ),
        "outside": (
            "answer-outside-slides",
            "chain",
            lambda out: {k: v for k, v in (out.get("outside") or {}).items() if k in ("source", "answer")},
        ),
    }

    def _traced(self, step: str, fn):
        spec = self.TRACED_STEPS.get(step)
        if spec is None:
            return fn
        name, as_type, show = spec

        def run(s: State) -> State:
            with tracing.observe(name, as_type=as_type) as obs:
                out = fn(s)
                if obs is not None:
                    tracing.update(obs, output=show(out))
            return out

        return run

    def _graph(self):
        from langgraph.graph import END, START, StateGraph

        g = StateGraph(State)
        for name, fn in (
            ("retrieve", self._retrieve),
            ("syllabus", self._syllabus),
            ("trusted", self._trusted),
            ("generate", self._generate),
            ("check", self._check),
            ("retry", self._retry),
            ("self_check", self._self_check),
            ("finish", self._finish),
            ("outside", self._outside),
            ("finish_outside", self._finish_outside),
        ):
            g.add_node(name, self._traced(name, fn))
        g.add_edge(START, "retrieve")
        g.add_edge("retrieve", "syllabus")
        g.add_conditional_edges(
            "syllabus",
            lambda s: "outside" if s.get("refusal") else ("trusted" if s.get("deeper") else "generate"),
            {"outside": "outside", "trusted": "trusted", "generate": "generate"},
        )
        g.add_edge("trusted", "generate")
        g.add_conditional_edges(
            "generate", lambda s: "outside" if s.get("refusal") else "check", {"outside": "outside", "check": "check"}
        )
        g.add_conditional_edges(
            "check", self._after_check, {"retry": "retry", "self_check": "self_check", "finish": "finish"}
        )
        g.add_edge("retry", "generate")
        g.add_edge("self_check", "finish")
        g.add_edge("outside", "finish_outside")
        g.add_edge("finish", END)
        g.add_edge("finish_outside", END)
        return g.compile()

    # ---------------------------------------------------- entry point

    def _cache_key(self, question: str, marks: int, deeper: bool) -> str:
        settings = searching.settings_path(self.subject)
        parts = [
            " ".join(question.lower().split()),
            marks,
            deeper,
            PROMPT_VERSION,
            self.searcher.meta.get("fingerprint"),
            settings.read_text(encoding="utf-8") if settings.exists() else "",
            json.dumps([config.answer(), config.exam_pattern(), config.answer_examples()], sort_keys=True, default=str),
            json.dumps(config.trusted_sites(), sort_keys=True, default=str),
            self.syllabus,
        ]
        return hashlib.sha256(json.dumps(parts, default=str).encode()).hexdigest()

    def answer(
        self,
        question: str,
        marks: int | None = None,
        deeper: bool = False,
        use_cache: bool = True,
        session_id: str | None = None,
        tags: list[str] | None = None,
    ) -> dict:
        """One answer, traced as `answer-question` (session_id groups a chat's questions;
        tags name the feature, "ask" by default)."""
        question = question.strip()
        if not question:
            raise ValueError("the question is empty")
        marks = marks_spec(int(marks or self.cfg.get("default_marks", 5)))["marks"]
        with tracing.trace(
            "answer-question",
            input=question,
            session_id=session_id,
            tags=tags or ["ask"],
            metadata={"subject": self.subject, "marks": marks, "deeper": deeper},
            version=PROMPT_VERSION,
        ) as root:
            result = self._answer(question, marks, deeper, use_cache, root)
            tracing.update(
                root,
                output="\n\n".join(x for x in (result.get("notice"), result["answer"]) if x),
                metadata={
                    "cache_hit": result["cached"],
                    "answer_id": result.get("id"),
                    "in_slides": result.get("in_slides"),
                    "in_syllabus": result.get("in_syllabus"),
                    "outside_source": result.get("outside_source"),
                    "sources": [s.get("where") for s in result.get("sources") or []],
                },
            )
            if root is not None and not result["cached"]:
                tracing.score("in-slides", 1 if result["in_slides"] else 0, data_type="BOOLEAN")
                if result["in_slides"]:
                    passed = not (result.get("checks") or {}).get("problems")
                    tracing.score("checks-passed", 1 if passed else 0, data_type="BOOLEAN")
                    tracing.score("attempts", float(result.get("attempts") or 0), data_type="NUMERIC")
        return result

    def _answer(self, question: str, marks: int, deeper: bool, use_cache: bool, root: Any) -> dict:
        cache = use_cache and bool(self.cfg.get("cache", True))
        key = self._cache_key(question, marks, deeper)
        if cache:
            row = self.con.execute("SELECT id, result FROM answers WHERE key = ?", (key,)).fetchone()
            if row:
                return {**json.loads(row[1]), "id": row[0], "cached": True}
        started = time.perf_counter()
        with llm.track_usage() as usage:
            state = self.graph.invoke({"question": question, "marks": marks, "deeper": deeper})
        result: dict[str, Any] = {
            "question": question,
            "marks": marks,
            "deeper": deeper,
            **state["final"],
            "search": state.get("search_info") or {},
            "ms": round((time.perf_counter() - started) * 1000, 1),
            "trace_id": tracing.trace_id(root),
            "tokens": {
                "prompt": usage["prompt_tokens"],
                "completion": usage["completion_tokens"],
                "calls": usage["calls"],
                "cached_calls": usage["cached"],
                "models": usage["models"],
            },
        }
        cur = self.con.execute(
            "INSERT OR REPLACE INTO answers (key, question, marks, deeper, result, created) VALUES (?,?,?,?,?,?)",
            (key, question, result["marks"], int(deeper), json.dumps(result, ensure_ascii=False), ledger.now()),
        )
        self.con.commit()
        return {**result, "id": int(cur.lastrowid or 0), "cached": False}


def answer(
    subject: str,
    question: str,
    marks: int | None = None,
    deeper: bool = False,
    use_cache: bool = True,
    session_id: str | None = None,
):
    a = Answerer(subject)
    try:
        return a.answer(question, marks, deeper, use_cache, session_id=session_id)
    finally:
        a.close()


def feedback(subject: str, answer_id: int | None, rating: int, comment: str | None = None) -> int:
    """Log a thumbs up (1) or down (-1) for an answer. Returns the feedback id."""
    if rating not in (1, -1):
        raise ValueError("rating must be 1 (thumbs up) or -1 (thumbs down)")
    con = ledger.connect(ledger.safe_subject(subject))
    try:
        migrate(con)
        question = trace = None
        if answer_id is not None:
            row = con.execute("SELECT question, result FROM answers WHERE id = ?", (answer_id,)).fetchone()
            if row is None:
                raise ValueError(f"no answer with id {answer_id}")
            question = row[0]
            try:
                trace = json.loads(row[1]).get("trace_id")
            except (TypeError, ValueError):
                trace = None
        cur = con.execute(
            "INSERT INTO feedback (answer_id, question, rating, comment, created) VALUES (?,?,?,?,?)",
            (answer_id, question, rating, (comment or "")[:2000] or None, ledger.now()),
        )
        con.commit()
        if trace:  # the thumbs up or down also shows on the answer's trace in Langfuse
            tracing.score("user-feedback", float(rating), trace_id=trace, data_type="NUMERIC", comment=comment)
        return int(cur.lastrowid or 0)
    finally:
        con.close()


def to_markdown(result: dict) -> str:
    """An answer as Markdown: notices (not in the slides or syllabus), the answer, its sources,
    and extra explanations with the slides and the trusted sites kept apart."""
    lines = [f"## {result['question']}", "", f"*{result['marks']}-mark answer*", ""]
    if result.get("notice"):
        lines += [f"**{result['notice']}**", ""]
    lines += [result["answer"], ""]
    if not result.get("in_slides", True):
        if result.get("outside_source") == "trusted":
            lines += ["**Trusted sources**", ""]
            lines += [f"{w['n']}. [{w['title'] or w['site']}]({w['url']})" for w in result["sources"]]
        return "\n".join(lines).rstrip() + "\n"
    lines += ["**From your slides**", ""]
    lines += [f"{s['n']}. {s['where']}" + (f" ({s['topic']})" if s.get("topic") else "") for s in result["sources"]]
    slides = [(n, it) for n, items in result.get("layers", {}).items() for it in items if not it["beyond"]]
    beyond = [(n, it) for n, items in result.get("layers", {}).items() for it in items if it["beyond"]]
    if slides:
        lines += ["", "### More from your slides", ""]
        lines += [f"- **{LAYER_TITLES[n]}:** {it['text']} ({'; '.join(it['slides'])})" for n, it in slides]
    if beyond:
        lines += ["", "### Beyond your slides", ""]
        for n, it in beyond:
            links = ", ".join(f"[{x['site']}]({x['url']})" for x in it["links"])
            lines.append(f"- **{LAYER_TITLES[n]}:** {it['text']} ({links})")
    if result.get("related_topics"):
        lines += ["", "**Related topics:** " + ", ".join(result["related_topics"])]
    return "\n".join(lines) + "\n"
