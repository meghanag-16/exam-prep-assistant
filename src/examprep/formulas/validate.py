"""Check that a formula's LaTeX is well formed before it is stored.

The check is structural and needs no LaTeX installation: delimiters stripped,
braces and \\left/\\right balanced, \\begin/\\end environments matched, and every
command known to KaTeX (the renderer the website uses). Broken LaTeX is marked
invalid; unknown commands or prose-like text are kept but marked for review.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Commands KaTeX renders (common subset). Anything else is flagged for review, not rejected.
KNOWN = set(
    """
    alpha beta gamma delta epsilon varepsilon zeta eta theta vartheta iota kappa lambda mu nu xi pi varpi rho
    varrho sigma varsigma tau upsilon phi varphi chi psi omega Gamma Delta Theta Lambda Xi Pi Sigma Upsilon Phi
    Psi Omega frac dfrac tfrac cfrac binom dbinom tbinom sqrt sum prod coprod int iint iiint oint lim limsup
    liminf sup inf max min arg argmax argmin det exp log ln lg sin cos tan cot sec csc arcsin arccos arctan sinh
    cosh tanh deg dim gcd hom ker Pr mod bmod pmod left right big Big bigg Bigg bigl bigr Bigl Bigr cdot cdots
    ldots dots vdots ddots times div pm mp ast star circ bullet oplus ominus otimes oslash odot cap cup sqcap
    sqcup vee wedge setminus wr diamond le leq ge geq neq ne approx equiv sim simeq cong propto ll gg subset
    supset subseteq supseteq nsubseteq in notin ni not forall exists nexists emptyset varnothing infty nabla
    partial prime hbar ell Re Im aleph to gets rightarrow leftarrow leftrightarrow Rightarrow Leftarrow
    Leftrightarrow implies impliedby iff mapsto longrightarrow longleftarrow uparrow downarrow mid parallel
    perp angle triangle square therefore because top bot vdash models land lor lnot neg bowtie Join ltimes
    rtimes uplus bigcup bigcap bigvee bigwedge bigoplus bigotimes bigodot biguplus bigsqcup hat widehat bar
    overline underline vec dot ddot tilde widetilde acute grave breve check overbrace underbrace overrightarrow
    overleftarrow text textbf textit textrm texttt mathrm mathbf mathit mathsf mathtt mathcal mathbb mathfrak
    mathscr boldsymbol operatorname displaystyle textstyle scriptstyle quad qquad space enspace thinspace begin
    end hline cline substack stackrel overset underset atop choose pmatrix bmatrix vmatrix matrix cases
    aligned array colon lfloor rfloor lceil rceil langle rangle lvert rvert lVert rVert vert Vert backslash
    lbrace rbrace lbrack rbrack sharp flat natural surd circledcirc leqslant geqslant nleq ngeq prec succ
    preceq succeq dashv ldotp cdotp iint oiint underleftarrow xrightarrow xleftarrow tag nonumber label
    phantom hphantom vphantom limits nolimits mathop mathbin mathrel mathord color textcolor boxed fbox
    leftharpoonup rightharpoonup rightleftharpoons hookrightarrow nearrow searrow sum_ smallint
    """.split()
)
ENVIRONMENTS = {
    "matrix", "pmatrix", "bmatrix", "Bmatrix", "vmatrix", "Vmatrix", "cases", "aligned", "align", "align*",
    "array", "gathered", "gather", "split", "equation", "equation*", "smallmatrix", "subarray", "rcases",
}  # fmt: skip
DELIMITERS = [("$$", "$$"), ("\\[", "\\]"), ("\\(", "\\)"), ("$", "$")]


@dataclass
class Check:
    latex: str  # cleaned LaTeX
    valid: bool  # well formed: will render
    issues: list[str] = field(default_factory=list)  # anything a person should look at

    @property
    def status(self) -> str:
        return "ok" if self.valid and not self.issues else "needs_review"


def clean(latex: str) -> str:
    t = " ".join(str(latex or "").split())
    # Models sometimes escape backslashes twice (\\begin, \\frac). A line break \\ followed straight
    # by the letters of a known command is never meant, so those become single backslashes.
    t = re.sub(r"\\\\([A-Za-z]+)", lambda m: "\\" + m.group(1) if m.group(1) in KNOWN else m.group(0), t)
    for a, b in DELIMITERS:
        if t.startswith(a) and t.endswith(b) and len(t) > len(a) + len(b):
            t = t[len(a) : -len(b)].strip()
            break
    return t


def normalise(latex: str) -> str:
    """Key used to spot the same formula on several pages."""
    return re.sub(r"\s+|\\[,;:! ]", "", clean(latex))


def check(latex: str) -> Check:
    raw = str(latex or "")
    t = clean(latex)
    errors: list[str] = []
    # A backspace or form feed is what is left of \begin or \frac after a JSON escape went wrong;
    # a tab inside a command (" imes", " heta") is \times or \theta lost the same way.
    if re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", raw) or re.search(r"\t(?=[a-z])", raw):
        errors.append("broken backslash command (control character in the LaTeX)")
    review: list[str] = []
    if not t:
        return Check(t, False, ["empty"])
    if len(t) > 2000:
        errors.append("longer than 2000 characters")
    # Braces, ignoring escaped \{ and \}.
    depth = 0
    for ch in re.sub(r"\\[{}]", "", t):
        depth += ch == "{"
        depth -= ch == "}"
        if depth < 0:
            break
    if depth != 0:
        errors.append("unbalanced braces")
    if len(re.findall(r"\\left\b", t)) != len(re.findall(r"\\right\b", t)):
        errors.append("\\left and \\right do not match")
    stack: list[str] = []
    for kind, env in re.findall(r"\\(begin|end)\{([^}]*)\}", t):
        if kind == "begin":
            stack.append(env)
            if env not in ENVIRONMENTS:
                review.append(f"environment '{env}' may not render")
        elif not stack or stack.pop() != env:
            errors.append(f"\\end{{{env}}} without matching \\begin")
    if stack:
        errors.append(f"\\begin{{{stack[-1]}}} is never closed")
    unknown = sorted({c for c in re.findall(r"\\([A-Za-z]+)", t) if c not in KNOWN})
    if unknown:
        review.append("unknown command(s): " + ", ".join("\\" + c for c in unknown[:5]))
    # Prose instead of maths: many long plain words and no operator.
    words = re.findall(r"(?<!\\)\b[A-Za-z]{4,}\b", re.sub(r"\\text\w*\{[^}]*\}", "", t))
    if len(words) >= 6 and not re.search(r"[=<>+\-*/^_]|\\(frac|sum|int|le|ge|neq|in|times|cdot)", t):
        review.append("looks like a sentence, not a formula")
    return Check(t, not errors, errors + review)
