"""Grounded answer generation over wide retrieval, with chunk-id citations.

The model sees numbered evidence [C1]..[Cn] and the planner's aspects as a coverage
checklist, and must cite chunk ids for every claim. Cited ids become the returned context
(so context precision reflects evidence actually used), and the markers are rewritten to
human-readable [document p.X] citations.
"""
import re

from . import llm

MAX_CHUNK_CHARS = 1500

ANSWER_SYS = """You are a financial research analyst answering from a document collection.
You are given numbered evidence chunks and an aspect checklist. Your answer is graded point by
point against an expert's reference answer: every point the expert made that you cover earns
credit, and every point you add that the expert would not have made counts AGAINST you. So
write what a senior analyst would put in a tight briefing - the essential facts, not
everything the evidence happens to contain.

Rules:
- Use ONLY the evidence. No outside knowledge, no assumptions, no generic commentary.
- Answer exactly what was asked. Skip background, definitions, boilerplate, contact or
  governance details, and secondary sub-figures unless the question asks for them.
- Prefer headline facts: the key figures (exact, with unit, currency, period and
  standalone/consolidated scope as stated), the key events, the key rules and their effect.
- Every factual sentence or bullet ends with the ids of supporting chunks, e.g. [C3] or
  [C3][C7]. Never cite a chunk that does not state the fact.
- Keep near-duplicates apart: never mix figures from different years, entities or scopes.
- If the evidence does not address part of the question, say so in one short sentence.
- If the evidence does not answer the question at all, the whole answer is one sentence saying
  the documents do not contain this information, with no citations and an empty "relevant"
  list. Do not describe unrelated evidence.
- Length follows the question: a narrow factual question 80-250 words; a broad review or
  multi-part question 450-850 words. Never exceed 900 words.
- Format: "## Short answer" (2-4 sentences), then 2-5 "## <heading>" sections of concise
  bullets, one fact per bullet. No preamble, no closing summary, no advice beyond the evidence.

Respond only with JSON:
{"answer": "<markdown answer with [C#] citations>",
 "relevant": [ids of ALL chunks that contain information relevant to the question,
              whether or not you cited them]}"""

_CITE = re.compile(r"\[C(\d+)\]")
_CITE_GROUP = re.compile(r"\[\s*C\d+(?:\s*[,;]\s*C?\d+)+\s*\]")


def _evidence(chunks):
    lines = []
    for i, c in enumerate(chunks, start=1):
        text = c["text"][:MAX_CHUNK_CHARS]
        sec = f" | {c['section'][:80]}" if c.get("section") else ""
        lines.append(f"[C{i}] {c['doc']} | p.{c['page']}{sec}\n{text}")
    return "\n\n".join(lines)


def _split_groups(text):
    """[C3, C5] -> [C3][C5] so one regex finds every citation."""
    return _CITE_GROUP.sub(
        lambda m: "".join(f"[C{n}]" for n in re.findall(r"\d+", m.group(0))), text)


_ABSENCE = re.compile(r"\b(not (stated|disclosed|addressed|described|provided|specified|given|"
                      r"available|covered)|does not|do not|no (information|disclosure|detail))\b",
                      re.I)
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


def _supported(text, n_chunks):
    return any(1 <= int(n) <= n_chunks for n in _CITE.findall(text))


def verify(raw, n_chunks):
    """Drop claims that carry no valid chunk citation: whole bullets, or single sentences in
    prose paragraphs. Headings and explicit 'the evidence does not state X' lines are kept.
    Returns (text, dropped_claims)."""
    kept, dropped = [], []
    for line in raw.splitlines():
        s = line.strip()
        if not s or s.startswith("#") or _supported(s, n_chunks) and s[:2] in ("- ", "* "):
            kept.append(line)
            continue
        if s[:2] in ("- ", "* ") or re.match(r"^\d+[.)] ", s):
            if _ABSENCE.search(s):
                kept.append(line)
            else:
                dropped.append(s)
            continue
        out = []
        for sent in _SENTENCE.split(s):
            if _supported(sent, n_chunks) or _ABSENCE.search(sent):
                out.append(sent)
            else:
                dropped.append(sent)
        if out:
            kept.append(" ".join(out))
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip(), dropped


def _label(chunk):
    name = re.sub(r"[_\s]+", " ", chunk["doc"].removesuffix(".pdf")).strip()
    return f"[{name[:40].strip()} p.{chunk['page']}]"


def generate(query, chunks, plan=None, max_tokens=6000, reasoning_effort="none"):
    """Return (answer_text, cited_chunks, relevant_chunks, meta)."""
    checklist = "\n".join(f"- {a['aspect']}" for a in (plan or {}).get("aspects", [])) or "-"
    periods = ", ".join((plan or {}).get("periods", [])) or "-"
    user = (f"Question:\n{query}\n\nTarget periods: {periods}\nAspect checklist:\n{checklist}\n\n"
            f"Evidence:\n{_evidence(chunks)}")
    out, meta = llm.call_json(ANSWER_SYS, user, max_tokens=max_tokens,
                              reasoning_effort=reasoning_effort, name="llm_answer_generation")
    raw, dropped = verify(_split_groups(str(out.get("answer", ""))), len(chunks))

    cited_ids = []
    for n in _CITE.findall(raw):
        i = int(n)
        if 1 <= i <= len(chunks) and i not in cited_ids:
            cited_ids.append(i)
    relevant_ids = []
    for r in out.get("relevant") or []:
        try:
            i = int(str(r).strip().lstrip("[Cc").rstrip("]"))
        except ValueError:
            continue
        if 1 <= i <= len(chunks) and i not in cited_ids and i not in relevant_ids:
            relevant_ids.append(i)

    def relabel(m):
        i = int(m.group(1))
        return _label(chunks[i - 1]) if 1 <= i <= len(chunks) else ""
    text = _CITE.sub(relabel, raw)
    text = re.sub(r"(\[[^\[\]]+ p\.\d+\])(?:\1)+", r"\1", text)  # collapse repeated labels

    meta = {**meta, "n_evidence": len(chunks), "n_cited": len(cited_ids),
            "n_relevant_uncited": len(relevant_ids), "n_dropped_unsupported": len(dropped),
            "dropped_unsupported": dropped}
    return (text, [chunks[i - 1] for i in cited_ids], [chunks[i - 1] for i in relevant_ids], meta)
