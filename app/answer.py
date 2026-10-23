"""Phase B step 4: proof-linked answer synthesis from the contract and selected chunks."""
from . import config, llm
from .contract import EvidenceContract
from .tracing import observe

SYS = (
    "You answer questions strictly from the numbered evidence passages provided. Rules: "
    "(1) Every factual statement must be supported by a passage; cite it inline as [Document p.N] "
    "using the passage's document name and page. (2) Never use outside knowledge. "
    "(3) Address every requirement; if the passages do not contain what a requirement needs, say "
    "explicitly that the documents do not provide it - do not guess or fill gaps. "
    "(4) If passages disagree, report both values with their citations. (5) Keep numbers, units, "
    "periods and scope (e.g. consolidated vs standalone) exactly as written. "
    "(6) Be concise: answer exactly what was asked, lead with the direct answer, and include only "
    "points the question needs. No preamble, background, restating the question, or closing summary. "
    "Use a short list only when the question asks for several items. "
    "Never mention requirement IDs (R1, R2...). Output only the answer text in markdown."
)


@observe(name="generate_grounded_answer")
def generate(question: str, contract: EvidenceContract, chunks: list[dict]) -> str:
    if not chunks:
        return "The indexed documents do not contain information relevant to this question."
    reqs = "\n".join(f"- {r.id} ({r.type}, {r.breadth}): {r.description}" for r in contract.requirements)
    ev = "\n\n".join(f"[{c['document']} p.{c['page']}] {c['text']}" for c in chunks)
    user = f"Question:\n{question}\n\nRequirements to address:\n{reqs}\n\nEvidence passages:\n{ev}"
    return llm.call_claude(SYS, user, config.ANSWER_MODEL, max_tokens=2500)
