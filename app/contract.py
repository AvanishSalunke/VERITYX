"""Phase B step 1: compile a question into a typed Evidence Contract."""
import json
from typing import Literal, Optional

import anthropic
from pydantic import BaseModel, Field, ValidationError

from . import config, llm
from .tracing import observe

ReqType = Literal["FACT_LOOKUP", "DEFINITION_CHECK", "NARRATIVE_EXPLANATION", "DETERMINISTIC_CALC"]


class Requirement(BaseModel):
    id: str
    type: ReqType
    description: str                       # what must be found, in plain words
    search_text: str                       # keyword-rich query for retrieval
    breadth: Literal["narrow", "medium", "broad"] = "medium"
    metric: Optional[str] = None
    entity: Optional[str] = None
    period: Optional[str] = None
    scope: Optional[str] = None
    unit: Optional[str] = None
    operation: Optional[str] = None        # e.g. absolute_difference, percent_change
    target_doc_ids: list[str] = Field(default_factory=list)


class EvidenceContract(BaseModel):
    target_doc_ids: list[str]
    requirements: list[Requirement]


SYS = ("You compile questions into evidence contracts for a document QA system. "
       "Use only the provided document manifest to choose documents. Respond only with JSON.")


def manifest_for(docs: list[dict]) -> str:
    rows = []
    for d in docs:
        p = d["profile"]
        rows.append({"doc_id": d["doc_id"], "file": d["filenames"][0], "type": p.get("doc_type"),
                     "entities": (p.get("entities") or [])[:5], "periods": p.get("reporting_periods"),
                     "scope": p.get("scope"), "summary": p.get("summary")})
    return json.dumps(rows, ensure_ascii=False)


@observe(name="compile_evidence_contract")
def compile_contract(question: str, docs: list[dict]) -> EvidenceContract:
    valid = {d["doc_id"] for d in docs}
    user = (
        f"Document manifest:\n{manifest_for(docs)}\n\nQuestion:\n{question}\n\n"
        "Return JSON: {\"target_doc_ids\": [doc_ids that could contain the answer; include every "
        "plausible document, and keep different years/versions separate], \"requirements\": "
        "[{\"id\":\"R1\",\"type\":\"FACT_LOOKUP|DEFINITION_CHECK|NARRATIVE_EXPLANATION|DETERMINISTIC_CALC\","
        "\"description\":\"...\",\"search_text\":\"keywords to search\",\"breadth\":\"narrow|medium|broad\","
        "\"metric\":null,\"entity\":null,\"period\":null,\"scope\":null,\"unit\":null,\"operation\":null,"
        "\"target_doc_ids\":[subset of the above]}]}. One requirement per distinct sub-question; "
        "use broad for questions needing many pages of synthesis, narrow for single facts."
    )
    for attempt in range(2):
        try:
            c = llm.call_claude(SYS, user, config.CONTRACT_MODEL, max_tokens=2000,
                                output_format=EvidenceContract)
            c.target_doc_ids = [d for d in c.target_doc_ids if d in valid]
            for r in c.requirements:
                r.target_doc_ids = [d for d in r.target_doc_ids if d in valid]
            if c.requirements:
                return c
        except (ValidationError, ValueError, RuntimeError, anthropic.APIError):
            continue
    # safe fallback: one broad requirement over the whole collection
    return EvidenceContract(target_doc_ids=[], requirements=[Requirement(
        id="R1", type="NARRATIVE_EXPLANATION", description=question, search_text=question, breadth="broad")])
