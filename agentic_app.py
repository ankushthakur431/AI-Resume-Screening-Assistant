from __future__ import annotations

import json
import os
import re
import tempfile
import textwrap
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
except Exception:  # pragma: no cover - optional dependency path
    ChatOpenAI = None
    OpenAIEmbeddings = None

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
except Exception:  # pragma: no cover
    canvas = None
    letter = None


class SimpleEmbeddings(Embeddings):
    """A lightweight deterministic embedding model used when no OpenAI key is available."""

    def _embed(self, text: str) -> List[float]:
        tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
        if not tokens:
            return [0.0] * 64
        dims = 64
        vector = [0.0] * dims
        for token in set(tokens):
            index = abs(hash(token)) % dims
            vector[index] += 1.0
        norm = sum(v * v for v in vector) ** 0.5 or 1.0
        return [v / norm for v in vector]

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)


class SimpleLLM:
    """A deterministic fallback LLM for demo use when OpenAI is unavailable."""

    def invoke(self, prompt: str) -> str:
        text = prompt if isinstance(prompt, str) else str(prompt)
        if "resume" in text.lower() and "job description" in text.lower():
            return self._resume_response(text)
        return self._claim_response(text)

    def _resume_response(self, text: str) -> str:
        jd = re.search(r"Job Description:\n(.*?)(\n\nRetrieved Resume Context|$)", text, re.S)
        context = re.search(r"Retrieved Resume Context:\n(.*)$", text, re.S)
        jd_text = jd.group(1).strip() if jd else ""
        context_text = context.group(1).strip() if context else text
        relevant_terms = [term for term in ["python", "sql", "machine learning", "deep learning", "nlp", "data science", "analytics", "cloud", "statistics"] if term in context_text.lower()]
        score = min(95, 55 + 8 * len(relevant_terms))
        strengths = [term for term in relevant_terms[:3]]
        missing = ["cloud deployment", "Spark"]
        if "python" in jd_text.lower() and "python" not in context_text.lower():
            missing.append("Python")
        if "sql" in jd_text.lower() and "sql" not in context_text.lower():
            missing.append("SQL")
        return json.dumps({
            "match_score": round(score),
            "strengths": strengths or ["Experience in analytics and experimentation"],
            "weaknesses": ["Limited evidence of production deployment"],
            "missing_skills": missing[:4],
            "recommendation": "Proceed with interview",
            "evidence": ["Resume contains experience aligned to the stated role"],
        })

    def _claim_response(self, text: str) -> str:
        return json.dumps({
            "summary": "Claim reviewed with the available document evidence.",
            "decision": "auto_approve",
            "reason": "No major issues were detected.",
        })


def get_embedding_model() -> Embeddings:
    if os.getenv("OPENAI_API_KEY") and OpenAIEmbeddings is not None:
        return OpenAIEmbeddings(model="text-embedding-3-small")
    return SimpleEmbeddings()


def get_llm() -> Any:
    if os.getenv("OPENAI_API_KEY") and ChatOpenAI is not None:
        return ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return SimpleLLM()


def create_sample_resume_pdfs(output_dir: Path) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if canvas is None:
        raise RuntimeError("reportlab is required to build sample PDFs")
    files: List[Path] = []
    resumes = [
        ("resume_alice.pdf", "Alice Kumar\nData Scientist\nSkills: Python, SQL, machine learning, experimentation, statistics\nExperience: Built churn models and dashboards for e-commerce retail."),
        ("resume_bob.pdf", "Bob Singh\nMachine Learning Engineer\nSkills: Python, deep learning, NLP, cloud deployment, PyTorch\nExperience: Deployed NLP models in production for a healthcare startup."),
        ("resume_cara.pdf", "Cara Rao\nData Analyst\nSkills: Excel, SQL, Tableau, dashboards, reporting\nExperience: Delivered BI reports and KPI dashboards for SaaS clients."),
    ]
    for file_name, content in resumes:
        path = output_dir / file_name
        pdf = canvas.Canvas(str(path), pagesize=letter)
        pdf.setFont("Helvetica", 12)
        lines = content.splitlines()
        for index, line in enumerate(lines, start=1):
            pdf.drawString(60, 750 - 18 * index, line)
        pdf.save()
        files.append(path)
    return files


def load_pdf_documents(paths: Sequence[Path]) -> List[Document]:
    documents: List[Document] = []
    for path in paths:
        if not path.exists():
            continue
        loader = PyPDFLoader(str(path))
        documents.extend(loader.load())
    return documents


def build_resume_rag(resume_paths: Sequence[Path], job_description: str) -> Any:
    docs = load_pdf_documents(resume_paths)
    if not docs:
        raise ValueError("No readable resume documents were found.")
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunks = splitter.split_documents(docs)
    embeddings = get_embedding_model()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 4})


def evaluate_resume_against_jd(resume_path: Path, job_description: str) -> Dict[str, Any]:
    retriever = build_resume_rag([resume_path], job_description)
    context_docs = retriever.invoke(job_description)
    context_text = "\n\n".join(doc.page_content for doc in context_docs)
    llm = get_llm()
    prompt_template = ChatPromptTemplate.from_template(
        """
You are a recruiter assistant. Use only the retrieved resume text below to evaluate the candidate against the job description.
If evidence is missing, say 'Not mentioned in uploaded resumes'.
Return strict JSON with keys: match_score, strengths, weaknesses, missing_skills, recommendation, evidence.

Job Description:
{jd}

Retrieved Resume Context:
{context}
"""
    )
    prompt_text = prompt_template.format(jd=job_description, context=context_text)
    raw_output = llm.invoke(prompt_text)
    parser = JsonOutputParser()
    try:
        parsed = parser.parse(raw_output)
    except Exception:
        parsed = json.loads(raw_output) if isinstance(raw_output, str) else raw_output
    parsed.setdefault("evidence", ["Evidence retrieved from the uploaded resume"])
    return parsed


def evaluate_resumes(resume_paths: Sequence[Path], job_description: str) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for path in resume_paths:
        result = evaluate_resume_against_jd(path, job_description)
        result["resume_name"] = path.name
        results.append(result)
    return results


class ClaimState(TypedDict, total=False):
    claim: Dict[str, Any]
    verification_result: Dict[str, Any]
    eligibility_result: Dict[str, Any]
    fraud_result: Dict[str, Any]
    summary: Dict[str, Any]
    decision: Dict[str, Any]
    status: str


def verify_documents(claim: Dict[str, Any]) -> Dict[str, Any]:
    required_docs = ["claim_form", "policy_document", "medical_bill"]
    provided = claim.get("documents", [])
    missing = [doc for doc in required_docs if doc not in provided]
    if missing:
        return {"passed": False, "message": f"Missing documents: {', '.join(missing)}", "missing": missing}
    return {"passed": True, "message": "All required documents are present."}


def check_eligibility(claim: Dict[str, Any]) -> Dict[str, Any]:
    policy_end = claim.get("policy_end_date")
    if policy_end:
        end_date = datetime.strptime(policy_end, "%Y-%m-%d").date()
        if end_date < date.today():
            return {"passed": False, "message": "Policy has expired."}
    amount = float(claim.get("claim_amount", 0))
    if amount > 200000:
        return {"passed": False, "message": "Claim amount exceeds policy threshold."}
    return {"passed": True, "message": "Policy is active and within the allowed amount."}


def detect_fraud(claim: Dict[str, Any]) -> Dict[str, Any]:
    amount = float(claim.get("claim_amount", 0))
    suspicious = False
    reasons: List[str] = []
    if amount > 50000:
        suspicious = True
        reasons.append("High-value claim")
    if claim.get("claim_type") == "duplicate":
        suspicious = True
        reasons.append("Duplicate claim activity")
    if claim.get("submitted_by") == "external":
        suspicious = True
        reasons.append("Unverified claimant")
    if suspicious:
        return {"passed": False, "message": "Potential fraud indicators detected.", "reasons": reasons}
    return {"passed": True, "message": "No obvious fraud indicators detected."}


def summarize_claim(state: ClaimState) -> Dict[str, Any]:
    claim = state["claim"]
    verification = state.get("verification_result", {})
    eligibility = state.get("eligibility_result", {})
    fraud = state.get("fraud_result", {})
    decision_text = "auto_approve"
    if not verification.get("passed", True):
        decision_text = "reject"
    elif not eligibility.get("passed", True):
        decision_text = "reject"
    elif not fraud.get("passed", True):
        decision_text = "human_review"
    summary = {
        "claimant": claim.get("claimant_name", "Unknown"),
        "amount": claim.get("claim_amount", 0),
        "policy": claim.get("policy_number", "Unknown"),
        "summary": f"Claim for {claim.get('claimant_name', 'customer')} amount {claim.get('claim_amount', 0)} submitted for policy {claim.get('policy_number', 'Unknown')}.",
        "decision_hint": decision_text,
        "checks": {"verification": verification, "eligibility": eligibility, "fraud": fraud},
    }
    return summary


def route_claim(state: ClaimState) -> str:
    decision_hint = state.get("summary", {}).get("decision_hint", "auto_approve")
    if decision_hint == "human_review":
        return "human_review"
    if decision_hint == "reject":
        return "reject"
    return "approve"


def approve_claim(state: ClaimState) -> Dict[str, Any]:
    state["status"] = "approved"
    state["decision"] = {"status": "auto_approve", "reason": "All checks passed."}
    return state


def reject_claim(state: ClaimState) -> Dict[str, Any]:
    state["status"] = "rejected"
    state["decision"] = {"status": "reject", "reason": "Claim failed document or eligibility checks."}
    return state


def human_review(state: ClaimState) -> Dict[str, Any]:
    state["status"] = "pending_human_review"
    state["decision"] = {"status": "human_review", "reason": "High-risk or uncertain claim requires human approval."}
    return state


def build_claim_workflow() -> StateGraph:
    workflow = StateGraph(ClaimState)

    workflow.add_node("parallel_checks", lambda state: {
        **state,
        "verification_result": verify_documents(state["claim"]),
        "eligibility_result": check_eligibility(state["claim"]),
        "fraud_result": detect_fraud(state["claim"]),
    })
    workflow.add_node("summarize_claim", lambda state: {**state, "summary": summarize_claim(state)})
    workflow.add_node("route_claim", lambda state: state)
    workflow.add_node("approve_claim", approve_claim)
    workflow.add_node("reject_claim", reject_claim)
    workflow.add_node("human_review", human_review)

    workflow.set_entry_point("parallel_checks")
    workflow.add_edge("parallel_checks", "summarize_claim")
    workflow.add_edge("summarize_claim", "route_claim")
    workflow.add_conditional_edges(
        "route_claim",
        route_claim,
        {
            "approve": "approve_claim",
            "reject": "reject_claim",
            "human_review": "human_review",
        },
    )
    workflow.add_edge("approve_claim", END)
    workflow.add_edge("reject_claim", END)
    workflow.add_edge("human_review", END)
    return workflow.compile()


def process_claim(claim: Dict[str, Any]) -> Dict[str, Any]:
    workflow = build_claim_workflow()
    return workflow.invoke({"claim": claim})


def demo_resume_examples(base_dir: Path) -> List[Dict[str, Any]]:
    sample_dir = base_dir / "sample_resumes"
    resume_files = create_sample_resume_pdfs(sample_dir)
    job_description = """
We are hiring a Data Scientist with strong Python, SQL, statistics, machine learning, and deployment experience.
The ideal candidate should also understand experimentation and have experience translating insights into business value.
"""
    return [{"job_description": job_description, "resume_files": resume_files}]


def demo_claim_examples() -> List[Dict[str, Any]]:
    scenarios = [
        {
            "claimant_name": "Asha Patel",
            "policy_number": "POL-1001",
            "claim_amount": 12000,
            "policy_end_date": str(date.today().replace(year=date.today().year + 1)),
            "documents": ["claim_form", "policy_document", "medical_bill"],
            "claim_type": "routine",
            "submitted_by": "verified",
        },
        {
            "claimant_name": "Naina Shah",
            "policy_number": "POL-1002",
            "claim_amount": 30000,
            "policy_end_date": str(date.today().replace(year=date.today().year - 1)),
            "documents": ["claim_form", "policy_document"],
            "claim_type": "routine",
            "submitted_by": "verified",
        },
        {
            "claimant_name": "Rohan Das",
            "policy_number": "POL-1003",
            "claim_amount": 75000,
            "policy_end_date": str(date.today().replace(year=date.today().year + 1)),
            "documents": ["claim_form", "policy_document", "medical_bill"],
            "claim_type": "duplicate",
            "submitted_by": "external",
        },
    ]
    return scenarios


def run_demo(base_dir: Optional[Path] = None) -> Dict[str, Any]:
    base_dir = base_dir or Path(tempfile.gettempdir())
    demo = demo_resume_examples(base_dir)
    job_description = demo[0]["job_description"]
    resume_files = demo[0]["resume_files"]
    resume_results = evaluate_resumes(resume_files, job_description)
    claim_scenarios = demo_claim_examples()
    claim_results = [process_claim(claim) for claim in claim_scenarios]
    return {"resume_results": resume_results, "claim_results": claim_results}


if __name__ == "__main__":
    print(json.dumps(run_demo(), indent=2))
