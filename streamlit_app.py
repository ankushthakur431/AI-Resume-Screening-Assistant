import json
import tempfile
from pathlib import Path

import streamlit as st

from agentic_app import create_sample_resume_pdfs, demo_claim_examples, evaluate_resumes, process_claim

st.set_page_config(
    page_title="GenAI + Agentic AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    body, .main {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        background: radial-gradient(circle at top left, rgba(124, 77, 255, 0.15), transparent 35%),
                    radial-gradient(circle at bottom right, rgba(0, 184, 148, 0.14), transparent 30%),
                    #f4f7ff;
    }
    .app-header {
        background: linear-gradient(135deg, #6c5ce7 0%, #00b894 100%);
        border-radius: 24px;
        padding: 2rem 2rem 1.5rem 2rem;
        color: white;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.15);
        margin-bottom: 1.75rem;
        position: relative;
        overflow: hidden;
    }
    .app-header::before {
        content: "";
        position: absolute;
        width: 180px;
        height: 180px;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.15);
        top: -40px;
        right: -40px;
        filter: blur(18px);
    }
    .app-header h1 {
        margin-bottom: 0.25rem;
        font-size: 3rem;
        letter-spacing: -0.03em;
        font-weight: 800;
    }
    .app-header p {
        margin: 0;
        color: #e0f7fa;
        font-size: 1.05rem;
    }
    .pitch-card {
        background: rgba(255, 255, 255, 0.12);
        border: 1px solid rgba(255, 255, 255, 0.18);
        border-radius: 20px;
        padding: 1.5rem;
        margin-top: 1.5rem;
        color: #ebf5ff;
        backdrop-filter: blur(10px);
    }
    .pitch-card h2 {
        margin: 0 0 0.75rem 0;
        font-size: 1.75rem;
        letter-spacing: -0.02em;
        font-weight: 700;
    }
    .pitch-card p {
        margin: 0.25rem 0;
        font-size: 1.03rem;
        line-height: 1.7;
        color: #ddf2ff;
    }
    .pitch-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: rgba(255, 255, 255, 0.18);
        padding: 0.6rem 1rem;
        border-radius: 999px;
        font-weight: 700;
        margin-bottom: 1rem;
        color: #fff;
    }
    .badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        background: #eff6ff;
        color: #0b3b52;
        padding: 0.55rem 1rem;
        border-radius: 999px;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .highlight {
        color: #6c5ce7;
        font-weight: 800;
        background: rgba(108, 92, 231, 0.12);
        padding: 0.2rem 0.45rem;
        border-radius: 0.65rem;
    }
    .app-header .subtext {
        font-size: 1rem;
        opacity: 0.92;
    }
    .card {
        background: #ffffff;
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 16px 40px rgba(0, 0, 0, 0.08);
        margin-bottom: 1.5rem;
    }
    .section-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #2d3436;
        margin-bottom: 0.75rem;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #081c24 0%, #0b3b52 100%);
        color: #ffffff;
    }
    .sidebar .stButton>button {
        background-color: #0984e3;
        color: white;
        border-radius: 999px;
        padding: 0.95rem 1.3rem;
        font-weight: 700;
        box-shadow: 0 10px 30px rgba(9, 132, 227, 0.24);
    }
    .sidebar .stButton>button:hover {
        background-color: #74b9ff;
        color: #020202;
    }
    .result-card {
        border-radius: 1.5rem;
        padding: 1.5rem;
        background: linear-gradient(180deg, #f7f9fc 0%, #ffffff 100%);
        border: 1px solid #dfe6e9;
        margin-bottom: 1.25rem;
    }
    .result-card h3 {
        margin-bottom: 0.65rem;
        font-weight: 700;
    }
    .insight-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
        gap: 1rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .insight-card {
        padding: 1rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #f0f4ff, #ffffff);
        border: 1px solid #dfe4fc;
        box-shadow: 0 12px 30px rgba(16, 24, 40, 0.06);
    }
    .insight-card strong {
        display: block;
        margin-bottom: 0.5rem;
        color: #130f40;
    }
    .badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        background: #eff6ff;
        color: #0b3b52;
        padding: 0.55rem 1rem;
        border-radius: 999px;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .highlight {
        color: #6c5ce7;
        font-weight: 800;
        background: rgba(108, 92, 231, 0.12);
        padding: 0.2rem 0.45rem;
        border-radius: 0.65rem;
    }
    .info-text {
        color: #636e72;
    }
    .divider {
        height: 4px;
        width: 100%;
        background: linear-gradient(90deg, rgba(98,0,238,0) 0%, rgba(0,184,148,1) 50%, rgba(98,0,238,0) 100%);
        margin: 2rem 0;
        border: none;
    }
    .sidebar .stTextInput>div>input {
        border-radius: 14px;
    }
    .sidebar .stFileUploader>div {
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.2);
        background: rgba(255,255,255,0.06);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="app-header"><h1>GenAI + Agentic AI Assignment</h1><p class="subtext">AI-powered review for resumes and insurance workflows — fast, transparent, and beautifully designed.</p><div class="pitch-card"><div class="pitch-badge">🤖 AI Pitcher Engine</div><h2>Turn documents into decisions with AI-powered insights.</h2><p>Analyze resumes and insurance claims with an elegant intelligent assistant that speeds hiring and claim adjudication while keeping every result <span class="highlight">clear</span>, <span class="highlight">actionable</span>, and <span class="highlight">human-friendly</span>.</p></div></div>',
    unsafe_allow_html=True,
)

if "resume_results" not in st.session_state:
    st.session_state.resume_results = []
if "claim_results" not in st.session_state:
    st.session_state.claim_results = []

with st.sidebar:
    st.markdown("<div class='section-title'>🧾 Resume Screening</div>", unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Upload resumes (PDF)",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload candidate resumes to evaluate against the job description.",
    )
    jd = st.text_area(
        "Job Description",
        value="We are hiring a Data Scientist with strong Python, SQL, statistics, machine learning, and deployment experience.",
        height=140,
    )
    if st.button("🚀 Run Resume Evaluation"):
        temp_dir = Path(tempfile.gettempdir()) / "resume_demo"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_paths = []
        if uploaded_files:
            for upload in uploaded_files:
                path = temp_dir / upload.name
                path.write_bytes(upload.getvalue())
                temp_paths.append(path)
        else:
            temp_paths = create_sample_resume_pdfs(temp_dir)
        st.session_state.resume_results = evaluate_resumes(temp_paths, jd)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>📄 Insurance Claims</div>", unsafe_allow_html=True)
    if st.button("⚡ Run Claim Scenarios"):
        st.session_state.claim_results = [process_claim(claim) for claim in demo_claim_examples()]

st.markdown('<div class="card"><div class="section-title">1. Resume Screening Assistant</div>', unsafe_allow_html=True)
if st.session_state.resume_results:
    for item in st.session_state.resume_results:
        match_score = item.get('match_score', 'N/A')
        recommendation = item.get('recommendation', 'Review manually')
        strengths = item.get('strengths', [])
        missing_skills = item.get('missing_skills', [])
        evidence = item.get('evidence', [])
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown(f"<div class='badge-pill'>✨ Resume Insight</div><h3>{item.get('resume_name', 'Resume')}</h3>", unsafe_allow_html=True)
        st.markdown('<p class="info-text">Next-level candidate scoring with recommendation badges and insight blocks.</p>', unsafe_allow_html=True)
        st.markdown('<div class="insight-grid">', unsafe_allow_html=True)
        st.markdown(f"<div class='insight-card'><strong>🌟 Match Score</strong><span class='highlight'>{match_score}%</span><p>How well this resume aligns with the job description.</p></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='insight-card'><strong>✅ Recommendation</strong><span class='highlight'>{recommendation}</span><p>Suggested next step based on document analysis.</p></div>", unsafe_allow_html=True)
        top_strength = strengths[0] if strengths else 'Key strengths will appear here.'
        st.markdown(f"<div class='insight-card'><strong>💪 Top Strength</strong><span class='highlight'>{top_strength}</span><p>Primary skill or experience identified from the resume.</p></div>", unsafe_allow_html=True)
        missing_text = ', '.join(missing_skills) if missing_skills else 'No major gaps detected.'
        st.markdown(f"<div class='insight-card'><strong>⚠️ Missing Skills</strong><span class='highlight'>{missing_text}</span><p>Skills the resume is weaker on compared to the job requirements.</p></div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        if evidence:
            st.markdown(f"<div class='insight-card' style='grid-column: span 2;'><strong>📌 Evidence Summary</strong><p>{evidence[0]}</p></div>", unsafe_allow_html=True)
        st.markdown('<div class="info-text">Detailed JSON output is shown below for full traceability.</div>', unsafe_allow_html=True)
        st.json(item)
        st.markdown('</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="result-card"><p class="info-text">Upload PDFs or click the button to run demo evaluations.</p></div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="card"><div class="section-title">2. Insurance Claim Processing Agent</div>', unsafe_allow_html=True)
if st.session_state.claim_results:
    for index, item in enumerate(st.session_state.claim_results, start=1):
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown(f"<div class='badge-pill'>🔥 Claim Signal</div><h3>Scenario {index}</h3>", unsafe_allow_html=True)
        st.markdown('<p class="info-text">Prediction highlights: <span class="highlight">Approval status</span> and risk factors are shown prominently.</p>', unsafe_allow_html=True)
        st.json(item)
        st.markdown('</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="result-card"><p class="info-text">Click the button to run the claim scenarios.</p></div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
