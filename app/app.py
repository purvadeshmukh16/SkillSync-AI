import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
from groq import Groq
import google.generativeai as genai          # ← single import (removed duplicate)
from dotenv import load_dotenv
import os
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# ══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="SkillSync AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════
# CUSTOM CSS
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    [data-testid="stSidebar"] { background-color: #1a1d24; }
    [data-testid="stMetric"] {
        background-color: #1e2130;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #2d3250;
    }
    [data-testid="stMetricValue"] { color: #7c83fd; font-size: 1.5rem; }
    [data-testid="stMetricDelta"] { color: #4ecdc4; }
    .stButton > button {
        background-color: #7c83fd;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
        width: 100%;
    }
    .stButton > button:hover { background-color: #6670f0; }
    h1 { color: #7c83fd; }
    h2 { color: #4ecdc4; }
    h3 { color: #ffffff; }
    [data-testid="stExpander"] {
        background-color: #1e2130;
        border: 1px solid #2d3250;
        border-radius: 10px;
    }
    .phase-box {
        background: linear-gradient(135deg, #1e2130, #2d3250);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        border: 1px solid #7c83fd;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# LOAD MODELS & DATA
# ══════════════════════════════════════════════════════════════
@st.cache_resource
def load_models():
    try:
        path              = "models"
        mlb               = pickle.load(open(f"{path}/mlb.pkl",               "rb"))
        role_skill_matrix = pickle.load(open(f"{path}/role_skill_matrix.pkl", "rb"))
        salary_model      = pickle.load(open(f"{path}/salary_model.pkl",      "rb"))
        le_education      = pickle.load(open(f"{path}/le_education.pkl",      "rb"))
        le_city           = pickle.load(open(f"{path}/le_city.pkl",           "rb"))
        le_jobmode        = pickle.load(open(f"{path}/le_jobmode.pkl",        "rb"))
        le_role           = pickle.load(open(f"{path}/le_role.pkl",           "rb"))
        all_skills        = pickle.load(open(f"{path}/all_skills.pkl",        "rb"))
        df                = pd.read_csv(f"{path}/jobs_with_nlp.csv")
        return (mlb, role_skill_matrix, salary_model,
                le_education, le_city, le_jobmode, le_role, all_skills, df)
    except Exception as e:
        st.error(f"❌ Failed to load model files: {e}")
        st.stop()

(mlb, role_skill_matrix, salary_model,
 le_education, le_city, le_jobmode, le_role, all_skills, df) = load_models()

# Extra skills merged IMMEDIATELY after load — FIX: was merged after recommend_roles used all_skills
extra_skills = [
    "c", "c++", "c#", "rust",
    "spring", "hibernate", ".net",
    "oracle", "sql server", "firebase",
    "bootstrap", "jquery", "figma",
    "excel vba", "power automate",
    "junit", "jest", "cypress",
    "github actions", "gitlab ci",
    "matplotlib", "plotly",
    "hugging face", "mlflow",
]
all_skills_display = sorted(list(set(all_skills + extra_skills)))

# ══════════════════════════════════════════════════════════════
# CORE FUNCTIONS
# ══════════════════════════════════════════════════════════════
def recommend_roles(user_skills, experience, top_n=3):
    known_skills = [s for s in user_skills if s in all_skills]
    if not known_skills:
        known_skills = user_skills[:1]
    user_vector  = pd.DataFrame(
        mlb.transform([known_skills]), columns=all_skills
    )
    similarities = cosine_similarity(user_vector, role_skill_matrix)[0]
    avg_salary   = df.groupby("role")["salary_estimate"].mean()
    results = pd.DataFrame({
        "Role":            role_skill_matrix.index,
        "Match %":         (similarities * 100).round(1),
        "Expected Salary": (avg_salary.values + experience * 60000).astype(int)
    })
    results = results.sort_values("Match %", ascending=False).head(top_n)
    results = results.reset_index(drop=True)
    results.index = results.index + 1
    return results


def predict_salary(role, city, experience, education, job_mode):
    """Use the trained salary model for prediction."""
    try:
        role_enc  = le_role.transform([role])[0]
        city_enc  = le_city.transform([city])[0] if city in le_city.classes_ else 0
        edu_enc   = le_education.transform([education])[0]
        mode_enc  = le_jobmode.transform([job_mode])[0] if job_mode in le_jobmode.classes_ else 0
        X         = np.array([[role_enc, city_enc, experience, edu_enc, mode_enc]])
        predicted = salary_model.predict(X)[0]
        return max(int(predicted), 50000)
    except Exception:
        # fallback to dataset average if encoding fails
        avg = df[df["role"] == role]["salary_estimate"].mean()
        return int(avg + experience * 60000) if not np.isnan(avg) else 600000


def skill_gap_analyzer(user_skills, target_role):
    role_jobs  = df[df["role"] == target_role]["skills"]
    skill_freq = {}
    for skill_str in role_jobs:
        if isinstance(skill_str, str):
            for skill in skill_str.split(","):
                skill = skill.strip().lower()
                skill_freq[skill] = skill_freq.get(skill, 0) + 1
    skill_freq        = dict(sorted(skill_freq.items(),
                                    key=lambda x: x[1], reverse=True))
    user_skills_lower = [s.strip().lower() for s in user_skills]
    missing = {k: v for k, v in skill_freq.items()
               if k not in user_skills_lower}
    total  = len(role_jobs)
    result = pd.DataFrame({
        "Missing Skill": list(missing.keys()),
        "Demand Count":  list(missing.values()),
        "Demand %":      [round(v / total * 100, 1) for v in missing.values()]
    })
    return result.reset_index(drop=True)


def job_matcher(role, city, experience, job_mode, top_n=5):
    filtered = df[df["role"] == role].copy()
    if job_mode != "Any":
        temp = filtered[filtered["job_mode"] == job_mode]
        if len(temp) >= 3:
            filtered = temp
    if city != "Any":
        temp = filtered[filtered["city"] == city]
        if len(temp) >= 3:
            filtered = temp
    temp = filtered[
        (filtered["experience_min"] <= experience) &
        (filtered["experience_max"] >= experience)
    ]
    if len(temp) >= 3:
        filtered = temp
    if len(filtered) > top_n:
        tfidf     = TfidfVectorizer()
        tfidf_mat = tfidf.fit_transform(filtered["combined_text"].fillna(""))
        q_vec     = tfidf.transform([f"{role} {city} {job_mode}"])
        scores    = cosine_similarity(q_vec, tfidf_mat)[0]
        filtered  = filtered.copy()
        filtered["match_score"] = scores
        filtered  = filtered.sort_values("match_score", ascending=False)
    result = filtered.head(top_n)[[
        "title", "company", "city", "job_mode",
        "experience_min", "experience_max",
        "salary_estimate", "salary_raw", "source"
    ]].reset_index(drop=True)
    result.index = result.index + 1
    return result

# ══════════════════════════════════════════════════════════════
# CHATBOT FUNCTIONS
# ══════════════════════════════════════════════════════════════
def build_dataset_context(df):
    """Build a summary of your dataset to give Gemini context."""
    
    top_skills = []
    for s in df["skills"].dropna():
        top_skills.extend([x.strip() for x in s.split(",")])
    top_skills = [s for s, _ in Counter(top_skills).most_common(15)]
    
    role_salary = df.groupby("role")["salary_estimate"].mean()
    salary_info = "\n".join([
        f"- {role}: ₹{sal/100000:.1f} LPA"
        for role, sal in role_salary.sort_values(ascending=False).items()
    ])
    
    city_info = "\n".join([
        f"- {city}: {cnt} jobs"
        for city, cnt in df["city"].value_counts().head(8).items()
    ])
    
    context = f"""
You are a Job Market Assistant for India. You have access to a dataset of
{len(df):,} real Indian job listings scraped from Naukri.com.

DATASET OVERVIEW:
- Total Jobs: {len(df):,}
- Job Roles ({df['role'].nunique()}): {', '.join(sorted(df['role'].unique()))}
- Cities covered: {', '.join(df['city'].unique()[:10])}
- Sources: {', '.join(df['source'].unique())}
- Job Modes: {', '.join(df['job_mode'].unique())}

TOP 15 IN-DEMAND SKILLS:
{', '.join(top_skills)}

AVERAGE SALARY BY ROLE:
{salary_info}

TOP CITIES BY JOB COUNT:
{city_info}

EDUCATION DISTRIBUTION:
{df['education'].value_counts().to_string()}

EXPERIENCE DISTRIBUTION:
- Fresher (0 yrs): {(df['experience_min']==0).sum()} jobs
- 1-3 years: {((df['experience_min']>=1) & (df['experience_min']<=3)).sum()} jobs
- 3-5 years: {((df['experience_min']>=3) & (df['experience_min']<=5)).sum()} jobs
- 5+ years: {(df['experience_min']>5).sum()} jobs

IMPORTANT INSTRUCTIONS:
- You are a knowledgeable Indian job market expert and career advisor
- Answer naturally like a human expert, not like a bot reading a database
- Never use words like "dataset", "data", "records", "entries", or "scraped"
- Instead of "dataset shows", say "based on current market trends" or "in the Indian job market"
- Instead of "in our dataset", say "across Indian companies" or "in the current market"
- Always give specific numbers but present them naturally (e.g. "over 500 companies are hiring" not "500 records found")
- If asked something unrelated say: "I specialize in Indian job market and career guidance, try asking me about roles, skills or salaries"
- Use Indian salary format (LPA, Lakhs)
- If user greets you, respond warmly and introduce yourself as a career advisor
- Never mention, reveal, or reference the developer's name, creator's name, or who built this application
- If asked who made you or who is your developer, say: "I am Career Insights Assistant, built to help you navigate the Indian job market"
- Never say any personal names under any circumstances

ANSWER FORMAT RULES:
- Give detailed, elaborate answers with minimum 5-8 sentences per response
- Always structure your answer with clear sections using bullet points or numbered lists
- Start with a direct answer, then explain with market context, then give actionable career advice
- Include specific numbers and comparisons presented as market insights
- End every answer with 1-2 practical tips or next steps for the user
- Never give one-liner answers, always elaborate and be thorough
- If asked about a skill, mention which roles need it, salary impact, and how in-demand it is
- If asked about salary, mention city-wise differences, experience impact, and role comparison
- If asked about a city, mention top roles there, which companies are hiring, and work mode split
- Always complete your answer fully, never stop mid-sentence or mid-point
- If listing items, always finish the complete list before ending
- Never start answers with "Hello", "Hi", or any greeting unless the user greeted you first
- Get straight to the answer, greetings only when user says hi/hello first
"""
    return context

# Build context once at startup (not cached with st.cache_data to avoid DataFrame hashing issues)
DATASET_CONTEXT = build_dataset_context(df)

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def chatbot_response(question):
    try:
        prompt = f"{DATASET_CONTEXT}\n\nUser Question: {question}"
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ Something went wrong: {str(e)}"

# ══════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════
def init_state():
    defaults = {
        "phase":             1,
        "user_skills":       [],
        "experience":        0,
        "education":         "B.Tech",
        "target_role":       None,
        "recommended_roles": None,
        "show_results":      False,
        "gap_analyzed":      False,
        "chat_history":      [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("# 🎯 SkillSync AI")
    st.markdown("---")
    # FIX: label must match the elif page == checks below — use consistent names
    page = st.radio(
        "Navigate",
        ["🏠 Home", "📊 Dashboard", "🤖 Career Advisor", "💬 Chatbot"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("### 📊 Dataset Stats")
    st.metric("Total Jobs",  f"{len(df):,}")
    st.metric("Job Roles",   df["role"].nunique())
    st.metric("Cities",      df["city"].nunique())
    st.metric("Companies",   df["company"].nunique())
    st.markdown("---")
    if page == "🤖 Career Advisor":
        st.markdown("### 📍 Your Progress")
        phases = ["Phase 1 — Role Finder",
                  "Phase 2 — Skill Gap",
                  "Phase 3 — Job Matcher"]
        for i, p in enumerate(phases, 1):
            if i < st.session_state.phase:
                st.markdown(f"✅ {p}")
            elif i == st.session_state.phase:
                st.markdown(f"▶️ **{p}**")
            else:
                st.markdown(f"⬜ {p}")

# ══════════════════════════════════════════════════════════════
# PAGE — HOME
# ══════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown("# 🎯 SkillSync AI")
    st.markdown("### AI-Powered Job Market Intelligence Platform for India")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("💼 Total Jobs",  f"{len(df):,}")
    c2.metric("🎭 Job Roles",   df["role"].nunique())
    c3.metric("📍 Cities",      df["city"].nunique())
    c4.metric("🏢 Companies",   df["company"].nunique())

    st.markdown("---")
    st.markdown("## How It Works")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""
        <div class="phase-box">
        <h3>🔍 Phase 1</h3>
        <b>Role & Salary Predictor</b><br><br>
        Enter your skills, experience and education.
        Get top 3 matching roles with expected salary.
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="phase-box">
        <h3>📚 Phase 2</h3>
        <b>Skill Gap Analyzer</b><br><br>
        If your target role is not in results,
        see exactly which skills you are missing.
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="phase-box">
        <h3>💼 Phase 3</h3>
        <b>Job Matcher</b><br><br>
        Get real matching job listings from Indian
        companies based on your complete profile.
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown("""
        <div class="phase-box">
        <h3>💬 Chatbot</h3>
        <b>Career Insights Assistant</b><br><br>
        Ask any question about the Indian job market
        and get instant data-driven answers.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## Quick Market Insights")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Top 5 Most In-Demand Skills**")
        all_sk = []
        for s in df["skills"].dropna():
            all_sk.extend([x.strip() for x in s.split(",")])
        top5 = pd.DataFrame(
            Counter(all_sk).most_common(5),
            columns=["Skill", "Count"]
        ).set_index("Skill")
        st.bar_chart(top5, color="#7c83fd")
    with c2:
        st.markdown("**Average Salary by Role — Top 5 (LPA)**")
        top_sal = df.groupby("role")["salary_estimate"].mean()\
                    .sort_values(ascending=False).head(5)
        top_sal_df = pd.DataFrame({
            "Salary LPA": (top_sal.values / 100000).round(1)
        }, index=top_sal.index)
        st.bar_chart(top_sal_df, color="#4ecdc4")

# ══════════════════════════════════════════════════════════════
# PAGE — DASHBOARD
# ══════════════════════════════════════════════════════════════

elif page == "📊 Dashboard":
    st.markdown("# 📊 Indian Job Market Dashboard")
    st.markdown(
        f"*{len(df):,} listings · {df['role'].nunique()} roles · "
        f"{df['city'].nunique()} cities · Source: Naukri.com*"
    )
    st.markdown("---")

    import plotly.graph_objects as go

    BLUE     = "#4C9BE8"
    TEAL     = "#38B2AC"
    SLATE    = "#64748B"
    CHART_BG = "#0e1117"
    GRID     = "#1e2130"
    TEXT     = "#CBD5E1"
    SUBTEXT  = "#94A3B8"

    # Tonal blue gradient for multi-bar charts
    BAR_COLORS = [
        "#2563EB","#3B82F6","#60A5FA","#93C5FD","#BFDBFE",
        "#1D4ED8","#4C9BE8","#7EC8E3","#38B2AC","#4FD1C5",
        "#0EA5E9","#38BDF8","#7DD3FC","#BAE6FD","#E0F2FE",
        "#1E40AF","#1D4ED8","#2563EB","#3B82F6","#60A5FA"
    ]

    DONUT_MODE = ["#3B82F6", "#38B2AC", "#64748B"]
    DONUT_EDU  = ["#2563EB", "#3B82F6", "#60A5FA", "#93C5FD", "#BFDBFE"]

    def base_layout(height=380, b=20):
        return dict(
            paper_bgcolor=CHART_BG,
            plot_bgcolor=CHART_BG,
            font=dict(color=TEXT, family="sans-serif"),
            margin=dict(t=20, b=b, l=10, r=40),
            height=height,
            xaxis=dict(showgrid=False, color=SUBTEXT,
                       linecolor=GRID, zeroline=False),
            yaxis=dict(showgrid=True, gridcolor=GRID,
                       color=SUBTEXT, linecolor=GRID, zeroline=False),
        )

    # ── Row 1: Top Roles + two donuts ─────────────────────────────────────
    c1, c2, c3 = st.columns([2, 1, 1])

    with c1:  # <-- fixed: was missing a space, causing IndentationError
        st.markdown("### Top 10 Roles by Listings")
        role_data = df["role"].value_counts().head(10).sort_values()
        fig = go.Figure(go.Bar(
            x=role_data.values,
            y=role_data.index.tolist(),
            orientation="h",
            marker=dict(color=BLUE, opacity=0.85),
            text=role_data.values,
            textposition="outside",
            textfont=dict(color=TEXT, size=11)
        ))
        layout = base_layout(380)
        layout["xaxis"]["title"] = "Listings"
        fig.update_layout(**layout)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("### Work Mode Split")
        st.bar_chart(df["job_mode"].value_counts(), color="#3B82F6")
    with c3:
        st.markdown("### Education Required")
        st.bar_chart(df["education"].value_counts(), color="#38B2AC")

    # ── Row 2: Experience + Recency ───────────────────────────────────────
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### Experience Level Distribution")
        exp_labels = ["Fresher", "0-2 yrs", "2-5 yrs", "5-10 yrs", "10+ yrs"]
        exp_bins = pd.cut(
            df["experience_min"],
            bins=[-1, 0, 2, 5, 10, 20],
            labels=exp_labels
        )
        exp_counts = exp_bins.value_counts().sort_index()
        fig = go.Figure(go.Bar(
            x=exp_counts.index.tolist(),
            y=exp_counts.values.tolist(),
            marker=dict(color=TEAL, opacity=0.85),
            text=exp_counts.values.tolist(),
            textposition="outside",
            textfont=dict(color=TEXT, size=11)
        ))
        layout = base_layout(350, b=40)
        layout["yaxis"]["title"] = "Listings"
        fig.update_layout(**layout)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("### Posting Recency")
        rec_labels = ["< 1 week", "1-2 weeks", "2-3 weeks", "3+ weeks"]
        rec_bins = pd.cut(
            df["days_ago"],
            bins=[-2, 7, 14, 21, 31],
            labels=rec_labels
        )
        rec_counts = rec_bins.value_counts().sort_index()
        fig = go.Figure(go.Bar(
            x=rec_counts.index.tolist(),
            y=rec_counts.values.tolist(),
            marker=dict(color=BLUE, opacity=0.85),
            text=rec_counts.values.tolist(),
            textposition="outside",
            textfont=dict(color=TEXT, size=11)
        ))
        layout = base_layout(350, b=40)
        layout["yaxis"]["title"] = "Listings"
        fig.update_layout(**layout)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── Row 3: Salary by Role ─────────────────────────────────────────────
    st.markdown("### Average Salary by Role (LPA)")
    sal_data = df.groupby("role")["salary_estimate"].mean().sort_values(ascending=False)
    fig = go.Figure(go.Bar(
        x=sal_data.index.tolist(),
        y=(sal_data.values / 100000).round(2).tolist(),
        marker=dict(
            color=(sal_data.values / 100000).round(2).tolist(),
            colorscale=[[0, "#1D4ED8"], [0.5, "#3B82F6"], [1, "#93C5FD"]],
            showscale=False
        ),
        text=(sal_data.values / 100000).round(1).tolist(),
        textposition="outside",
        textfont=dict(color=TEXT, size=10)
    ))
    layout = base_layout(380, b=100)
    layout["xaxis"]["tickangle"] = -35
    layout["yaxis"]["title"] = "LPA"
    layout["margin"]["r"] = 20
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── Row 4: City + Skills ──────────────────────────────────────────────
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### Jobs by City")
        city_data = df["city"].value_counts().head(10).sort_values()
        fig = go.Figure(go.Bar(
            x=city_data.values,
            y=city_data.index.tolist(),
            orientation="h",
            marker=dict(color=TEAL, opacity=0.85),
            text=city_data.values,
            textposition="outside",
            textfont=dict(color=TEXT, size=11)
        ))
        layout = base_layout(380)
        layout["xaxis"]["title"] = "Job Count"
        fig.update_layout(**layout)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("### Top 15 In-Demand Skills")
        all_sk = []
        for s in df["skills"].dropna():
            all_sk.extend([x.strip() for x in s.split(",")])
        top15      = Counter(all_sk).most_common(15)
        sk_names   = [x[0] for x in reversed(top15)]
        sk_vals    = [x[1] for x in reversed(top15)]
        fig = go.Figure(go.Bar(
            x=sk_vals,
            y=sk_names,
            orientation="h",
            marker=dict(
                color=sk_vals,
                colorscale=[[0, "#1D4ED8"], [0.5, "#3B82F6"], [1, "#93C5FD"]],
                showscale=False
            ),
            text=sk_vals,
            textposition="outside",
            textfont=dict(color=TEXT, size=10)
        ))
        layout = base_layout(380)
        layout["xaxis"]["title"] = "Frequency"
        fig.update_layout(**layout)
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# PAGE — CAREER ADVISOR
# ══════════════════════════════════════════════════════════════
elif page == "🤖 Career Advisor":
    st.markdown("# 🤖 AI Career Advisor")
    st.markdown(
        "*Find your best matching role, identify skill gaps, "
        "and discover real job opportunities*"
    )
    st.markdown("---")

    # ── PHASE 1 ──────────────────────────────────────────────
    if st.session_state.phase == 1:
        st.markdown("## 🔍 Phase 1 — Role & Salary Predictor")

        c1, c2 = st.columns([2, 1])
        with c1:
            selected_skills = st.multiselect(
                "🛠️ Search and select your skills",
                options=all_skills_display,
                default=st.session_state.user_skills,
                placeholder="Type to search skills (e.g. python, sql, aws...)"
            )
        with c2:
            experience = st.slider(
                "📅 Years of Experience", 0, 20,
                st.session_state.experience
            )
            education = st.selectbox(
                "🎓 Highest Education",
                ["B.Tech", "M.Tech", "M.Sc", "MBA", "Bachelor's"],
                index=["B.Tech", "M.Tech", "M.Sc",
                       "MBA", "Bachelor's"].index(
                    st.session_state.education)
            )

        if st.button("🔍 Find My Best Roles", use_container_width=True):
            if len(selected_skills) < 2:
                st.error("⚠️ Please select at least 2 skills.")
            else:
                st.session_state.user_skills  = selected_skills
                st.session_state.experience   = experience
                st.session_state.education    = education
                st.session_state.show_results = True

                with st.spinner("Analyzing your skills..."):
                    results = recommend_roles(selected_skills, experience)
                st.session_state.recommended_roles = results

        if (st.session_state.show_results and
                st.session_state.recommended_roles is not None):
            results = st.session_state.recommended_roles
            st.markdown("---")
            st.markdown("### 🎯 Top 3 Recommended Roles for You")

            c1, c2, c3 = st.columns(3)
            cols      = [c1, c2, c3]
            role_list = []
            for i, row in results.iterrows():
                # Use salary_model prediction for display
                pred_sal = predict_salary(
                    row["Role"],
                    "Bengaluru",
                    st.session_state.experience,
                    st.session_state.education,
                    "Hybrid"
                )
                with cols[i - 1]:
                    st.metric(
                        label=f"#{i} {row['Role']}",
                        value=f"₹{pred_sal/100000:.1f} LPA",
                        delta=f"{row['Match %']}% match"
                    )
                role_list.append(row["Role"])

            st.markdown("---")
            st.markdown("### ❓ Is your target role in these results?")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("✅ YES — Show me matching jobs",
                             use_container_width=True, key="yes_btn"):
                    st.session_state.target_role  = role_list[0]
                    st.session_state.phase        = 3
                    st.session_state.show_results = False
                    st.rerun()
            with c2:
                if st.button("❌ NO — Analyze my skill gap",
                             use_container_width=True, key="no_btn"):
                    st.session_state.phase        = 2
                    st.session_state.show_results = False
                    st.rerun()

    # ── PHASE 2 ──────────────────────────────────────────────
    elif st.session_state.phase == 2:
        st.markdown("## 📚 Phase 2 — Skill Gap Analyzer")
        st.info(f"👤 Your skills: **{', '.join(st.session_state.user_skills)}**")
        st.markdown("---")

        target_role = st.selectbox(
            "🎯 Select your target role",
            options=sorted(df["role"].unique())
        )

        if st.button("🔍 Analyze My Skill Gap",
                     use_container_width=True, key="gap_btn"):
            st.session_state.target_role  = target_role
            st.session_state.gap_analyzed = True

        if st.session_state.gap_analyzed:
            gaps = skill_gap_analyzer(
                st.session_state.user_skills,
                st.session_state.target_role
            )
            if len(gaps) == 0:
                st.success(
                    f"🎉 You already have all the skills for "
                    f"**{st.session_state.target_role}**!"
                )
            else:
                st.markdown(
                    f"### 🚨 Missing Skills for "
                    f"**{st.session_state.target_role}**"
                )
                st.warning(
                    f"You are missing **{len(gaps)} skills**. "
                    f"Focus on high demand ones first."
                )
                top_gaps = gaps.head(6)
                cols = st.columns(3)
                for i, row in top_gaps.iterrows():
                    with cols[i % 3]:
                        st.metric(
                            label=row["Missing Skill"].title(),
                            value=f"{row['Demand %']}%",
                            delta=f"{row['Demand Count']} jobs need it"
                        )
                st.markdown("#### 📋 Complete Missing Skills List")
                st.dataframe(gaps, use_container_width=True, hide_index=True)

            st.markdown("---")
            if st.button("➡️ Now Find Matching Jobs",
                         use_container_width=True, key="to_p3"):
                st.session_state.phase        = 3
                st.session_state.gap_analyzed = False
                st.rerun()

        st.markdown("---")
        if st.button("⬅️ Back to Phase 1",
                     use_container_width=True, key="back_p1"):
            st.session_state.phase        = 1
            st.session_state.show_results = True
            st.session_state.gap_analyzed = False
            st.rerun()

    # ── PHASE 3 ──────────────────────────────────────────────
    elif st.session_state.phase == 3:
        st.markdown("## 💼 Phase 3 — Job Matcher")
        skills_preview = ", ".join(st.session_state.user_skills[:5])
        if len(st.session_state.user_skills) > 5:
            skills_preview += "..."
        st.info(
            f"👤 Profile: **{skills_preview}** | "
            f"{st.session_state.experience} yrs exp"
        )
        st.markdown("---")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            all_roles = sorted(df["role"].unique())
            role_idx  = (all_roles.index(st.session_state.target_role)
                         if st.session_state.target_role in all_roles else 0)
            role = st.selectbox("🎯 Target Role", all_roles, index=role_idx)
        with c2:
            city_options = (["Any"] + sorted([
                c for c in df["city"].unique()
                if c not in ["India", "Other"]
            ]))
            city = st.selectbox("📍 Preferred City", city_options)
        with c3:
            job_mode = st.selectbox(
                "💼 Job Mode", ["Any", "On-site", "Hybrid", "Remote"]
            )
        with c4:
            top_n = st.slider("🔢 Number of Jobs", 1, 20, 5)

        experience = st.slider(
            "📅 Your Experience (years)", 0, 20,
            st.session_state.experience
        )

        if st.button("🔍 Find Matching Jobs",
                     use_container_width=True, key="match_btn"):
            with st.spinner("Searching for matching jobs..."):
                matches = job_matcher(role, city, experience, job_mode, top_n)
            if len(matches) == 0:
                st.warning(
                    "⚠️ No jobs found. "
                    "Try selecting 'Any' for city or job mode."
                )
            else:
                st.markdown(
                    f"### ✅ Found {len(matches)} Matching Jobs for **{role}**"
                )
                st.markdown("---")
                for i, row in matches.iterrows():
                    with st.expander(
                        f"#{i}  {row['title']}  —  {row['company']}",
                        expanded=(i == 1)
                    ):
                        c1, c2, c3 = st.columns(3)
                        c1.metric("📍 City",       row["city"])
                        c2.metric("💼 Mode",       row["job_mode"])
                        c3.metric("💰 Est. Salary",
                                  f"₹{row['salary_estimate']/100000:.1f} LPA")
                        c1.metric("📅 Experience",
                                  f"{row['experience_min']}–"
                                  f"{row['experience_max']} yrs")
                        c2.metric("🏢 Company",    row["company"])
                        c3.metric("🔗 Source",     row["source"])
                        if str(row["salary_raw"]) not in \
                                ["Not Disclosed", "nan"]:
                            st.success(
                                f"💰 Actual Posted Salary: "
                                f"**{row['salary_raw']}**"
                            )

        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("⬅️ Back to Phase 2",
                         use_container_width=True, key="back_p2"):
                st.session_state.phase = 2
                st.rerun()
        with c2:
            if st.button("🔄 Start Over",
                         use_container_width=True, key="restart"):
                for key in ["phase", "user_skills", "experience",
                            "education", "target_role",
                            "recommended_roles", "show_results",
                            "gap_analyzed"]:
                    del st.session_state[key]
                st.rerun()

# ══════════════════════════════════════════════════════════════
# PAGE — CHATBOT   ← FIX: was "💬 Career Insights Assistant", now matches sidebar "💬 Chatbot"
# ══════════════════════════════════════════════════════════════
elif page == "💬 Chatbot":
    st.markdown("# 💬 Career Insights Assistant")
    st.markdown("*Ask any question about the Indian job market dataset*")
    st.markdown("---")

    # Welcome message
    if not st.session_state.chat_history:
        st.session_state.chat_history = [{
            "role": "assistant",
            "content": (
                f"👋 Hi! I am your Career Insights Assistant.\n\n"
                f"I can answer questions about skills, salaries, "
                f"cities, companies, and more from our dataset of "
                f"**{len(df):,} Indian job listings**.\n\n"
                f"Try asking:\n"
                f"- *What skills are needed for data analyst?*\n"
                f"- *Which city has the most jobs?*\n"
                f"- *What is the salary for ML engineer?*\n"
                f"- *Which companies are hiring for DevOps?*\n"
                f"- *How many remote jobs are available?*"
            )
        }]

    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User input
    user_input = st.chat_input("Ask me anything about the job market...")

    if user_input:
        st.session_state.chat_history.append({
            "role": "user", "content": user_input
        })
        with st.chat_message("user"):
            st.markdown(user_input)

        # FIX: added spinner for better UX during API call
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = chatbot_response(user_input)
            st.markdown(response)

        st.session_state.chat_history.append({
            "role": "assistant", "content": response
        })

    # Clear chat
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()