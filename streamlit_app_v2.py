import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import date, datetime
import random
import numpy as np
from PIL import Image
import tensorflow as tf

# ────────────────────────────────────────────────────────────────────────────
# AI MODEL CONFIGURATION & INITIALIZATION
# ────────────────────────────────────────────────────────────────────────────
CLASS_NAMES = ["Healthy", "ALL", "AML", "CLL", "CML"]
IMG_SIZE = (128, 128)

@st.cache_resource
def load_tflite_model():
    """Loads the TFLite model into global cache so it isn't reloaded on every click."""
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
    return interpreter

def predict_smear(uploaded_file, interpreter):
    """Processes the image and runs inference via TFLite."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    img = Image.open(uploaded_file).convert('RGB')
    img = img.resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img_batch)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    predictions = output_data[0]
    
    predicted_class_idx = np.argmax(predictions)
    confidence = predictions[predicted_class_idx]
    
    return CLASS_NAMES[predicted_class_idx], float(confidence)


# ────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BloodLens AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ────────────────────────────────────────────────────────────────────────────
# THEME / CSS
# ────────────────────────────────────────────────────────────────────────────
NAVY = "#0f2c3f"
NAVY_DARK = "#0b2130"
TEAL = "#2ec4c0"
TEAL_DARK = "#1a8f8c"
LIGHT_BG = "#f4f8fa"
CARD_BG = "#ffffff"

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,600;1,600&family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"]  {{
        font-family: 'Inter', sans-serif;
    }}

    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}

    .stApp {{
        background: {LIGHT_BG};
    }}

    /* ---- Sidebar ---- */
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {NAVY} 0%, {NAVY_DARK} 100%);
    }}
    section[data-testid="stSidebar"] * {{
        color: #dce9ee !important;
    }}
    section[data-testid="stSidebar"] .stButton button {{
        background: transparent;
        border: none;
        text-align: left;
        width: 100%;
        padding: 10px 14px;
        border-radius: 8px;
        font-weight: 500;
        color: #cfe3e9 !important;
    }}
    section[data-testid="stSidebar"] .stButton button:hover {{
        background: rgba(46, 196, 192, 0.15);
        color: #ffffff !important;
    }}

    /* ---- Generic buttons ---- */
    .stButton>button {{
        background: linear-gradient(90deg, {NAVY} 0%, {TEAL} 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6em 1.2em;
        font-weight: 600;
        transition: 0.2s;
    }}
    .stButton>button:hover {{
        opacity: 0.9;
        transform: translateY(-1px);
    }}

    /* ---- Cards ---- */
    .bl-card {{
        background: {CARD_BG};
        border-radius: 14px;
        padding: 22px 24px;
        box-shadow: 0 2px 10px rgba(15, 44, 63, 0.06);
        border: 1px solid #e8eef1;
        margin-bottom: 18px;
    }}
    .bl-hero {{
        background: linear-gradient(120deg, {NAVY} 0%, {TEAL_DARK} 120%);
        border-radius: 16px;
        padding: 32px 34px;
        color: white;
        margin-bottom: 24px;
    }}
    .bl-hero h2 {{ margin: 0 0 6px 0; }}
    .bl-hero p {{ opacity: 0.9; margin-bottom: 0; }}
    .bl-kicker {{
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-size: 0.75em;
        font-weight: 700;
        color: {TEAL};
    }}
    .bl-metric-label {{ color: #6b7d87; font-size: 0.85em; margin-bottom: 4px; }}
    .bl-metric-value {{ font-size: 2em; font-weight: 700; color: {NAVY}; }}
    .bl-badge {{
        background: #e8f7f6;
        color: {TEAL_DARK};
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75em;
        font-weight: 700;
        letter-spacing: 0.5px;
    }}
    .bl-warning {{
        background: #fff7ea;
        border: 1px solid #f3dcae;
        border-radius: 10px;
        padding: 12px 16px;
        font-size: 0.85em;
        color: #6b5610;
    }}
    .bl-title-serif {{
        font-family: 'Playfair Display', serif;
        font-weight: 600;
    }}
</style>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ────────────────────────────────────────────────────────────────────────────
defaults = {
    "authenticated": False,
    "email": "",
    "profession": "",
    "page": "Overview",
    "cbc_reports": [],   # list of dicts
    "smear_analyses": [],  # list of dicts
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def is_pathologist():
    return st.session_state.profession == "Pathologist"


# ────────────────────────────────────────────────────────────────────────────
# LOGIN PAGE
# ────────────────────────────────────────────────────────────────────────────
def login_page():
    left, right = st.columns([1.1, 1], gap="large")

    with left:
        st.markdown(f"""
        <div style="background:linear-gradient(160deg, {NAVY} 0%, {NAVY_DARK} 100%);
                    border-radius:18px; padding:48px 42px; height:640px; color:white;">
            <span class="bl-badge" style="background:rgba(255,255,255,0.1); color:#bfe9e6;">
                ● Private clinical workspace
            </span>
            <div class="bl-kicker" style="margin-top:36px;">AI-assisted screening support</div>
            <h1 style="font-size:2.6em; line-height:1.15; margin:8px 0 0 0;">
                See deeper.<br>
                <span class="bl-title-serif" style="font-style:italic; color:{TEAL};">Track smarter.</span>
            </h1>
            <p style="opacity:0.85; margin-top:18px; max-width:420px;">
                Use blood smear screening and longitudinal CBC tracking as separate
                clinical-support tools built around professional judgment.
            </p>
            <div style="display:flex; gap:40px; margin-top:70px;">
                <div>
                    <div style="font-weight:700;">Private by design</div>
                    <div style="opacity:0.75; font-size:0.85em;">Your prototype data stays on this device</div>
                </div>
                <div>
                    <div style="font-weight:700;">No invented records</div>
                    <div style="opacity:0.75; font-size:0.85em;">Charts appear only after you add data</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown(f"""
        <div style="text-align:center; margin-bottom:6px;">
            <div style="font-size:2.4em;">🔬</div>
            <div style="font-size:1.6em; font-weight:800; color:{NAVY};">
                BloodLens <span style="color:{TEAL_DARK};">AI</span>
            </div>
            <div style="font-size:0.75em; letter-spacing:2px; color:#8aa1a9;">
                SEE DEEPER · DIAGNOSE SMARTER
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="bl-card">', unsafe_allow_html=True)
        st.markdown(f'<span class="bl-kicker">Protected access</span>', unsafe_allow_html=True)
        st.markdown("### Welcome to BloodLens")
        st.caption("Sign in to your device-local clinical workspace.")

        email = st.text_input("Email address", placeholder="you@example.com")
        password = st.text_input("Password", type="password", placeholder="At least 8 characters")
        profession = st.selectbox(
            "What is your profession?",
            ["Select your profession", "Pathologist", "Hematologist", "General Physician",
             "Nurse", "Lab Technician", "Medical Student", "Other Clinician"],
        )

        if st.button("Sign in securely →", use_container_width=True):
            if not email or not password:
                st.error("Please enter an email and password.")
            elif profession == "Select your profession":
                st.error("Please select your profession to continue.")
            else:
                st.session_state.authenticated = True
                st.session_state.email = email
                st.session_state.profession = profession
                st.session_state.page = "Overview"
                st.rerun()

        st.markdown("""
        <div class="bl-warning" style="margin-top:14px;">
        <b>Prototype privacy note:</b> this password gate protects data only within this
        browser session. Production use requires a secure authentication backend,
        encrypted database, and institutional security review.
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────────────
# SIDEBAR (role-based, non-overlapping)
# ────────────────────────────────────────────────────────────────────────────
def sidebar():
    with st.sidebar:
        st.markdown(f"""
        <div style="padding: 6px 4px 18px 4px;">
            <div style="font-size:1.3em; font-weight:800; color:white;">🔬 BloodLens <span style="color:{TEAL};">AI</span></div>
        </div>
        """, unsafe_allow_html=True)

        if is_pathologist():
            nav_items = ["Overview", "Smear analysis", "Past reports", "About & safety"]
        else:
            nav_items = ["Overview", "CBC trends", "Past reports", "About & safety"]

        for item in nav_items:
            if st.button(item, key=f"nav_{item}", use_container_width=True):
                st.session_state.page = item
                st.rerun()

        st.markdown("<div style='margin-top:280px;'></div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-size:0.75em; color:{TEAL}; font-weight:700;">● Device-local mode</div>
        <div style="font-size:0.72em; opacity:0.7; margin-bottom:14px;">Records are stored only in this session.</div>
        """, unsafe_allow_html=True)

        if st.button("↩  Sign out", key="signout", use_container_width=True):
            for k in defaults:
                st.session_state[k] = defaults[k]
            st.rerun()


# ────────────────────────────────────────────────────────────────────────────
# TOP BAR
# ────────────────────────────────────────────────────────────────────────────
def topbar(title):
    c1, c2, c3 = st.columns([3, 2, 2])
    with c1:
        st.markdown(f"#### {title}")
    with c3:
        st.markdown(f"""
        <div style="text-align:right;">
            <span class="bl-badge">SECURE PROTOTYPE</span>
            &nbsp;&nbsp; <span style="color:{NAVY}; font-weight:600;">🧑‍⚕️ {st.session_state.email}</span>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("<hr style='margin-top:6px; margin-bottom:22px;'>", unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────────────
# OVERVIEW PAGE (role aware — shows only the workflow relevant to the user)
# ────────────────────────────────────────────────────────────────────────────
def overview_page():
    topbar("Overview")
    st.markdown(f'<span class="bl-kicker">Connected clinical workspace</span>', unsafe_allow_html=True)
    st.markdown(f"## Welcome, {st.session_state.email.split('@')[0]}.")
    st.caption("Your overview reflects only reports you add.")

    if is_pathologist():
        st.markdown(f"""
        <div class="bl-hero">
            <div class="bl-kicker" style="color:#bfe9e6;">Pathologist workspace</div>
            <h2>Blood smear screening</h2>
            <p>Upload de-identified microscopy images for AI-assisted screening support.
            Results are demonstration outputs only, never a diagnosis.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open smear analysis →"):
            st.session_state.page = "Smear analysis"
            st.rerun()

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="bl-card">', unsafe_allow_html=True)
            st.markdown('<div class="bl-metric-label">Smear analyses</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="bl-metric-value">{len(st.session_state.smear_analyses)}</div>', unsafe_allow_html=True)
            st.caption("Created from your uploads")
            st.markdown('</div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="bl-card">', unsafe_allow_html=True)
            st.markdown('<div class="bl-metric-label">Latest analysis</div>', unsafe_allow_html=True)
            latest = st.session_state.smear_analyses[-1]["date"] if st.session_state.smear_analyses else "—"
            st.markdown(f'<div class="bl-metric-value">{latest}</div>', unsafe_allow_html=True)
            st.caption("Updates when you add a scan")
            st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.markdown(f"""
        <div class="bl-hero">
            <div class="bl-kicker" style="color:#bfe9e6;">Clinical workspace</div>
            <h2>Longitudinal CBC tracking</h2>
            <p>Track complete blood count values over time and compare trends across visits.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open CBC tracking →"):
            st.session_state.page = "CBC trends"
            st.rerun()

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="bl-card">', unsafe_allow_html=True)
            st.markdown('<div class="bl-metric-label">Saved blood reports</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="bl-metric-value">{len(st.session_state.cbc_reports)}</div>', unsafe_allow_html=True)
            st.caption("No sample data included")
            st.markdown('</div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="bl-card">', unsafe_allow_html=True)
            st.markdown('<div class="bl-metric-label">Latest CBC date</div>', unsafe_allow_html=True)
            latest = st.session_state.cbc_reports[-1]["date"] if st.session_state.cbc_reports else "—"
            st.markdown(f'<div class="bl-metric-value">{latest}</div>', unsafe_allow_html=True)
            st.caption("Updates when you add a report")
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="bl-card">', unsafe_allow_html=True)
    st.markdown("**Need help understanding the workspace?**")
    st.caption("See what BloodLens does, what it does not do, and how your data is handled.")
    if st.button("Open safety guide →", key="safety_from_overview"):
        st.session_state.page = "About & safety"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────────────
# CBC TRENDS PAGE (non-pathologists only)
# ────────────────────────────────────────────────────────────────────────────
def cbc_page():
    topbar("CBC trends")
    st.markdown('<span class="bl-kicker">Longitudinal tracking</span>', unsafe_allow_html=True)
    st.markdown("## Build your CBC history")
    st.caption("Add results below. Nothing is pre-filled — charts populate only from what you enter.")

    with st.expander("➕ Add a new CBC report", expanded=len(st.session_state.cbc_reports) == 0):
        with st.form("cbc_form", clear_on_submit=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                rdate = st.date_input("Report date", value=date.today())
                wbc = st.number_input("WBC (×10⁹/L)", min_value=0.0, max_value=50.0, value=7.0, step=0.1)
            with c2:
                rbc = st.number_input("RBC (×10¹²/L)", min_value=0.0, max_value=8.0, value=4.8, step=0.1)
                hgb = st.number_input("Hemoglobin (g/dL)", min_value=0.0, max_value=25.0, value=13.5, step=0.1)
            with c3:
                hct = st.number_input("Hematocrit (%)", min_value=0.0, max_value=65.0, value=40.0, step=0.1)
                plt_ = st.number_input("Platelets (×10⁹/L)", min_value=0.0, max_value=900.0, value=250.0, step=1.0)

            submitted = st.form_submit_button("Save CBC report")
            if submitted:
                st.session_state.cbc_reports.append({
                    "date": rdate.isoformat(), "WBC": wbc, "RBC": rbc,
                    "Hemoglobin": hgb, "Hematocrit": hct, "Platelets": plt_,
                })
                st.success("CBC report saved.")
                st.rerun()

    if not st.session_state.cbc_reports:
        st.markdown('<div class="bl-card" style="text-align:center; color:#7c8b93;">No CBC reports yet. Add one above to see trends.</div>', unsafe_allow_html=True)
        return

    df = pd.DataFrame(st.session_state.cbc_reports).sort_values("date")
    metric = st.selectbox("Metric to chart", ["WBC", "RBC", "Hemoglobin", "Hematocrit", "Platelets"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"], y=df[metric], mode="lines+markers",
        line=dict(color=TEAL_DARK, width=3), marker=dict(size=9, color=NAVY),
        fill="tozeroy", fillcolor="rgba(46,196,192,0.12)",
    ))
    fig.update_layout(
        height=380, margin=dict(l=10, r=10, t=20, b=10),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis_title="Date", yaxis_title=metric,
    )
    st.markdown('<div class="bl-card">', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("#### Report history")
    st.dataframe(df, use_container_width=True, hide_index=True)


# ────────────────────────────────────────────────────────────────────────────
# SMEAR ANALYSIS PAGE (pathologists only)
# ────────────────────────────────────────────────────────────────────────────
def smear_page():
    topbar("Smear analysis")
    st.markdown('<span class="bl-kicker">New screening</span>', unsafe_allow_html=True)
    st.markdown("## Analyze a blood smear")
    st.caption("Upload a de-identified microscopy image. Smear analysis remains separate from CBC tracking.")

    st.markdown('<div class="bl-card">', unsafe_allow_html=True)
    uploaded = st.file_uploader("Drop a blood smear image here, or click to browse", type=["jpg", "jpeg", "png", "tiff"])
    if uploaded:
        st.image(uploaded, caption=uploaded.name, use_container_width=True)

    # Added disabled state so the user must upload an image first
    if st.button("Run demonstration analysis →", use_container_width=True, disabled=not uploaded):
        with st.spinner("Analyzing cell morphology..."):
            
            # 1. Load the interpreter
            interpreter = load_tflite_model()
            
            # 2. Run the actual inference
            predicted_class, confidence = predict_smear(uploaded, interpreter)
            
            # 3. Format the result
            analysis_result = f"Detected pattern consistent with {predicted_class} (Confidence: {confidence:.1%})."
            
            record = {
                "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "filename": uploaded.name,
                "finding": analysis_result,
            }
            st.session_state.smear_analyses.append(record)
            
        st.success(f"Analysis complete: {analysis_result}")

    st.markdown("""
    <div class="bl-warning" style="margin-top:14px;">
    <b>Not a diagnosis:</b> the image result is a local inference response until a validated
    production model API is connected. No clinical claims are generated.
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.smear_analyses:
        st.markdown("#### Recent analyses")
        st.dataframe(pd.DataFrame(st.session_state.smear_analyses), use_container_width=True, hide_index=True)


# ────────────────────────────────────────────────────────────────────────────
# PAST REPORTS PAGE (role-scoped, non-overlapping)
# ────────────────────────────────────────────────────────────────────────────
def past_reports_page():
    topbar("Past reports")
    st.markdown('<span class="bl-kicker">Your saved history</span>', unsafe_allow_html=True)
    st.markdown("## Past reports")
    st.caption("Each profession sees only its own workflow history — CBC and smear records are never mixed.")

    if is_pathologist():
        data = st.session_state.smear_analyses
        label = "smear analyses"
    else:
        data = st.session_state.cbc_reports
        label = "CBC reports"

    st.markdown('<div class="bl-card">', unsafe_allow_html=True)
    if not data:
        st.markdown(f"""
        <div style="text-align:center; padding:30px;">
            <div style="font-weight:700; font-size:1.1em;">No reports yet</div>
            <div style="color:#7c8b93; margin-top:4px;">Your saved {label} will appear here.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────────────
# ABOUT & SAFETY PAGE
# ────────────────────────────────────────────────────────────────────────────
def about_page():
    topbar("About & safety")
    st.markdown('<span class="bl-kicker">Trust & transparency</span>', unsafe_allow_html=True)
    st.markdown("## Support clinical judgment, never replace it.")
    st.caption("This is an interactive prototype. It does not currently include validated diagnostics, "
               "production authentication, or cloud medical-record storage.")

    c1, c2, c3 = st.columns(3)
    cards = [
        ("01", "Real input only", "Dashboard counts, report history, and charts begin empty and update only from information you enter."),
        ("02", "Separate workflows", "CBC tracking and smear screening remain independent tools, each with its own records and purpose."),
        ("03", "Safety boundary", "No result is a diagnosis. A production version requires validated models, secure backend authentication, encryption, auditing, and regulatory review."),
    ]
    for col, (num, title, body) in zip([c1, c2, c3], cards):
        with col:
            st.markdown(f"""
            <div class="bl-card" style="height:190px;">
                <div class="bl-kicker">{num}</div>
                <div style="font-weight:700; font-size:1.1em; margin:6px 0;">{title}</div>
                <div style="color:#6b7d87; font-size:0.88em;">{body}</div>
            </div>
            """, unsafe_allow_html=True)

    with st.expander("Are CBC and smear records connected?"):
        st.write("No. By design, CBC tracking and smear analysis are kept as fully separate "
                 "workflows and histories, matched to the reviewing professional's role.")
    with st.expander("Where is prototype data stored?"):
        st.write("In this session's memory only. Nothing is written to a persistent database "
                 "or sent to external storage.")
    with st.expander("Does BloodLens provide a diagnosis?"):
        st.write("No. All outputs are demonstration or informational support only, intended "
                 "to sit alongside — never replace — professional clinical judgment.")


# ────────────────────────────────────────────────────────────────────────────
# ROUTER
# ────────────────────────────────────────────────────────────────────────────
def main():
    if not st.session_state.authenticated:
        login_page()
        return

    # Guard: keep workflows from overlapping if role changes or a stale page is set
    allowed = (["Overview", "Smear analysis", "Past reports", "About & safety"]
               if is_pathologist() else
               ["Overview", "CBC trends", "Past reports", "About & safety"])
    if st.session_state.page not in allowed:
        st.session_state.page = "Overview"

    sidebar()

    routes = {
        "Overview": overview_page,
        "CBC trends": cbc_page,
        "Smear analysis": smear_page,
        "Past reports": past_reports_page,
        "About & safety": about_page,
    }
    routes[st.session_state.page]()


if __name__ == "__main__":
    main()
