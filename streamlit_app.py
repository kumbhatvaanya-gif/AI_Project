from __future__ import annotations

import hashlib
import hmac
import os
import sqlite3
import uuid
from datetime import date
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image


# --------------------------------------------------
# Configuration
# --------------------------------------------------

APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
DATABASE_PATH = DATA_DIR / "bloodlens.sqlite"
LOGO_PATH = APP_DIR / "bloodlens-logo.png"

DATA_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)

st.set_page_config(
    page_title="BloodLens AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# Styling
# --------------------------------------------------

st.markdown(
    """
    <style>
    :root {
        --navy: #102a43;
        --blue: #2878a7;
        --cyan: #35a7b8;
        --ice: #eef6f8;
        --ink: #173042;
        --muted: #667b88;
        --line: #dbe5ea;
    }

    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(circle at 82% 5%,
            rgba(53,167,184,.09), transparent 24%),
            linear-gradient(180deg, #f8fbfc, #f2f7f9);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #102a43, #153f59);
    }

    [data-testid="stSidebar"] * {
        color: #dbe9ef;
    }

    [data-testid="stSidebar"] label {
        padding: 7px 9px;
        border-radius: 10px;
    }

    [data-testid="stSidebar"] label:hover {
        background: rgba(255,255,255,.08);
    }

    h1, h2, h3 {
        color: var(--navy);
    }

    .block-container {
        max-width: 1380px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    .eyebrow {
        color: var(--cyan);
        font-size: 11px;
        font-weight: 800;
        letter-spacing: .16em;
        text-transform: uppercase;
    }

    .hero {
        position: relative;
        overflow: hidden;
        padding: 30px 34px;
        margin: 15px 0 24px;
        border-radius: 20px;
        color: white;
        background:
            linear-gradient(125deg, #153c59, #1f7594 65%, #35a7b8);
        box-shadow: 0 18px 42px rgba(19,70,96,.16);
    }

    .hero::after {
        content: "";
        position: absolute;
        width: 260px;
        height: 260px;
        right: -70px;
        top: -120px;
        border: 1px solid rgba(255,255,255,.18);
        border-radius: 50%;
        box-shadow:
            0 0 0 45px rgba(255,255,255,.045),
            0 0 0 90px rgba(255,255,255,.025);
    }

    .hero h2 {
        color: white;
        margin: 7px 0;
        font-size: 34px;
    }

    .hero p {
        max-width: 750px;
        color: #d5eaf0;
        line-height: 1.65;
    }

    .clinical-card {
        min-height: 150px;
        padding: 20px;
        background: white;
        border: 1px solid var(--line);
        border-radius: 17px;
        transition: .2s;
    }

    .clinical-card:hover {
        transform: translateY(-3px);
        border-color: #b8dae2;
        box-shadow: 0 15px 35px rgba(16,42,67,.08);
    }

    .clinical-card p {
        color: var(--muted);
        font-size: 13px;
        line-height: 1.55;
    }

    .step {
        display: inline-grid;
        place-items: center;
        width: 34px;
        height: 34px;
        background: #e3f2f5;
        color: var(--blue);
        border-radius: 50%;
        font-size: 11px;
        font-weight: 800;
    }

    .empty {
        padding: 36px 20px;
        text-align: center;
        color: var(--muted);
        background: #f8fbfc;
        border: 1px dashed #b9d2d9;
        border-radius: 15px;
    }

    .safety {
        padding: 15px 17px;
        color: #79582f;
        background: #fff7ec;
        border: 1px solid #efd7b5;
        border-radius: 11px;
        font-size: 12px;
        line-height: 1.6;
    }

    .privacy {
        padding: 14px 16px;
        color: #55717e;
        background: #edf4f6;
        border: 1px solid #d8e7eb;
        border-radius: 11px;
        font-size: 12px;
        line-height: 1.55;
    }

    .status {
        display: inline-block;
        padding: 6px 10px;
        color: #2d8476;
        background: #dff2ef;
        border-radius: 20px;
        font-size: 10px;
        font-weight: 800;
    }

    [data-testid="stMetric"] {
        padding: 16px 18px;
        background: white;
        border: 1px solid var(--line);
        border-top: 3px solid var(--cyan);
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(16,42,67,.04);
    }

    div[data-testid="stFileUploader"] {
        padding: 10px;
        background: #f4fafb;
        border: 1px dashed #a9cbd4;
        border-radius: 14px;
    }

    .stButton > button,
    .stFormSubmitButton > button {
        border-radius: 10px;
        font-weight: 700;
        transition: .2s;
    }

    .stButton > button:hover,
    .stFormSubmitButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 8px 20px rgba(40,120,167,.16);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Database
# --------------------------------------------------

@st.cache_resource
def connect_database():
    connection = sqlite3.connect(
        DATABASE_PATH,
        check_same_thread=False,
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")

    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            email TEXT NOT NULL UNIQUE COLLATE NOCASE,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS cbc_reports (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            report_date TEXT NOT NULL,
            label TEXT NOT NULL,
            notes TEXT,
            source_filename TEXT,
            wbc REAL NOT NULL,
            rbc REAL NOT NULL,
            hemoglobin REAL NOT NULL,
            hematocrit REAL,
            platelets REAL NOT NULL,
            neutrophils REAL,
            lymphocytes REAL,
            mcv REAL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id)
                REFERENCES users(id)
                ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS smear_records (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            record_date TEXT NOT NULL,
            label TEXT NOT NULL,
            original_filename TEXT NOT NULL,
            stored_filename TEXT NOT NULL,
            mime_type TEXT NOT NULL,
            size_bytes INTEGER NOT NULL,
            analysis_status TEXT NOT NULL DEFAULT 'not_configured',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id)
                REFERENCES users(id)
                ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS audit_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            action TEXT NOT NULL,
            entity_type TEXT,
            entity_id TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        """
    )

    connection.commit()
    return connection


db = connect_database()


def fetch_rows(query, parameters=()):
    result = db.execute(query, parameters).fetchall()
    return [dict(row) for row in result]


def audit(user_id, action, entity_type="", entity_id=""):
    db.execute(
        """
        INSERT INTO audit_logs
        (user_id, action, entity_type, entity_id)
        VALUES (?, ?, ?, ?)
        """,
        (user_id, action, entity_type, entity_id),
    )
    db.commit()


# --------------------------------------------------
# Password authentication
# --------------------------------------------------

def hash_password(password):
    salt = os.urandom(16)

    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode(),
        salt,
        600_000,
    )

    return f"{salt.hex()}${digest.hex()}"


def verify_password(password, stored_value):
    try:
        salt_hex, digest_hex = stored_value.split("$")

        calculated = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode(),
            bytes.fromhex(salt_hex),
            600_000,
        )

        return hmac.compare_digest(
            calculated,
            bytes.fromhex(digest_hex),
        )
    except Exception:
        return False


def create_account(email, password):
    email = email.strip().lower()

    if "@" not in email:
        return False, "Enter a valid email address."

    if len(password) < 8:
        return False, "Password must contain at least 8 characters."

    existing = db.execute(
        "SELECT id FROM users WHERE email = ?",
        (email,),
    ).fetchone()

    if existing:
        return False, "An account already exists for this email."

    user_id = str(uuid.uuid4())

    db.execute(
        """
        INSERT INTO users
        (id, email, password_hash)
        VALUES (?, ?, ?)
        """,
        (user_id, email, hash_password(password)),
    )
    db.commit()

    audit(user_id, "auth.register", "user", user_id)

    st.session_state.user = {
        "id": user_id,
        "email": email,
    }

    return True, "Account created."


def authenticate(email, password):
    user = db.execute(
        """
        SELECT id, email, password_hash
        FROM users
        WHERE email = ?
        """,
        (email.strip().lower(),),
    ).fetchone()

    if not user:
        return False

    if not verify_password(password, user["password_hash"]):
        return False

    st.session_state.user = {
        "id": user["id"],
        "email": user["email"],
    }

    audit(user["id"], "auth.login", "user", user["id"])
    return True


# --------------------------------------------------
# Data access
# --------------------------------------------------

def get_cbc_reports(user_id):
    return fetch_rows(
        """
        SELECT *
        FROM cbc_reports
        WHERE user_id = ?
        ORDER BY report_date DESC, created_at DESC
        """,
        (user_id,),
    )


def get_smear_records(user_id):
    return fetch_rows(
        """
        SELECT *
        FROM smear_records
        WHERE user_id = ?
        ORDER BY record_date DESC, created_at DESC
        """,
        (user_id,),
    )


# --------------------------------------------------
# Login page
# --------------------------------------------------

def login_page():
    left, right = st.columns([1.15, 1], gap="large")

    with left:
        st.markdown(
            """
            <div class="hero"
                 style="min-height:470px;
                        display:flex;
                        flex-direction:column;
                        justify-content:center">

                <div class="eyebrow" style="color:#8ce4eb">
                    AI-assisted screening support
                </div>

                <h2 style="font-size:56px;line-height:1">
                    See deeper.<br>Track smarter.
                </h2>

                <p>
                    Use blood smear screening and longitudinal CBC
                    tracking as separate clinical-support tools built
                    around professional judgment.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), width=310)

        st.markdown(
            '<div class="eyebrow">Protected access</div>',
            unsafe_allow_html=True,
        )

        st.header("Welcome to BloodLens")

        mode = st.radio(
            "Access mode",
            ["Sign in", "Create account"],
            horizontal=True,
        )

        with st.form("authentication_form"):
            email = st.text_input(
                "Email address",
                placeholder="name@hospital.org",
            )

            password = st.text_input(
                "Password",
                type="password",
                placeholder="At least 8 characters",
            )

            submitted = st.form_submit_button(
                mode,
                type="primary",
                use_container_width=True,
            )

        if submitted:
            if mode == "Create account":
                success, message = create_account(
                    email,
                    password,
                )

                if success:
                    st.rerun()
                else:
                    st.error(message)

            elif authenticate(email, password):
                st.rerun()

            else:
                st.error("Email or password is incorrect.")

        st.markdown(
            """
            <div class="privacy">
                <b>Prototype privacy note:</b>
                passwords are securely hashed and records are separated
                by account. Do not enter identifiable patient information
                until approved encrypted production storage is connected.
            </div>
            """,
            unsafe_allow_html=True,
        )


# --------------------------------------------------
# Dashboard
# --------------------------------------------------

def dashboard(user):
    cbc = get_cbc_reports(user["id"])
    smears = get_smear_records(user["id"])

    name = (
        user["email"]
        .split("@")[0]
        .replace(".", " ")
        .split()[0]
        .title()
    )

    st.markdown(
        '<div class="eyebrow">Clinical workspace</div>',
        unsafe_allow_html=True,
    )

    st.title(f"Welcome, {name}.")
    st.caption("Your overview reflects only records you add.")

    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow" style="color:#8ce4eb">
                Two clinical-support pathways
            </div>

            <h2>Choose the tool that fits your task.</h2>

            <p>
                Track CBC values over time or screen a smear image
                using a separate workflow. No fabricated reports,
                predictions, or confidence scores are included.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    first, second, third = st.columns(3)

    first.metric("CBC reports", len(cbc))
    second.metric("Smear records", len(smears))

    latest_date = cbc[0]["report_date"] if cbc else "—"
    third.metric("Latest CBC date", latest_date)

    st.subheader("Available workflows")

    first, second, third = st.columns(3)

    with first:
        st.markdown(
            """
            <div class="clinical-card">
                <span class="step">01</span>
                <h3>CBC tracking</h3>
                <p>
                    Enter report values and build a longitudinal
                    blood-value history.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button(
            "Open CBC tracking →",
            use_container_width=True,
        ):
            st.session_state.navigation = "CBC trends"
            st.rerun()

    with second:
        st.markdown(
            """
            <div class="clinical-card">
                <span class="step">02</span>
                <h3>Smear screening</h3>
                <p>
                    Upload a de-identified microscopy image using
                    a separate workflow.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button(
            "Open smear analysis →",
            use_container_width=True,
        ):
            st.session_state.navigation = "Smear analysis"
            st.rerun()

    with third:
        st.markdown(
            """
            <div class="clinical-card">
                <span class="step">03</span>
                <h3>Separate histories</h3>
                <p>
                    Review CBC and smear records independently.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button(
            "Open past reports →",
            use_container_width=True,
        ):
            st.session_state.navigation = "Past reports"
            st.rerun()

    st.subheader("Recent activity")

    activity = [
        {
            "Type": "CBC",
            "Date": item["report_date"],
            "Label": item["label"],
            "Created": item["created_at"],
        }
        for item in cbc
    ]

    activity.extend(
        {
            "Type": "Smear",
            "Date": item["record_date"],
            "Label": item["label"],
            "Created": item["created_at"],
        }
        for item in smears
    )

    activity.sort(
        key=lambda item: item["Created"],
        reverse=True,
    )

    if activity:
        st.dataframe(
            pd.DataFrame(activity[:5]).drop(
                columns=["Created"]
            ),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.markdown(
            """
            <div class="empty">
                <b>No reports yet</b><br>
                Add a CBC report or upload a smear image.
                Nothing is pre-filled.
            </div>
            """,
            unsafe_allow_html=True,
        )


# --------------------------------------------------
# CBC page
# --------------------------------------------------

def cbc_page(user):
    st.markdown(
        '<div class="eyebrow">Longitudinal blood data</div>',
        unsafe_allow_html=True,
    )

    st.title("CBC trends")

    st.caption(
        "Enter values exactly as shown on the laboratory report."
    )

    entry, trend = st.columns(
        [1, 1.25],
        gap="large",
    )

    with entry:
        st.subheader("Add a CBC report")

        with st.form(
            "cbc_form",
            clear_on_submit=True,
        ):
            report_date = st.date_input(
                "Report date",
                value=date.today(),
            )

            label = st.text_input(
                "Report label",
                placeholder="Annual CBC",
            )

            left, right = st.columns(2)

            wbc = left.number_input(
                "WBC (×10⁹/L)",
                min_value=0.0,
                step=0.01,
            )

            rbc = right.number_input(
                "RBC (×10¹²/L)",
                min_value=0.0,
                step=0.01,
            )

            hemoglobin = left.number_input(
                "Hemoglobin (g/dL)",
                min_value=0.0,
                step=0.1,
            )

            hematocrit = right.number_input(
                "Hematocrit (%)",
                min_value=0.0,
                step=0.1,
            )

            platelets = left.number_input(
                "Platelets (×10⁹/L)",
                min_value=0.0,
                step=1.0,
            )

            neutrophils = right.number_input(
                "Neutrophils (%)",
                min_value=0.0,
                max_value=100.0,
                step=0.1,
            )

            lymphocytes = left.number_input(
                "Lymphocytes (%)",
                min_value=0.0,
                max_value=100.0,
                step=0.1,
            )

            mcv = right.number_input(
                "MCV (fL)",
                min_value=0.0,
                step=0.1,
            )

            source = st.file_uploader(
                "Source report — optional",
                type=["pdf", "png", "jpg", "jpeg"],
            )

            notes = st.text_area(
                "Notes",
                max_chars=2000,
            )

            save = st.form_submit_button(
                "Save report and update trends",
                type="primary",
                use_container_width=True,
            )

        if save:
            report_id = str(uuid.uuid4())

            db.execute(
                """
                INSERT INTO cbc_reports (
                    id,
                    user_id,
                    report_date,
                    label,
                    notes,
                    source_filename,
                    wbc,
                    rbc,
                    hemoglobin,
                    hematocrit,
                    platelets,
                    neutrophils,
                    lymphocytes,
                    mcv
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    report_id,
                    user["id"],
                    report_date.isoformat(),
                    label.strip() or "CBC report",
                    notes.strip() or None,
                    source.name if source else None,
                    wbc,
                    rbc,
                    hemoglobin,
                    hematocrit,
                    platelets,
                    neutrophils,
                    lymphocytes,
                    mcv,
                ),
            )

            db.commit()

            audit(
                user["id"],
                "cbc.create",
                "cbc_report",
                report_id,
            )

            st.success("CBC report saved.")
            st.rerun()

    with trend:
        st.subheader("Trend summary")

        reports = list(
            reversed(
                get_cbc_reports(user["id"])
            )
        )

        if not reports:
            st.markdown(
                """
                <div class="empty">
                    <b>No trend data yet</b><br>
                    Add your first CBC report to begin a real
                    longitudinal view.
                </div>
                """,
                unsafe_allow_html=True,
            )

        else:
            frame = pd.DataFrame(reports)

            metric = st.selectbox(
                "Metric",
                [
                    "wbc",
                    "rbc",
                    "hemoglobin",
                    "platelets",
                    "hematocrit",
                    "neutrophils",
                    "lymphocytes",
                    "mcv",
                ],
                format_func=lambda value: value.title(),
            )

            figure = go.Figure()

            figure.add_trace(
                go.Scatter(
                    x=frame["report_date"],
                    y=frame[metric],
                    mode="lines+markers",
                    line={
                        "color": "#2878A7",
                        "width": 3,
                    },
                    marker={
                        "size": 9,
                        "color": "#35A7B8",
                    },
                    fill="tozeroy",
                    fillcolor="rgba(53,167,184,.08)",
                )
            )

            figure.update_layout(
                height=350,
                margin={
                    "l": 15,
                    "r": 15,
                    "t": 20,
                    "b": 15,
                },
                plot_bgcolor="white",
                paper_bgcolor="rgba(0,0,0,0)",
                xaxis_title="Report date",
                yaxis_title=metric.title(),
            )

            st.plotly_chart(
                figure,
                use_container_width=True,
            )

            st.markdown(
                """
                <div class="safety">
                    <b>Trend display only:</b>
                    values are shown exactly as entered.
                    BloodLens does not determine whether a value
                    is normal or abnormal.
                </div>
                """,
                unsafe_allow_html=True,
            )


# --------------------------------------------------
# Smear page
# --------------------------------------------------

def smear_page(user):
    st.markdown(
        '<div class="eyebrow">Independent image workflow</div>',
        unsafe_allow_html=True,
    )

    st.title("Smear analysis")

    st.caption(
        "Upload a de-identified microscopy image. "
        "CBC reports are not linked."
    )

    upload_column, guidance_column = st.columns(
        [1.3, 0.7],
        gap="large",
    )

    with upload_column:
        uploaded_file = st.file_uploader(
            "Blood smear image",
            type=[
                "jpg",
                "jpeg",
                "png",
                "tif",
                "tiff",
            ],
        )

        label = st.text_input(
            "Record label",
            value="Blood smear analysis",
        )

        valid_image = False

        if uploaded_file:
            try:
                image = Image.open(uploaded_file)

                st.image(
                    image,
                    caption=uploaded_file.name,
                    use_container_width=True,
                )

                uploaded_file.seek(0)
                valid_image = True

            except Exception:
                st.error(
                    "The selected file could not be read "
                    "as an image."
                )

        save = st.button(
            "Save smear record",
            type="primary",
            use_container_width=True,
            disabled=not valid_image,
        )

        if save:
            payload = uploaded_file.getvalue()

            if len(payload) > 20 * 1024 * 1024:
                st.error(
                    "The image exceeds the 20 MB limit."
                )

            else:
                record_id = str(uuid.uuid4())

                extension = Path(
                    uploaded_file.name
                ).suffix.lower()

                if extension not in {
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".tif",
                    ".tiff",
                }:
                    st.error("Unsupported image type.")
                    return

                stored_filename = (
                    f"{uuid.uuid4()}{extension}"
                )

                (
                    UPLOAD_DIR / stored_filename
                ).write_bytes(payload)

                db.execute(
                    """
                    INSERT INTO smear_records (
                        id,
                        user_id,
                        record_date,
                        label,
                        original_filename,
                        stored_filename,
                        mime_type,
                        size_bytes
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record_id,
                        user["id"],
                        date.today().isoformat(),
                        label.strip()
                        or "Blood smear analysis",
                        Path(
                            uploaded_file.name
                        ).name,
                        stored_filename,
                        uploaded_file.type,
                        len(payload),
                    ),
                )

                db.commit()

                audit(
                    user["id"],
                    "smear.create",
                    "smear_record",
                    record_id,
                )

                st.success(
                    "Smear record saved. "
                    "No prediction was generated."
                )

    with guidance_column:
        st.markdown(
            """
            <div class="clinical-card">
                <span class="step">01</span>
                <h3>De-identify</h3>
                <p>
                    Remove patient names, accession numbers,
                    and identifying labels before uploading.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="clinical-card" style="margin-top:16px">
                <span class="step">02</span>
                <h3>Check image quality</h3>
                <p>
                    Use even illumination, sharp focus,
                    and visible cell morphology.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="safety" style="margin-top:16px">
                <b>No diagnostic output:</b>
                a validated model is not connected, so the app
                stores the image without producing a prediction
                or confidence score.
            </div>
            """,
            unsafe_allow_html=True,
        )


# --------------------------------------------------
# Reports page
# --------------------------------------------------

def reports_page(user):
    st.markdown(
        '<div class="eyebrow">Saved history</div>',
        unsafe_allow_html=True,
    )

    st.title("Past reports")

    st.caption(
        "CBC reports and smear records remain separate."
    )

    cbc_tab, smear_tab = st.tabs(
        ["CBC reports", "Smear records"]
    )

    with cbc_tab:
        reports = get_cbc_reports(user["id"])

        if not reports:
            st.markdown(
                """
                <div class="empty">
                    <b>No CBC reports</b><br>
                    Add a report from CBC Trends.
                </div>
                """,
                unsafe_allow_html=True,
            )

        for report in reports:
            title = (
                f"{report['report_date']} · "
                f"{report['label']}"
            )

            with st.expander(title):
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "WBC": report["wbc"],
                                "RBC": report["rbc"],
                                "Hemoglobin":
                                    report["hemoglobin"],
                                "Hematocrit":
                                    report["hematocrit"],
                                "Platelets":
                                    report["platelets"],
                                "Neutrophils":
                                    report["neutrophils"],
                                "Lymphocytes":
                                    report["lymphocytes"],
                                "MCV": report["mcv"],
                            }
                        ]
                    ),
                    hide_index=True,
                    use_container_width=True,
                )

                if report["notes"]:
                    st.write(
                        "**Notes:**",
                        report["notes"],
                    )

                if st.button(
                    "Delete CBC report",
                    key=f"delete-cbc-{report['id']}",
                ):
                    db.execute(
                        """
                        DELETE FROM cbc_reports
                        WHERE id = ? AND user_id = ?
                        """,
                        (
                            report["id"],
                            user["id"],
                        ),
                    )

                    db.commit()

                    audit(
                        user["id"],
                        "cbc.delete",
                        "cbc_report",
                        report["id"],
                    )

                    st.rerun()

    with smear_tab:
        records = get_smear_records(user["id"])

        if not records:
            st.markdown(
                """
                <div class="empty">
                    <b>No smear records</b><br>
                    Upload an image from Smear Analysis.
                </div>
                """,
                unsafe_allow_html=True,
            )

        for record in records:
            title = (
                f"{record['record_date']} · "
                f"{record['label']}"
            )

            with st.expander(title):
                image_path = (
                    UPLOAD_DIR
                    / record["stored_filename"]
                )

                if image_path.exists():
                    st.image(
                        str(image_path),
                        use_container_width=True,
                    )

                st.markdown(
                    '<span class="status">'
                    'No model configured'
                    '</span>',
                    unsafe_allow_html=True,
                )

                st.caption(
                    "Original file: "
                    f"{record['original_filename']}"
                )

                if st.button(
                    "Delete smear record",
                    key=f"delete-smear-{record['id']}",
                ):
                    db.execute(
                        """
                        DELETE FROM smear_records
                        WHERE id = ? AND user_id = ?
                        """,
                        (
                            record["id"],
                            user["id"],
                        ),
                    )

                    db.commit()
                    image_path.unlink(missing_ok=True)

                    audit(
                        user["id"],
                        "smear.delete",
                        "smear_record",
                        record["id"],
                    )

                    st.rerun()


# --------------------------------------------------
# About page
# --------------------------------------------------

def about_page():
    st.markdown(
        '<div class="eyebrow">Trust and transparency</div>',
        unsafe_allow_html=True,
    )

    st.title(
        "Support clinical judgment, never replace it."
    )

    st.caption(
        "BloodLens is a clinical-support prototype, "
        "not a validated medical device."
    )

    first, second, third = st.columns(3)

    first.markdown(
        """
        <div class="clinical-card">
            <span class="step">01</span>
            <h3>Real input only</h3>
            <p>
                Counts, history, and charts begin empty
                and update only from information you enter.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    second.markdown(
        """
        <div class="clinical-card">
            <span class="step">02</span>
            <h3>Separate workflows</h3>
            <p>
                CBC tracking and smear screening remain
                independent tools with separate histories.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    third.markdown(
        """
        <div class="clinical-card">
            <span class="step">03</span>
            <h3>Safety boundary</h3>
            <p>
                No result is a diagnosis.
                Qualified clinical review remains essential.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Frequently asked questions")

    with st.expander("Where is data stored?"):
        st.write(
            "This version stores data in a local SQLite "
            "database and saves smear images locally."
        )

    with st.expander(
        "Are CBC and smear records connected?"
    ):
        st.write(
            "No. They are separate workflows and neither "
            "record type references the other."
        )

    with st.expander(
        "Does BloodLens diagnose leukemia?"
    ):
        st.write(
            "No. No validated model is connected and the "
            "app does not generate predictions or confidence "
            "scores."
        )


# --------------------------------------------------
# Application routing
# --------------------------------------------------

if "user" not in st.session_state:
    st.session_state.user = None

if not st.session_state.user:
    login_page()
    st.stop()

user = st.session_state.user

with st.sidebar:
    st.markdown("## BloodLens AI")
    st.caption("Clinical-support prototype")

    pages = [
        "Overview",
        "CBC trends",
        "Smear analysis",
        "Past reports",
        "About and safety",
    ]

    if "navigation" not in st.session_state:
        st.session_state.navigation = "Overview"

    selected_page = st.radio(
        "Navigation",
        pages,
        key="navigation",
        label_visibility="collapsed",
    )

    st.divider()
    st.caption(user["email"])

    st.markdown(
        '<span class="status">● Local database mode</span>',
        unsafe_allow_html=True,
    )

    if st.button(
        "Sign out",
        use_container_width=True,
    ):
        audit(
            user["id"],
            "auth.logout",
            "user",
            user["id"],
        )

        st.session_state.clear()
        st.rerun()

if selected_page == "Overview":
    dashboard(user)

elif selected_page == "CBC trends":
    cbc_page(user)

elif selected_page == "Smear analysis":
    smear_page(user)

elif selected_page == "Past reports":
    reports_page(user)

else:
    about_page()
