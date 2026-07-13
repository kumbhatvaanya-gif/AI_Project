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

from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
import re
import uuid

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
import streamlit as st


GOOGLE_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

SPREADSHEET_ID = st.secrets["google"]["spreadsheet_id"]
DRIVE_FOLDER_ID = st.secrets["google"]["drive_folder_id"]


@st.cache_resource
def get_google_services():
    service_account_info = dict(
        st.secrets["gcp_service_account"]
    )

    credentials = (
        Credentials.from_service_account_info(
            service_account_info,
            scopes=GOOGLE_SCOPES,
        )
    )

    sheets_service = build(
        "sheets",
        "v4",
        credentials=credentials,
        cache_discovery=False,
    )

    drive_service = build(
        "drive",
        "v3",
        credentials=credentials,
        cache_discovery=False,
    )

    return sheets_service, drive_service


sheets_service, drive_service = get_google_services()


def utc_timestamp():
    return datetime.now(
        timezone.utc
    ).isoformat()


def append_sheet_row(tab_name, values):
    """
    Append one row of text/metadata to Google Sheets.

    Passwords must never be included in values.
    """

    sheets_service.spreadsheets().values().append(
        spreadsheetId=SPREADSHEET_ID,
        range=f"'{tab_name}'!A:Z",
        valueInputOption="RAW",
        insertDataOption="INSERT_ROWS",
        body={"values": [values]},
    ).execute()


def get_sheet_rows(tab_name):
    result = (
        sheets_service.spreadsheets()
        .values()
        .get(
            spreadsheetId=SPREADSHEET_ID,
            range=f"'{tab_name}'!A:Z",
        )
        .execute()
    )

    values = result.get("values", [])

    if len(values) < 2:
        return []

    headers = values[0]
    records = []

    for row in values[1:]:
        padded_row = row + [""] * (
            len(headers) - len(row)
        )

        records.append(
            dict(zip(headers, padded_row))
        )

    return records


def clean_filename(filename):
    filename = Path(filename).name

    return re.sub(
        r"[^A-Za-z0-9._-]",
        "_",
        filename,
    )


def upload_image_to_drive(uploaded_file, user_id):
    """
    Upload an image to the configured private Drive folder.

    The function does not make the image public.
    """

    safe_filename = clean_filename(
        uploaded_file.name
    )

    drive_filename = (
        f"{user_id}_"
        f"{uuid.uuid4()}_"
        f"{safe_filename}"
    )

    file_bytes = uploaded_file.getvalue()

    media = MediaIoBaseUpload(
        BytesIO(file_bytes),
        mimetype=(
            uploaded_file.type
            or "application/octet-stream"
        ),
        resumable=True,
    )

    metadata = {
        "name": drive_filename,
        "parents": [DRIVE_FOLDER_ID],
    }

    result = (
        drive_service.files()
        .create(
            body=metadata,
            media_body=media,
            fields=(
                "id,name,mimeType,size,"
                "createdTime,webViewLink"
            ),
            supportsAllDrives=True,
        )
        .execute()
    )

    return result


# ==================================================
# CONFIGURATION
# ==================================================

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
DATABASE_PATH = DATA_DIR / "bloodlens.sqlite"
LOGO_PATH = APP_DIR / "bloodlens-logo.png"

DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

PAGES = [
    "Overview",
    "CBC trends",
    "Smear analysis",
    "Past reports",
    "About and safety",
]

st.set_page_config(
    page_title="BloodLens AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==================================================
# CSS
# ==================================================

st.html(
    """
    <style>
    :root {
        --navy: #102a43;
        --navy-light: #173f5f;
        --blue: #2878a7;
        --cyan: #35a7b8;
        --ice: #eef6f8;
        --white: #ffffff;
        --ink: #173042;
        --muted: #667b88;
        --line: #dbe5ea;
        --success: #2d8476;
        --warning-bg: #fff7ec;
        --warning-border: #efd7b5;
    }

    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(
                circle at 84% 4%,
                rgba(53, 167, 184, 0.10),
                transparent 25%
            ),
            linear-gradient(
                180deg,
                #f8fbfc 0%,
                #f2f7f9 100%
            );
    }

    [data-testid="stHeader"] {
        background: rgba(248, 251, 252, 0.86);
        backdrop-filter: blur(12px);
    }

    [data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #102a43 0%,
                #153f59 100%
            );
    }

    [data-testid="stSidebar"]::before {
        content: "";
        position: absolute;
        inset: 0;
        pointer-events: none;
        opacity: 0.15;
        background-image:
            linear-gradient(
                rgba(255,255,255,.08) 1px,
                transparent 1px
            ),
            linear-gradient(
                90deg,
                rgba(255,255,255,.08) 1px,
                transparent 1px
            );
        background-size: 30px 30px;
        mask-image:
            linear-gradient(
                to bottom,
                black,
                transparent 70%
            );
    }

    [data-testid="stSidebar"] * {
        color: #dceaf0;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: white !important;
    }

    .block-container {
        max-width: 1380px;
        padding-top: 2.2rem;
        padding-bottom: 4rem;
    }

    h1, h2, h3 {
        color: var(--navy);
    }

    .eyebrow {
        color: var(--cyan);
        font-size: 0.67rem;
        font-weight: 800;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
    }

    .hero {
        position: relative;
        overflow: hidden;
        padding: 2rem 2.2rem;
        margin: 1rem 0 1.4rem;
        color: white;
        background:
            linear-gradient(
                125deg,
                #153c59 0%,
                #1f7594 65%,
                #35a7b8 100%
            );
        border-radius: 20px;
        box-shadow:
            0 18px 42px
            rgba(19, 70, 96, 0.16);
    }

    .hero::after {
        content: "";
        position: absolute;
        width: 270px;
        height: 270px;
        right: -70px;
        top: -125px;
        border: 1px solid
            rgba(255,255,255,.17);
        border-radius: 50%;
        box-shadow:
            0 0 0 45px
                rgba(255,255,255,.045),
            0 0 0 90px
                rgba(255,255,255,.025);
    }

    .hero h2 {
        color: white !important;
        font-size: 2.15rem;
        margin: 0.35rem 0 0.55rem;
    }

    .hero p {
        max-width: 760px;
        color: #d4e9ef;
        line-height: 1.65;
        margin-bottom: 0;
    }

    .login-hero {
        min-height: 500px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    .login-hero h2 {
        font-size: 4rem;
        line-height: 0.98;
    }

    .clinical-card {
        min-height: 155px;
        padding: 1.3rem;
        background: white;
        border: 1px solid var(--line);
        border-radius: 17px;
        transition:
            transform .2s,
            box-shadow .2s,
            border-color .2s;
    }

    .clinical-card:hover {
        transform: translateY(-3px);
        border-color: #b8dae2;
        box-shadow:
            0 15px 35px
            rgba(16,42,67,.08);
    }

    .clinical-card h3 {
        margin: 0.75rem 0 0.4rem;
    }

    .clinical-card p {
        color: var(--muted);
        font-size: 0.82rem;
        line-height: 1.55;
        margin-bottom: 0;
    }

    .step {
        display: inline-grid;
        place-items: center;
        width: 35px;
        height: 35px;
        color: var(--blue);
        background: #e3f2f5;
        border-radius: 50%;
        font-size: 0.7rem;
        font-weight: 800;
    }

    .empty-state {
        padding: 2.3rem 1.2rem;
        color: var(--muted);
        background: #f8fbfc;
        border: 1px dashed #b9d2d9;
        border-radius: 15px;
        text-align: center;
    }

    .empty-state strong {
        color: var(--navy);
    }

    .privacy-box {
        padding: 0.9rem 1rem;
        color: #55717e;
        background: #edf4f6;
        border: 1px solid #d8e7eb;
        border-radius: 11px;
        font-size: 0.75rem;
        line-height: 1.6;
    }

    .safety-box {
        padding: 0.95rem 1rem;
        color: #79582f;
        background: var(--warning-bg);
        border: 1px solid var(--warning-border);
        border-radius: 11px;
        font-size: 0.76rem;
        line-height: 1.6;
    }

    .status-badge {
        display: inline-block;
        padding: 0.35rem 0.65rem;
        color: var(--success);
        background: #dff2ef;
        border-radius: 20px;
        font-size: 0.66rem;
        font-weight: 800;
    }

    [data-testid="stMetric"] {
        padding: 1rem 1.1rem;
        background: white;
        border: 1px solid var(--line);
        border-top: 3px solid var(--cyan);
        border-radius: 15px;
        box-shadow:
            0 8px 25px
            rgba(16,42,67,.04);
    }

    div[data-testid="stFileUploader"] {
        padding: 0.7rem;
        background: #f4fafb;
        border: 1px dashed #a9cbd4;
        border-radius: 14px;
    }

    .stButton > button,
    .stFormSubmitButton > button {
        border-radius: 10px;
        font-weight: 700;
        transition:
            transform .2s,
            box-shadow .2s;
    }

    .stButton > button:hover,
    .stFormSubmitButton > button:hover {
        transform: translateY(-1px);
        box-shadow:
            0 8px 20px
            rgba(40,120,167,.16);
    }

    @media (max-width: 700px) {
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .hero {
            padding: 1.5rem;
        }

        .login-hero h2 {
            font-size: 2.8rem;
        }
    }
    </style>
    """
)


# ==================================================
# DATABASE
# ==================================================

@st.cache_resource
def get_database():
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
            created_at TEXT NOT NULL
                DEFAULT CURRENT_TIMESTAMP
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
            created_at TEXT NOT NULL
                DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id)
                REFERENCES users(id)
                ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS
        cbc_user_date_index
        ON cbc_reports(
            user_id,
            report_date DESC
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
            analysis_status TEXT NOT NULL
                DEFAULT 'not_configured',
            created_at TEXT NOT NULL
                DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id)
                REFERENCES users(id)
                ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS
        smear_user_date_index
        ON smear_records(
            user_id,
            record_date DESC
        );

        CREATE TABLE IF NOT EXISTS audit_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            action TEXT NOT NULL,
            entity_type TEXT,
            entity_id TEXT,
            created_at TEXT NOT NULL
                DEFAULT CURRENT_TIMESTAMP
        );
        """
    )

    connection.commit()
    return connection


db = get_database()


def query_rows(query, parameters=()):
    return [
        dict(row)
        for row in db.execute(
            query,
            parameters,
        ).fetchall()
    ]


def audit(
    user_id,
    action,
    entity_type="",
    entity_id="",
):
    db.execute(
        """
        INSERT INTO audit_logs (
            user_id,
            action,
            entity_type,
            entity_id
        )
        VALUES (?, ?, ?, ?)
        """,
        (
            user_id,
            action,
            entity_type,
            entity_id,
        ),
    )

    db.commit()


# ==================================================
# PASSWORD AUTHENTICATION
# ==================================================

def hash_password(password):
    salt = os.urandom(16)

    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        600_000,
    )

    return f"{salt.hex()}${digest.hex()}"


def verify_password(
    password,
    stored_password,
):
    try:
        salt_hex, digest_hex = (
            stored_password.split("$")
        )

        calculated = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
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
        return (
            False,
            "Password must contain at least 8 characters.",
        )

    existing = db.execute(
        """
        SELECT id
        FROM users
        WHERE email = ?
        """,
        (email,),
    ).fetchone()

    if existing:
        return (
            False,
            "An account already exists for this email.",
        )

    user_id = str(uuid.uuid4())

    db.execute(
        """
        INSERT INTO users (
            id,
            email,
            password_hash
        )
        VALUES (?, ?, ?)
        """,
        (
            user_id,
            email,
            hash_password(password),
        ),
    )

    db.commit()

    audit(
        user_id,
        "auth.register",
        "user",
        user_id,
    )

    st.session_state.user = {
        "id": user_id,
        "email": email,
    }

    return True, "Account created."


def authenticate(email, password):
    user = db.execute(
        """
        SELECT
            id,
            email,
            password_hash
        FROM users
        WHERE email = ?
        """,
        (email.strip().lower(),),
    ).fetchone()

    if not user:
        return False

    if not verify_password(
        password,
        user["password_hash"],
    ):
        return False

    st.session_state.user = {
        "id": user["id"],
        "email": user["email"],
    }

    audit(
        user["id"],
        "auth.login",
        "user",
        user["id"],
    )

    return True


# ==================================================
# NAVIGATION
# ==================================================

def navigate_to(page_name):
    """
    Callback used by buttons.

    This runs before Streamlit reruns the page, so it
    avoids modifying a widget key after instantiation.
    """
    st.session_state.current_page = page_name


def sign_out(user_id):
    audit(
        user_id,
        "auth.logout",
        "user",
        user_id,
    )

    st.session_state.user = None
    st.session_state.current_page = "Overview"


# ==================================================
# DATA ACCESS
# ==================================================

def get_cbc_reports(user_id):
    return query_rows(
        """
        SELECT *
        FROM cbc_reports
        WHERE user_id = ?
        ORDER BY
            report_date DESC,
            created_at DESC
        """,
        (user_id,),
    )


def get_smear_records(user_id):
    return query_rows(
        """
        SELECT *
        FROM smear_records
        WHERE user_id = ?
        ORDER BY
            record_date DESC,
            created_at DESC
        """,
        (user_id,),
    )


# ==================================================
# LOGIN
# ==================================================

def login_page():
    left, right = st.columns(
        [1.15, 1],
        gap="large",
    )

    with left:
        st.html(
            """
            <div class="hero login-hero">
                <div
                    class="eyebrow"
                    style="color:#8ce4eb"
                >
                    AI-assisted screening support
                </div>

                <h2>
                    See deeper.<br>
                    Track smarter.
                </h2>

                <p>
                    Use blood smear screening and
                    longitudinal CBC tracking as separate
                    clinical-support tools built around
                    professional judgment.
                </p>
            </div>
            """
        )

    with right:
        if LOGO_PATH.exists():
            st.image(
                str(LOGO_PATH),
                width=310,
            )

        st.html(
            """
            <div class="eyebrow">
                Protected access
            </div>
            """
        )

        st.header("Welcome to BloodLens")

        mode = st.radio(
            "Access mode",
            [
                "Sign in",
                "Create account",
            ],
            horizontal=True,
        )

        with st.form(
            "authentication_form",
            clear_on_submit=False,
        ):
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

            else:
                if authenticate(
                    email,
                    password,
                ):
                    st.rerun()
                else:
                    st.error(
                        "Email or password is incorrect."
                    )

        st.html(
            """
            <div class="privacy-box">
                <b>Prototype privacy note:</b>
                passwords are securely hashed and records
                are separated by account. Do not enter
                identifiable patient information until
                approved encrypted production storage is
                connected.
            </div>
            """
        )


# ==================================================
# DASHBOARD
# ==================================================

def dashboard(user):
    cbc_reports = get_cbc_reports(
        user["id"]
    )

    smear_records = get_smear_records(
        user["id"]
    )

    name = (
        user["email"]
        .split("@")[0]
        .replace(".", " ")
        .replace("_", " ")
        .split()[0]
        .title()
    )

    st.html(
        """
        <div class="eyebrow">
            Clinical workspace
        </div>
        """
    )

    st.title(f"Welcome, {name}.")
    st.caption(
        "Your overview reflects only records you add."
    )

    st.html(
        """
        <div class="hero">
            <div
                class="eyebrow"
                style="color:#8ce4eb"
            >
                Two clinical-support pathways
            </div>

            <h2>
                Choose the tool that fits your task.
            </h2>

            <p>
                Track CBC values over time or screen a
                smear image using a separate workflow.
                No fabricated reports, predictions, or
                confidence scores are included.
            </p>
        </div>
        """
    )

    metric_one, metric_two, metric_three = (
        st.columns(3)
    )

    metric_one.metric(
        "CBC reports",
        len(cbc_reports),
    )

    metric_two.metric(
        "Smear records",
        len(smear_records),
    )

    metric_three.metric(
        "Latest CBC date",
        (
            cbc_reports[0]["report_date"]
            if cbc_reports
            else "—"
        ),
    )

    st.subheader("Available workflows")

    first, second, third = st.columns(3)

    with first:
        st.html(
            """
            <div class="clinical-card">
                <span class="step">01</span>

                <h3>CBC tracking</h3>

                <p>
                    Enter report values and build a
                    longitudinal blood-value history.
                </p>
            </div>
            """
        )

        st.button(
            "Open CBC tracking →",
            key="open_cbc",
            use_container_width=True,
            on_click=navigate_to,
            args=("CBC trends",),
        )

    with second:
        st.html(
            """
            <div class="clinical-card">
                <span class="step">02</span>

                <h3>Smear screening</h3>

                <p>
                    Upload a de-identified microscopy
                    image using a separate workflow.
                </p>
            </div>
            """
        )

        st.button(
            "Open smear analysis →",
            key="open_smear",
            use_container_width=True,
            on_click=navigate_to,
            args=("Smear analysis",),
        )

    with third:
        st.html(
            """
            <div class="clinical-card">
                <span class="step">03</span>

                <h3>Separate histories</h3>

                <p>
                    Review CBC and smear records
                    independently using separate tabs.
                </p>
            </div>
            """
        )

        st.button(
            "Open past reports →",
            key="open_reports",
            use_container_width=True,
            on_click=navigate_to,
            args=("Past reports",),
        )

    st.subheader("Recent activity")

    activity = [
        {
            "Type": "CBC",
            "Date": item["report_date"],
            "Label": item["label"],
            "Created": item["created_at"],
        }
        for item in cbc_reports
    ]

    activity.extend(
        {
            "Type": "Smear",
            "Date": item["record_date"],
            "Label": item["label"],
            "Created": item["created_at"],
        }
        for item in smear_records
    )

    activity.sort(
        key=lambda item: item["Created"],
        reverse=True,
    )

    if activity:
        frame = pd.DataFrame(
            activity[:5]
        ).drop(
            columns=["Created"]
        )

        st.dataframe(
            frame,
            hide_index=True,
            use_container_width=True,
        )

    else:
        st.html(
            """
            <div class="empty-state">
                <strong>No reports yet</strong>
                <br>
                Add a CBC report or upload a smear
                image. Nothing is pre-filled.
            </div>
            """
        )


# ==================================================
# CBC PAGE
# ==================================================

def cbc_page(user):
    st.html(
        """
        <div class="eyebrow">
            Longitudinal blood data
        </div>
        """
    )

    st.title("CBC trends")

    st.caption(
        "Enter values exactly as shown on the "
        "laboratory report."
    )

    entry_column, trend_column = st.columns(
        [1, 1.25],
        gap="large",
    )

    with entry_column:
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

            source_file = st.file_uploader(
                "Source report — optional",
                type=[
                    "pdf",
                    "png",
                    "jpg",
                    "jpeg",
                ],
            )

            notes = st.text_area(
                "Notes",
                max_chars=2000,
            )

            save_report = (
                st.form_submit_button(
                    "Save report and update trends",
                    type="primary",
                    use_container_width=True,
                )
            )

        if save_report:
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
                VALUES (
                    ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    report_id,
                    user["id"],
                    report_date.isoformat(),
                    (
                        label.strip()
                        or "CBC report"
                    ),
                    (
                        notes.strip()
                        or None
                    ),
                    (
                        source_file.name
                        if source_file
                        else None
                    ),
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

    with trend_column:
        st.subheader("Trend summary")

        reports = list(
            reversed(
                get_cbc_reports(
                    user["id"]
                )
            )
        )

        if not reports:
            st.html(
                """
                <div class="empty-state">
                    <strong>
                        No trend data yet
                    </strong>
                    <br>
                    Add your first CBC report to
                    begin a longitudinal view.
                </div>
                """
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
                format_func=lambda value: (
                    value.replace(
                        "_",
                        " ",
                    ).title()
                ),
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
                    fillcolor=(
                        "rgba(53,167,184,.08)"
                    ),
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
                paper_bgcolor=(
                    "rgba(0,0,0,0)"
                ),
                xaxis_title="Report date",
                yaxis_title=metric.title(),
            )

            st.plotly_chart(
                figure,
                use_container_width=True,
            )

            st.html(
                """
                <div class="safety-box">
                    <b>Trend display only:</b>
                    values are shown exactly as
                    entered. BloodLens does not
                    determine whether a value is
                    normal or abnormal.
                </div>
                """
            )


# ==================================================
# SMEAR PAGE
# ==================================================

def smear_page(user):
    st.html(
        """
        <div class="eyebrow">
            Independent image workflow
        </div>
        """
    )

    st.title("Smear analysis")

    st.caption(
        "Upload a de-identified microscopy image. "
        "CBC reports are not linked."
    )

    uploader_column, guidance_column = (
        st.columns(
            [1.3, 0.7],
            gap="large",
        )
    )

    with uploader_column:
        uploaded_file = st.file_uploader(
            "Blood smear image",
            type=[
                "jpg",
                "jpeg",
                "png",
                "tif",
                "tiff",
            ],
            key="smear_upload",
        )

        label = st.text_input(
            "Record label",
            value="Blood smear analysis",
        )

        valid_image = False

        if uploaded_file:
            try:
                image = Image.open(
                    uploaded_file
                )

                image.verify()
                uploaded_file.seek(0)

                display_image = Image.open(
                    uploaded_file
                )

                st.image(
                    display_image,
                    caption=uploaded_file.name,
                    use_container_width=True,
                )

                uploaded_file.seek(0)
                valid_image = True

            except Exception:
                st.error(
                    "The selected file could not "
                    "be read as an image."
                )

        if st.button(
            "Save smear record",
            key="save_smear",
            type="primary",
            use_container_width=True,
            disabled=not valid_image,
        ):
            payload = (
                uploaded_file.getvalue()
            )

            if len(payload) > (
                20 * 1024 * 1024
            ):
                st.error(
                    "The image exceeds the "
                    "20 MB limit."
                )

            else:
                extension = (
                    Path(
                        uploaded_file.name
                    )
                    .suffix
                    .lower()
                )

                allowed_extensions = {
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".tif",
                    ".tiff",
                }

                if extension not in (
                    allowed_extensions
                ):
                    st.error(
                        "Unsupported image type."
                    )

                else:
                    record_id = str(
                        uuid.uuid4()
                    )

                    stored_filename = (
                        f"{uuid.uuid4()}"
                        f"{extension}"
                    )

                    stored_path = (
                        UPLOAD_DIR
                        / stored_filename
                    )

                    stored_path.write_bytes(
                        payload
                    )

                    db.execute(
                        """
                        INSERT INTO
                        smear_records (
                            id,
                            user_id,
                            record_date,
                            label,
                            original_filename,
                            stored_filename,
                            mime_type,
                            size_bytes
                        )
                        VALUES (
                            ?, ?, ?, ?, ?,
                            ?, ?, ?
                        )
                        """,
                        (
                            record_id,
                            user["id"],
                            date.today()
                                .isoformat(),
                            (
                                label.strip()
                                or
                                "Blood smear analysis"
                            ),
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
                        "No prediction was "
                        "generated."
                    )

    with guidance_column:
        st.html(
            """
            <div class="clinical-card">
                <span class="step">01</span>

                <h3>De-identify</h3>

                <p>
                    Remove patient names, accession
                    numbers, and identifying labels
                    before uploading.
                </p>
            </div>
            """
        )

        st.html(
            """
            <div
                class="clinical-card"
                style="margin-top:16px"
            >
                <span class="step">02</span>

                <h3>Check image quality</h3>

                <p>
                    Use even illumination, sharp
                    focus, and visible cell
                    morphology.
                </p>
            </div>
            """
        )

        st.html(
            """
            <div
                class="safety-box"
                style="margin-top:16px"
            >
                <b>No diagnostic output:</b>
                a validated model is not connected,
                so the app stores the image without
                producing a prediction or confidence
                score.
            </div>
            """
        )


# ==================================================
# REPORTS PAGE
# ==================================================

def reports_page(user):
    st.html(
        """
        <div class="eyebrow">
            Saved history
        </div>
        """
    )

    st.title("Past reports")

    st.caption(
        "CBC reports and smear records "
        "remain separate."
    )

    cbc_tab, smear_tab = st.tabs(
        [
            "CBC reports",
            "Smear records",
        ]
    )

    with cbc_tab:
        reports = get_cbc_reports(
            user["id"]
        )

        if not reports:
            st.html(
                """
                <div class="empty-state">
                    <strong>
                        No CBC reports
                    </strong>
                    <br>
                    Add a report from CBC Trends.
                </div>
                """
            )

        for report in reports:
            title = (
                f"{report['report_date']} · "
                f"{report['label']}"
            )

            with st.expander(title):
                frame = pd.DataFrame(
                    [
                        {
                            "WBC": report["wbc"],
                            "RBC": report["rbc"],
                            "Hemoglobin": (
                                report[
                                    "hemoglobin"
                                ]
                            ),
                            "Hematocrit": (
                                report[
                                    "hematocrit"
                                ]
                            ),
                            "Platelets": (
                                report[
                                    "platelets"
                                ]
                            ),
                            "Neutrophils": (
                                report[
                                    "neutrophils"
                                ]
                            ),
                            "Lymphocytes": (
                                report[
                                    "lymphocytes"
                                ]
                            ),
                            "MCV": report["mcv"],
                        }
                    ]
                )

                st.dataframe(
                    frame,
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
                    key=(
                        "delete_cbc_"
                        f"{report['id']}"
                    ),
                ):
                    db.execute(
                        """
                        DELETE FROM
                        cbc_reports
                        WHERE
                            id = ?
                            AND user_id = ?
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
        records = get_smear_records(
            user["id"]
        )

        if not records:
            st.html(
                """
                <div class="empty-state">
                    <strong>
                        No smear records
                    </strong>
                    <br>
                    Upload an image from
                    Smear Analysis.
                </div>
                """
            )

        for record in records:
            title = (
                f"{record['record_date']} · "
                f"{record['label']}"
            )

            with st.expander(title):
                image_path = (
                    UPLOAD_DIR
                    / record[
                        "stored_filename"
                    ]
                )

                if image_path.exists():
                    st.image(
                        str(image_path),
                        use_container_width=True,
                    )

                st.html(
                    """
                    <span class="status-badge">
                        No model configured
                    </span>
                    """
                )

                st.caption(
                    "Original file: "
                    f"{record['original_filename']}"
                )

                if st.button(
                    "Delete smear record",
                    key=(
                        "delete_smear_"
                        f"{record['id']}"
                    ),
                ):
                    db.execute(
                        """
                        DELETE FROM
                        smear_records
                        WHERE
                            id = ?
                            AND user_id = ?
                        """,
                        (
                            record["id"],
                            user["id"],
                        ),
                    )

                    db.commit()

                    image_path.unlink(
                        missing_ok=True
                    )

                    audit(
                        user["id"],
                        "smear.delete",
                        "smear_record",
                        record["id"],
                    )

                    st.rerun()


# ==================================================
# ABOUT PAGE
# ==================================================

def about_page():
    st.html(
        """
        <div class="eyebrow">
            Trust and transparency
        </div>
        """
    )

    st.title(
        "Support clinical judgment, "
        "never replace it."
    )

    st.caption(
        "BloodLens is a clinical-support "
        "prototype, not a validated medical device."
    )

    first, second, third = st.columns(3)

    with first:
        st.html(
            """
            <div class="clinical-card">
                <span class="step">01</span>

                <h3>Real input only</h3>

                <p>
                    Counts, history, and charts
                    begin empty and update only
                    from information you enter.
                </p>
            </div>
            """
        )

    with second:
        st.html(
            """
            <div class="clinical-card">
                <span class="step">02</span>

                <h3>Separate workflows</h3>

                <p>
                    CBC tracking and smear
                    screening remain independent
                    tools with separate histories.
                </p>
            </div>
            """
        )

    with third:
        st.html(
            """
            <div class="clinical-card">
                <span class="step">03</span>

                <h3>Safety boundary</h3>

                <p>
                    No result is a diagnosis.
                    Qualified clinical review
                    remains essential.
                </p>
            </div>
            """
        )

    st.subheader(
        "Frequently asked questions"
    )

    with st.expander(
        "Where is data stored?"
    ):
        st.write(
            "This version stores data in a "
            "SQLite database and saves smear "
            "images to local app storage."
        )

    with st.expander(
        "Are CBC and smear records connected?"
    ):
        st.write(
            "No. They are separate workflows "
            "and neither record type references "
            "the other."
        )

    with st.expander(
        "Does BloodLens diagnose leukemia?"
    ):
        st.write(
            "No. No validated model is connected "
            "and the app does not generate "
            "predictions or confidence scores."
        )


# ==================================================
# SESSION INITIALIZATION
# ==================================================

if "user" not in st.session_state:
    st.session_state.user = None

if "current_page" not in st.session_state:
    st.session_state.current_page = "Overview"


# ==================================================
# LOGIN GATE
# ==================================================

if not st.session_state.user:
    login_page()
    st.stop()


# ==================================================
# AUTHENTICATED APPLICATION
# ==================================================

user = st.session_state.user

with st.sidebar:
    st.markdown("## BloodLens AI")
    st.caption("Clinical-support prototype")

    st.divider()

    for page_name in PAGES:
        is_active = (
            st.session_state.current_page
            == page_name
        )

        st.button(
            page_name,
            key=f"nav_{page_name}",
            type=(
                "primary"
                if is_active
                else "secondary"
            ),
            use_container_width=True,
            on_click=navigate_to,
            args=(page_name,),
        )

    st.divider()

    st.caption(user["email"])

    st.html(
        """
        <span class="status-badge">
            ● Local database mode
        </span>
        """
    )

    st.button(
        "Sign out",
        key="sign_out",
        use_container_width=True,
        on_click=sign_out,
        args=(user["id"],),
    )


# ==================================================
# PAGE ROUTER
# ==================================================

current_page = st.session_state.current_page

if current_page == "Overview":
    dashboard(user)

elif current_page == "CBC trends":
    cbc_page(user)

elif current_page == "Smear analysis":
    smear_page(user)

elif current_page == "Past reports":
    reports_page(user)

else:
    about_page()
