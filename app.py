import streamlit as st
import time
import base64 # Added specifically for PDF preview

# Configure the page
st.set_page_config(page_title="HemeTracker_AI", layout="wide")

# -----------------------------------------------------------------------------
# UI Header & Logos
# -----------------------------------------------------------------------------
try:
    st.image("HemeTracker_AI.png", use_container_width=True)
except FileNotFoundError:
    st.markdown("<h1 style='text-align: center;'>HemeTracker_AI</h1>", unsafe_allow_html=True)

st.markdown("""
**HemeTracker_AI** is an AI platform designed to assist in better medical diagnosis using advanced artificial intelligence. It is built upon a computational framework originally developed by <a href="https://www.linkedin.com/in/deeptarup-biswas-039825178/" target="_blank">Dr. Deeptarup Biswas</a>.
""", unsafe_allow_html=True)

st.markdown("---")

# -----------------------------------------------------------------------------
# Main Application Logic
# -----------------------------------------------------------------------------
st.subheader("Diagnostic Tool Interface")
st.write("Upload a PDF report or enter clinical symptoms below.")

col1, col2 = st.columns(2)

with col1:
    patient_symptoms = st.text_area(
        "Patient Symptoms / Clinical Notes",
        placeholder="Enter observed symptoms here...",
        height=150
    )

with col2:
    # Browse option to upload PDF
    uploaded_file = st.file_uploader("Upload Medical Reports", type=["pdf"])
    
    # -------------------------------------------------------------------------
    # NEW: PDF Preview Logic
    # -------------------------------------------------------------------------
    if uploaded_file is not None:
        st.write("**PDF Preview:**")
        # Read the file and convert to base64
        base64_pdf = base64.b64encode(uploaded_file.getvalue()).decode('utf-8')
        # Embed the PDF using an HTML iframe
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="300" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# AI Processing Execution
# -----------------------------------------------------------------------------
if st.button("Run HemeTracker Analysis", type="primary"):
    
    if not patient_symptoms and uploaded_file is None:
        st.warning("Please enter patient symptoms or upload a PDF report to proceed.")
        
    else:
        with st.spinner("Analyzing patient data using HemeTracker_AI models..."):
            time.sleep(2) # Simulating processing time
            
        st.success("Analysis Complete!")
