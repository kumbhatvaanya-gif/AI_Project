import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import gspread
import io
import datetime
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
from PIL import Image

# ==========================================
# 1. Configuration & Constants
# ==========================================
st.set_page_config(page_title="Leukemia Classifier", layout="wide")

MODEL_PATH = "leukemia_model.h5" # Ensure your trained model is in the repo
CLASSES = ['Benign', 'Early Pre-B', 'Pre-B', 'Pro-B']

# Google Cloud Scopes
SCOPES = [
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/drive'
]

# Google IDs (Update these from your setup)
SPREADSHEET_ID = st.secrets["gcp_service_account"]["spreadsheet_id"]
DRIVE_FOLDER_ID = st.secrets["gcp_service_account"]["drive_folder_id"]

# ==========================================
# 2. Authentication & Cloud Functions
# ==========================================
@st.cache_resource
def get_google_credentials():
    """Load credentials securely from Streamlit Secrets."""
    creds_dict = st.secrets["gcp_service_account"]
    creds = service_account.Credentials.from_service_account_info(
        creds_dict, scopes=SCOPES
    )
    return creds

def upload_to_drive(file_buffer, filename, creds):
    """Uploads an image buffer to a specific Google Drive folder."""
    try:
        service = build('drive', 'v3', credentials=creds)
        file_metadata = {
            'name': filename,
            'parents': [DRIVE_FOLDER_ID]
        }
        media = MediaIoBaseUpload(file_buffer, mimetype='image/jpeg', resumable=True)
        file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
        return file.get('id')
    except Exception as e:
        st.error(f"Drive Upload Error: {e}")
        return None

def update_spreadsheet(data_row, creds):
    """Appends a new row to the Google Sheet."""
    try:
        client = gspread.authorize(creds)
        sheet = client.open_by_key(SPREADSHEET_ID).sheet1
        sheet.append_row(data_row)
        return True
    except Exception as e:
        st.error(f"Spreadsheet Error: {e}")
        return False

# ==========================================
# 3. Model Logic
# ==========================================
@st.cache_resource
def load_classification_model():
    """Loads the CNN model. Cached so it only loads once."""
    return tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(image):
    """Converts PIL image to array and resizes for the CNN."""
    img_array = np.array(image.convert('RGB'))
    # Ensure it's BGR if your model was trained via cv2.imread
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR) 
    img_resized = cv2.resize(img_bgr, (224, 224))
    img_expanded = np.expand_dims(img_resized, axis=0)
    return img_expanded

# ==========================================
# 4. User Interface
# ==========================================
st.title("🔬 Clinical Blood Smear Analysis")
st.markdown("Upload peripheral blood smear images for ALL subtype classification.")

# --- Patient Info Section ---
with st.sidebar:
    st.header("Patient Information")
    patient_id = st.text_input("Patient ID / MRN")
    patient_age = st.number_input("Age", min_value=0, max_value=120, step=1)
    patient_gender = st.selectbox("Gender", ["Male", "Female", "Other"])

# --- Main Interaction Area ---
uploaded_file = st.file_uploader("Upload Smear Image (JPG/PNG)", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)

    with col2:
        st.subheader("Diagnostic Prediction")
        
        # Run Inference
        with st.spinner("Analyzing image..."):
            model = load_classification_model()
            processed_img = preprocess_image(image)
            predictions = model.predict(processed_img)[0]
            
            predicted_class_idx = np.argmax(predictions)
            predicted_class = CLASSES[predicted_class_idx]
            confidence = predictions[predicted_class_idx] * 100
            
        st.success(f"**Predicted Subtype:** {predicted_class}")
        st.info(f"**Model Confidence:** {confidence:.2f}%")
        
        # Display breakdown of all classes
        st.write("Score Breakdown:")
        for i, class_name in enumerate(CLASSES):
            st.progress(float(predictions[i]), text=f"{class_name}: {predictions[i]*100:.1f}%")

    # --- Feedback & Data Saving Form ---
    st.markdown("---")
    st.subheader("Physician / Researcher Review")
    
    with st.form("feedback_form"):
        is_correct = st.radio("Do you agree with the model's prediction?", ["Yes", "No", "Uncertain"])
        comments = st.text_area("Additional Clinical Notes (Optional)")
        submit_btn = st.form_submit_button("Submit & Save to Cloud")
        
        if submit_btn:
            if not patient_id:
                st.warning("Please enter a Patient ID in the sidebar before saving.")
            else:
                with st.spinner("Saving data to Google Cloud..."):
                    creds = get_google_credentials()
                    
                    # 1. Save Image to Drive
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    new_filename = f"{patient_id}_{timestamp}.jpg"
                    
                    # Reset buffer pointer and upload
                    uploaded_file.seek(0)
                    drive_file_id = upload_to_drive(uploaded_file, new_filename, creds)
                    
                    # 2. Save Data to Sheets
                    if drive_file_id:
                        row_data = [
                            timestamp,
                            patient_id,
                            patient_age,
                            patient_gender,
                            predicted_class,
                            f"{confidence:.2f}%",
                            is_correct,
                            comments,
                            drive_file_id # Store the Drive file ID for reference
                        ]
                        
                        sheet_success = update_spreadsheet(row_data, creds)
                        
                        if sheet_success:
                            st.success("✅ Data and Image successfully saved to Cloud infrastructure!")
