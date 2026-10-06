import datetime
import re
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download VADER lexicon if not present
nltk.download('vader_lexicon', quiet=True)

# ---------------------------------------------------------
# 1. LOAD ARTIFACTS
# ---------------------------------------------------------
@st.cache_resource
def extract_date_features(X):
    
    dates = pd.Series(
        pd.to_datetime(
            np.asarray(X).ravel(),
            errors="coerce"
        )
    )

    return np.column_stack([
        dates.dt.year,
        dates.dt.month,
        dates.dt.day
    ])
def load_artifacts():
    # Load your trained Pipeline model (which includes ColumnTransformer)
    classifier = joblib.load(r"D:\customer_support-ai\notebooks\LinearSVC_ticket_priority.pkl")
    regressor = joblib.load(r"D:\customer_support-ai\notebooks\resolution_time_regressor.pkl")
    sia = SentimentIntensityAnalyzer()
    return classifier, regressor, sia

classifier, regressor, sia = load_artifacts()

# ---------------------------------------------------------
# 2. HELPER FUNCTIONS
# ---------------------------------------------------------
def clean_text(text: str) -> str:
    """Matches the text cleaning logic from preprocessing"""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Keep only alphabets
    text = re.sub(r'\s+', ' ', text).strip() # Collapse extra spaces
    return text

def analyze_sentiment(clean_description: str) -> float:
    """Computes compound sentiment score using VADER"""
    return sia.polarity_scores(clean_description)['compound']

# def get_satisfaction_category(rating: int) -> str:
#     """Maps numerical rating to categorical group matching training logic"""
#     if rating <= 2:
#         return "Low"
#     elif rating == 3:
#         return "Medium"
#     else:
#         return "High"

# ---------------------------------------------------------
# 3. USER INPUT FORM
# ---------------------------------------------------------
st.set_page_config(page_title="Ticket Classifier", layout="wide")
st.title("Automated Support Ticket Classification and Resolution Forecasting")
st.write("Fill in ticket details below to classify the ticket and analyze sentiment.")

with st.form("ticket_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        ticket_id = st.text_input("Ticket ID", value="TKT-2001")
        customer_name = st.text_input("Customer Name", value="John Doe")
        customer_email = st.text_input("Customer Email", value="john.doe@example.com")
        product_purchased = st.selectbox(
            "Product Purchased", 
            ["Wireless Mouse", "Laptop Pro 15", "Smartwatch V2", "Bluetooth Speaker", "Gaming Console Z"]
        )
        date_of_purchase = st.date_input("Date of Purchase", value=datetime.date.today())

    with col2:
        ticket_type = st.selectbox(
            "Ticket Type", 
            ["Technical Issue", "Billing Query", "Refund Request", "General Inquiry", "Product Feedback"]
        )
        ticket_subject = st.text_input("Ticket Subject", value="Device not connecting")
        # ticket_priority = st.selectbox("Ticket Priority", ["Low", "Medium", "High", "Critical"])
        # first_response_time = st.number_input("First Response Time (hrs)", min_value=0.0, value=2.5, step=0.1)
        # time_to_resolution = st.number_input("Time to Resolution (hrs)", min_value=0.0, value=24.0, step=0.5)
        # customer_satisfaction = st.slider("Customer Satisfaction Rating", min_value=1, max_value=5, value=3)

    ticket_description = st.text_area(
        "Ticket Description", 
        value="The Bluetooth connection drops continuously after 5 minutes of usage."
    )
    
    submit_button = st.form_submit_button("Process Ticket")

# ---------------------------------------------------------
# 4. PREPROCESSING & INFERENCE PIPELINE
# ---------------------------------------------------------
if submit_button:
    # A. Perform preprocessing transformations
    clean_desc = clean_text(ticket_description)
    clean_subj = clean_text(ticket_subject)
    sentiment_val = analyze_sentiment(clean_desc)
    # Calculate purchase_age_days relative to today
    today = datetime.date.today()
    purchase_age_days = (today - date_of_purchase).days
    # satisfaction_cat = get_satisfaction_category(customer_satisfaction)

    # B. Build full DataFrame matching features expected by ColumnTransformer
    input_data = {
        "clean_description": [clean_desc],
        "ticket_subject": [clean_subj],
        "ticket_type": [ticket_type],
        # "satisfaction_category": [satisfaction_cat],
        "date_of_purchase": [str(date_of_purchase)],
        "sentiment_score": [sentiment_val],
        "Ticket ID": [ticket_id],
        "Customer Name": [customer_name],
        "Customer Email": [customer_email],
        "Product Purchased": [product_purchased],
        # "Ticket Priority": [ticket_priority],
        # "First Response Time": [first_response_time],
        # "Time to Resolution": [time_to_resolution],
        # "Customer Satisfaction Rating": [customer_satisfaction]
    }
    df_input = pd.DataFrame(input_data)
    
    # C. Model Prediction
    try:
        # prediction = classifier.predict(df_input)
        predicted_priority = classifier.predict(df_input)[0]
# how can i give the input of priority classifier to regressor


                # Regressor output (handles log scale inversion)
        # Step 2: Inject classifier output and engineered feature into df_input for Regressor
        df_input["ticket_priority"] = predicted_priority
        df_input["purchase_age_days"] = purchase_age_days
        pred_log = regressor.predict(df_input)[0]
        predicted_time_hrs = np.expm1(pred_log)
        
        # ---------------------------------------------------------
        # 5. DISPLAY RESULTS
        # ---------------------------------------------------------
        st.subheader("Results")
        # st.success(f"**Predicted Category/Outcome:** {prediction[0]}")
        # st.info(f"**Sentiment Compound Score:** {sentiment_val:.4f}")
        st.subheader("Predictive Analytics Results")
        res_col1, res_col2, res_col3 = st.columns(3)
        
        # st.subheader("Processed Features Passed to Model")
        # st.dataframe(df_input)
        with res_col1:
            st.metric("Predicted Priority Class", str(predicted_priority))
        
        with res_col2:
            st.metric("Estimated Time to Resolution", f"{predicted_time_hrs:.1f} Hours")
            st.caption(f" Approx. {predicted_time_hrs / 24.0:.1f} Days")
            
        with res_col3:
            st.metric("VADER Sentiment Score", f"{sentiment_val:.4f}")
            
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.warning("Ensure column names and types in `df_input` match what was fitted in `preprocessor.fit()`.")
        
        
        
        