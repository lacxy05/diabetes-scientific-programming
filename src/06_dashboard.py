import streamlit as st
import requests
import pandas as pd

st.set_page_config(
    page_title="DiabRiskPredictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CONFIGURATION ---
DEFAULT_API_URL = "http://127.0.0.1:8000/predict"

# --- SIDEBAR ---
with st.sidebar:
    st.title("System Controls")
    
    st.markdown("### Connection Settings")
    api_uri = st.text_input("API Endpoint", value=DEFAULT_API_URL)
    
    st.markdown("---")
    st.caption(
        """
        **Note:** This system uses an optimized 
        decision threshold derived from training data.
        """
    )
    st.info(

        """
        **Disclaimer:** This system utilizes a machine learning model to estimate risk. 
        As an AI tool, it **may produce errors** and predictions are probabilistic. 
        This is a support tool and does **not** replace professional medical advice or diagnosis.
        """
    
    )
    st.caption("Scientific Programming Final Project")
    st.caption("MSc in Health Data Science (MHEDAS)")

# --- MAIN PAGE ---
st.title("DIABETES RISK PREDICTOR")
st.markdown("### Patient clinical data entry")

# Contextual Information Expander
with st.expander("About the Data and Variables"):
    st.markdown("""
    **Dataset Context:**
    This predictive model is based on the NIDDK Pima Indians Diabetes Database. All patients in the training set are females at least 21 years old.
    
    **Variable Definitions:**
    * **Glucose:** Plasma glucose concentration a 2 hours in an oral glucose tolerance test.
    * **Blood Pressure:** Diastolic blood pressure (mm Hg).
    * **Skin Thickness:** Triceps skin fold thickness (mm).
    * **Insulin:** 2-Hour serum insulin (mu U/ml).
    * **BMI:** Body mass index (weight in kg/(height in m)^2).
    * **Diabetes Pedigree Function:** A function which scores likelihood of diabetes based on family history.
    """)

# Form to collect inputs
with st.form("prediction_form"):
    
    # MIXED LAYOUT: Sliders for standard ranges, Inputs for precise/large ranges
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Demographics")
        age = st.slider("Age (Years)", min_value=21, max_value=100, value=30, step=1)
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0, step=1)
        bmi = st.slider("BMI (kg/m²)", min_value=10.0, max_value=80.0, value=25.0, format="%.1f")

    with col2:
        st.markdown("#### Metabolic profile")
        glucose = st.number_input("Glucose (mg/dL)", min_value=40, max_value=600, value=100, step=1)
        insulin = st.number_input("Insulin (mu U/ml)", min_value=0, max_value=900, value=80, step=1)
        dpf = st.number_input(
            "Diabetes pedigree function", 
            min_value=0.0, max_value=3.0, value=0.5, step=0.01, format="%.3f",
            help="Score indicating genetic influence."
        )

    with col3:
        st.markdown("#### Vitals and skin")
        bp = st.slider("Blood pressure (mm Hg)", min_value=30, max_value=150, value=70, step=1)
        skin_thickness = st.slider("Skin thickness (mm)", min_value=5, max_value=100, value=20, step=1)

    st.markdown("---")
    submitted = st.form_submit_button("Generate diagnosis", use_container_width=True, type="primary")

# --- LOGIC & PREDICTION ---
if submitted:
    # 1. Construct Payload matching app.py schema
    input_data = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": bp,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age
    }

    # 2. Call API
    try:
        with st.spinner("Processing data..."):
            response = requests.post(api_uri, json=input_data)
        
        # 3. Handle Response
        if response.status_code == 200:
            result = response.json()
            
            # --- Extract Data from API ---
            # The API now does the heavy lifting
            prob = result["probability"]
            diagnosis_text = result["diagnosis"]          # e.g., "Diabetes"
            threshold_used = result.get("threshold_used", 0.5) # Default to 0.5 if missing
            
            # --- RESULTS SECTION ---
            st.markdown("### Diagnostic assessment")
            
            r_col1, r_col2 = st.columns([1, 2])
            
            with r_col1:
                # Color coding based on result
                if diagnosis_text == "Diabetes":
                    st.error(f"**Prediction:** {diagnosis_text}")
                else:
                    st.success(f"**Prediction:** {diagnosis_text}")
                
                st.metric("Probability Score", f"{prob*100:.1f}%")
            
            with r_col2:
                st.markdown(f"**Risk Analysis** (Optimized Threshold: `{threshold_used:.2f}`)")
                
                # Visual logic
                if prob > threshold_used:
                    st.warning(f"The calculated risk ({prob:.2f}) exceeds the clinical threshold.")
                else:
                    st.info(f"The calculated risk ({prob:.2f}) is within safe limits.")
                
                # Custom Progress Bar with Threshold Marker
                # We create a visual representation of where the probability sits vs the threshold
                st.progress(prob)
                
                # Small legend
                st.caption(f"0.0 {'&nbsp;'*10} ▲ Threshold ({threshold_used}) {'&nbsp;'*10} 1.0")

        else:
            st.error(f"Server error {response.status_code}: {response.text}")
            
    except requests.exceptions.ConnectionError:
        st.error("Connection error: Unable to reach the API. Please ensure 'app.py' is running.")