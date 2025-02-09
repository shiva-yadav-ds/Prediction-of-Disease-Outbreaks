import os
import streamlit as st
from streamlit_option_menu import option_menu
import pickle
import numpy as np

# Page Configuration
st.set_page_config(
    page_title='Clinical AI Diagnostic Suite',
    layout='centered',
    page_icon="🧬",
    initial_sidebar_state='expanded',
    menu_items={
        'About': "## AI-Powered Clinical Decision Support System\nAccurate, Reliable, and Efficient Healthcare Diagnostics"
    }
)

# Medical Theme Colors
MEDICAL_THEME = {
    "primary": "#00204D",
    "secondary": "#00B4A8",
    "accent": "#E4F3F7",
    "alert": "#CC0200",
    "background": "#FFFFFF"
}

def inject_medical_style():
    st.markdown("""
        <style>
        html { scroll-behavior: smooth; }
        body {
            background: #FFFFFF;
            color: #00204D;
            font-family: 'Inter', system-ui;
        }
        h1 {
            color:rgb(247, 247, 247) !important;
            border-bottom: 3px solid #00B4A8;
            padding-bottom: 0.5rem;
            font-weight: 700;
            font-size: 2.5rem;
        }
        [data-testid="stSidebar"] {
            background:rgb(0, 32, 77) !important;
            border-right: 1px solid rgba(255,255,255,0.1);
        }
        .stNumberInput, .stTextInput { margin-bottom: 1.5rem; }
        input[type="number"] {
            border: 2px solid #E4F3F7 !important;
            border-radius: 8px !important;
            padding: 0.75rem !important;
        }
        .stButton > button {
            background: #00B4A8 !important;
            color: white !important;
            border-radius: 10px !important;
            padding: 0.8rem 2rem !important;
            font-weight: 600 !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            border: none !important;
        }
        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 15px #00B4A840;
            background: #009C90 !important;
        }
        </style>
    """, unsafe_allow_html=True)

inject_medical_style()


@st.cache_resource
def load_clinical_models():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    models = {
        'diabetes': {
            'model_path': os.path.join(BASE_DIR, "saved_models/diabetes_model.sav"),
            'scaler_path': os.path.join(BASE_DIR, "saved_models/diabetes_scaler.sav")
        },
        'heart': {
            'model_path': os.path.join(BASE_DIR, "saved_models/heart_model.sav"),
            'scaler_path': os.path.join(BASE_DIR, "saved_models/heart_scaler.sav")
        },
        'parkinsons': {
            'model_path': os.path.join(BASE_DIR, "saved_models/parkinsons_model.sav"),
            'scaler_path': os.path.join(BASE_DIR, "saved_models/parkinsons_scaler.sav")
        }
    }

    clinical_models = {}

    def load_model(file_path):
        if not os.path.exists(file_path):
            st.error(f"Model file not found: {file_path}")
            return None
        if os.path.getsize(file_path) == 0:
            st.error(f"Model file is empty: {file_path}")
            return None
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    try:
        clinical_models['diabetes'] = {
            'model': load_model(models['diabetes']['model_path']),
            'scaler': load_model(models['diabetes']['scaler_path'])
        }
        clinical_models['heart'] = {
            'model': load_model(models['heart']['model_path']),
            'scaler': load_model(models['heart']['scaler_path'])
        }
        clinical_models['parkinsons'] = {
            'model': load_model(models['parkinsons']['model_path']),
            'scaler': load_model(models['parkinsons']['scaler_path'])
        }

    except Exception as e:
        st.error(f"Model loading error: {str(e)}")
        return None

    return clinical_models


# Initialize models
clinical_models = load_clinical_models()

# Sidebar Navigation
with st.sidebar:
    selected_diagnosis = option_menu(
        menu_title='AI Clinical Suite',
        options=['Diabetes Analysis', 'Heart Health Check', "Parkinson's Screening"],
        icons=['droplet', 'heart', 'person'],
        menu_icon='hospital',
        default_index=0,
        styles={
            "container": {"background": "#1E3A5F", "padding": "1rem", "border-radius": "10px"},
            "nav-link": {
                "color": "#DDEEEF",
                "font-size": "15px",
                "margin": "0.5rem 0",
                "border-radius": "8px"
            },
            "nav-link-selected": {
                "background": "#009688",
                "color": "white",
                "font-weight": "600"
            }
        }
    )

# Diabetes Screening Interface
if selected_diagnosis == 'Diabetes Analysis':
    st.title('🩸 Diabetes Risk Assessment')
    # Implement assessment logic here
    
    if clinical_models and clinical_models.get('diabetes'):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, step=1)
            glucose = st.number_input('Glucose Level (mg/dL)', min_value=0, max_value=200, step=1)
            blood_pressure = st.number_input('Blood Pressure (mmHg)', min_value=0, max_value=122, step=1)
            
        with col2:
            skin_thickness = st.number_input('Skin Thickness (mm)', min_value=0, max_value=99, step=1)
            insulin = st.number_input('Insulin Level (IU/mL)', min_value=0, max_value=846, step=1)
            bmi = st.number_input('BMI', min_value=0.0, max_value=67.1, step=0.1)
            
        with col3:
            dpf = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.42, step=0.01)
            age = st.number_input('Age', min_value=0, max_value=81, step=1)

        if st.button('Assess Diabetes Risk'):
            try:
                input_data = np.array([
                    pregnancies, glucose, blood_pressure, skin_thickness,
                    insulin, bmi, dpf, age
                ]).reshape(1, -1)
                
                scaled_data = clinical_models['diabetes']['scaler'].transform(input_data)
                prediction = clinical_models['diabetes']['model'].predict(scaled_data)
                
                if prediction[0] == 1:
                  st.error('⚠ High Risk of Diabetes Detected')
                  st.markdown("""
                     ### **🔬 Clinical Recommendations:**
                    - 🩸 **Schedule fasting blood glucose test**  
                    - 🍽 **Implement dietary modifications (low sugar, high fiber diet)**  
                    - 🏥 **Consult an endocrinologist for further evaluation**  
                    - 🏃‍♂️ **Adopt a structured physical activity plan**  
                    """)
                else:
                  st.success('✅ No Significant Diabetes Risk Identified')
                  st.markdown("""
                    ### **🛡 Preventive Advice:**
                    - 🏋️‍♂️ **Maintain a healthy BMI and active  lifestyle**  
                    - 🍏 **Monitor glucose levels regularly (HbA1c, FBG tests)**  
                    - 💉 **Annual preventive health checkups**  
                    - 🚰 **Stay hydrated and reduce processed sugar intake**  
                   """)

            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
    else:
        st.error('Diabetes Diagnostic Module Unavailable')



# Diabetes Screening Interface
if selected_diagnosis == 'Heart Health Check':
    st.title('❤️ Heart Disease Risk Assessment')
    # Implement assessment logic here

    if clinical_models and clinical_models.get('heart'):
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input('Age', min_value=0, max_value=120, step=1)
            sex = st.selectbox('Sex', ['Male', 'Female'])
            cp = st.selectbox('Chest Pain Type', ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'])
            trestbps = st.number_input('Resting Blood Pressure (mmHg)', min_value=0, max_value=200, step=1)
            thal = st.selectbox('Thalassemia Type', ['Normal', 'Fixed Defect', 'Reversible Defect'])
        
        with col2:
            chol = st.number_input('Serum Cholesterol (mg/dl)', min_value=100, max_value=600, step=1)
            fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['No', 'Yes'])
            restecg = st.selectbox('Resting ECG Results', ['Normal', 'ST-T Abnormality', 'Left Ventricular Hypertrophy'])
            thalach = st.number_input('Maximum Heart Rate Achieved', min_value=0, max_value=220, step=1)

        with col3:
            exang = st.selectbox('Exercise-Induced Angina', ['No', 'Yes'])
            oldpeak = st.number_input('ST Depression Induced by Exercise', min_value=0.0, max_value=10.0, step=0.0001)
            slope = st.selectbox('Slope of the Peak Exercise ST Segment', ['Upsloping', 'Flat', 'Downsloping'])
            ca = st.number_input('Number of Major Vessels Colored by Fluoroscopy', min_value=0, max_value=4, step=1)
            

        # Mapping categorical values to numerical values
        sex = 1 if sex == 'Male' else 0
        cp_mapping = {'Typical Angina': 0, 'Atypical Angina': 1, 'Non-anginal Pain': 2, 'Asymptomatic': 3}
        fbs = 1 if fbs == 'Yes' else 0
        restecg_mapping = {'Normal': 0, 'ST-T Abnormality': 1, 'Left Ventricular Hypertrophy': 2}
        exang = 1 if exang == 'Yes' else 0
        slope_mapping = {'Upsloping': 0, 'Flat': 1, 'Downsloping': 2}
        thal_mapping = {'Normal': 0, 'Fixed Defect': 1, 'Reversible Defect': 2}

        # Convert selections to numerical values
        cp = cp_mapping[cp]
        restecg = restecg_mapping[restecg]
        slope = slope_mapping[slope]
        thal = thal_mapping[thal]

        # Perform prediction when button is clicked
        if st.button('Assess Heart Disease Risk'):
            try:
                input_data = np.array([
                    age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                    exang, oldpeak, slope, ca, thal
                ]).reshape(1, -1)

                # Check if scaler exists before using it
                if 'scaler' not in clinical_models['heart'] or clinical_models['heart']['scaler'] is None:
                    st.error(" Error: Heart Disease Scaler Not Loaded")
                else:
                    scaled_data = clinical_models['heart']['scaler'].transform(input_data)

                    # Check if model exists before predicting
                    if 'model' not in clinical_models['heart'] or clinical_models['heart']['model'] is None:
                        st.error(" Error: Heart Disease Model Not Loaded")
                    else:
                        prediction = clinical_models['heart']['model'].predict(scaled_data)

                        if prediction[0] == 1:
                           st.error('⚠ High Risk of Heart Disease Detected')
                           st.markdown("""
                              ### **🩺 Urgent Medical Recommendations:**
                              - 🔬 **Schedule an ECG and lipid profile test**  
                              - 🍽 **Adopt a heart-healthy diet (low sodium, high fiber, omega-3 rich foods)**  
                              - 🏃‍♂️ **Increase daily physical activity (minimum 30 min walk)**  
                              - 💊 **Consult a cardiologist for medication and lifestyle guidance**  
                            """)
                        else:
                          st.success('✅ No Significant Heart Disease Risk Identified')
                          st.markdown("""
                             ### **🛡 Heart Health Maintenance Tips:**
                             - 🚴 **Engage in at least 150 minutes of exercise per  week**  
                             - 🥗 **Consume heart-healthy foods like nuts, fish, and leafy greens**  
                             - 🚭 **Avoid smoking and limit alcohol intake**  
                             - 🔍 **Annual cardiovascular screening for long-term wellness**  
                          """)


            except Exception as e:
                st.error(f" Prediction error: {str(e)}")

    else:
        st.error(' Heart Disease Module Unavailable')


# Parkinson's Screening Interface
if selected_diagnosis == "Parkinson's Screening":
    st.title("🧠 Parkinson's Disease Assessment")
    # Implement assessment logic here

    if clinical_models and clinical_models.get('parkinsons'):
        st.markdown("### Enter the following vocal and derived features:")
        # Distribute the 22 features across three columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            mdvp_fo = st.number_input("MDVP:Fo(Hz)", value=0.0, step=0.00001, format="%.5f")
            mdvp_fhi = st.number_input("MDVP:Fhi(Hz)", value=0.0, step=0.00001, format="%.5f")
            mdvp_flo = st.number_input("MDVP:Flo(Hz)", value=0.0, step=0.00001, format="%.5f")
            mdvp_jitter_percent = st.number_input("MDVP:Jitter(%)", value=0.0, step=0.00001, format="%.5f")
            mdvp_jitter_abs = st.number_input("MDVP:Jitter(Abs)", value=0.0, step=0.00001, format="%.5f")
            mdvp_rap = st.number_input("MDVP:RAP", value=0.0, step=0.00001, format="%.5f")
            mdvp_ppq = st.number_input("MDVP:PPQ", value=0.0, step=0.00001, format="%.5f")
            jitter_ddp = st.number_input("Jitter:DDP", value=0.0, step=0.00001, format="%.5f")
            
        with col2:
            mdvp_shimmer = st.number_input("MDVP:Shimmer", value=0.0, step=0.00001, format="%.5f")
            mdvp_shimmer_db = st.number_input("MDVP:Shimmer(dB)", value=0.0, step=0.00001, format="%.5f")
            shimmer_apq3 = st.number_input("Shimmer:APQ3", value=0.0, step=0.00001, format="%.5f")
            shimmer_apq5 = st.number_input("Shimmer:APQ5", value=0.0, step=0.00001, format="%.5f")
            mdvp_apq = st.number_input("MDVP:APQ", value=0.0, step=0.00001, format="%.5f")
            shimmer_dda = st.number_input("Shimmer:DDA", value=0.0, step=0.00001, format="%.5f")
            nhr = st.number_input("NHR", value=0.0, step=0.00001, format="%.5f")
            
        with col3:
            hnr = st.number_input("HNR", value=0.0, step=0.00001, format="%.5f")
            rpde = st.number_input("RPDE", value=0.0, step=0.00001, format="%.5f")
            dfa = st.number_input("DFA", value=0.0, step=0.00001, format="%.5f")
            spread1 = st.number_input("spread1", value=0.0, step=0.00001, format="%.5f")
            spread2 = st.number_input("spread2", value=0.0, step=0.00001, format="%.5f")
            d2 = st.number_input("D2", value=0.0, step=0.00001, format="%.5f")
            ppe = st.number_input("PPE", value=0.0, step=0.00001, format="%.5f")

        if st.button("Assess Parkinson's Risk"):
            try:
                input_data = np.array([
                    mdvp_fo, mdvp_fhi, mdvp_flo, mdvp_jitter_percent, mdvp_jitter_abs,
                    mdvp_rap, mdvp_ppq, jitter_ddp, mdvp_shimmer, mdvp_shimmer_db,
                    shimmer_apq3, shimmer_apq5, mdvp_apq, shimmer_dda, nhr,
                    hnr, rpde, dfa, spread1, spread2, d2, ppe
                ]).reshape(1, -1)
                
                scaled_data = clinical_models['parkinsons']['scaler'].transform(input_data)
                prediction = clinical_models['parkinsons']['model'].predict(scaled_data)
                
                if prediction[0] == 1:
                   st.error('⚠ High Risk of Parkinson’s Disease Detected')
                   st.markdown("""
                       ### **🩺 Immediate Recommendations:**
                       - 🧠 **Neurological assessment by a specialist**  
                       - 🏥 **Schedule MRI/CT scans for brain imaging**  
                       - 💊 **Medication evaluation for symptom management**  
                       - 🏃‍♂️ **Engage in physical therapy and motor function exercises**  
                    """)
                else:
                    st.success('✅ No Significant Parkinson’s Risk Identified')
                    st.markdown("""
                       ### **🛡 Preventive Neurological Health Tips:**
                       - 🥦 **Consume an antioxidant-rich diet (berries, nuts, green tea)**  
                       - 🧩 **Engage in cognitive exercises like puzzles and memory games**  
                       - 🏋️ **Maintain an active lifestyle with strength & balance exercises**  
                       - 🔍 **Regular neurological checkups for early detection**  
                       """)

            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
    else:
        st.error("Parkinson's Diagnostic Module Unavailable")


st.markdown("---")
footer_col1, footer_col2 = st.columns([1,3])
with footer_col1:
    st.image("https://img.icons8.com/?size=100&id=5359&format=png&color=FFFFFF", width=50)
with footer_col2:
    st.caption("""
    **Clinical AI Suite v2.1**  
    Diagnostic Decision Support System  
    For research use only | Not for clinical procedures
    """)


