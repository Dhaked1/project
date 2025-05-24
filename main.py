import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

# Load scalers and models correctly
doctor_scaler = joblib.load("doctor_scaler.pkl")  # This should be a StandardScaler or similar
nurse_scaler = joblib.load("nurse_scaler.pkl")
staff_scaler = joblib.load("staff_scaler.pkl")
patient_scaler = joblib.load("patient_scaler.pkl")

#load cnn models

doctor_cnn = tf.keras.models.load_model("doctor_cnn_model.keras", compile=False)
nurse_cnn = tf.keras.models.load_model("nurse_cnn_model.keras", compile=False)
staff_cnn = tf.keras.models.load_model("staff_cnn_model.keras", compile=False)
patient_cnn = tf.keras.models.load_model("patient_cnn_model.keras", compile=False)

rf_model = joblib.load("combined_rf_model.pkl")

# List your exact questions (matching training column names)
doctor_questions = [ 'Doctors "always" communicated well','Doctors "always" explained things so they could understand','Doctors "always" listened carefully','Doctors "always" treated them with courtesy and  respect','Doctors "sometimes" or "never" communicated well','Doctors "sometimes" or "never" explained things so they could understand','Doctors "sometimes" or "never" listened carefully','Doctors "sometimes" or "never" treated them with courtesy and  respect','Doctors "usually"  treated them with courtesy and  respect','Doctors "usually" communicated well','Doctors "usually" explained things so they could understand','Doctors "usually" listened carefully']
nurse_questions = ['Nurses "always" communicated well',
       'Nurses "always" explained things so they could understand',
       'Nurses "always" listened carefully',
       'Nurses "always" treated them with courtesy and  respect',
       'Nurses "sometimes" or "never" communicated well',
       'Nurses "sometimes" or "never" explained things so they could understand',
       'Nurses "sometimes" or "never" listened carefully',
       'Nurses "sometimes" or "never" treated them with courtesy and  respect',
       'Nurses "usually"  treated them with courtesy and  respect',
       'Nurses "usually" communicated well',
       'Nurses "usually" explained things so they could understand',
       'Nurses "usually" listened carefully']
staff_questions = ['Patients who "Agree" that staff took their preferences into account',
       'Patients who "Disagree" or "Strongly Disagree" that staff took their preferences into account',
       'Patients who "Strongly Agree" that staff took their preferences into account',
       'Staff "always" explained', 'Staff "always" explained new medications',
       'Staff "always" explained possible side effects',
       'Staff "sometimes" or "never" explained',
       'Staff "sometimes" or "never" explained new medications',
       'Staff "sometimes" or "never" explained possible side effects',
       'Staff "usually" explained',
       'Staff "usually" explained new medications',
       'Staff "usually" explained possible side effects']
patient_questions = ['"Always" quiet at night',
       '"NO", patients would not recommend the hospital (they probably would not or definitely would not recommend it)',
       '"Sometimes" or "never" quiet at night', '"Usually" quiet at night',
       '"YES", patients would definitely recommend the hospital',
       '"YES", patients would probably recommend the hospital'
                     ]

all_questions = doctor_questions + nurse_questions + staff_questions + patient_questions

st.title("Hospital Rating Predictor")
st.markdown("### Fill the HCAHPS survey questions below:")

user_input = {}
for question in all_questions:
    user_input[question] = st.slider(f"{question}", 0, 10, 5)

if st.button("Predict Hospital Rating"):
    input_df = pd.DataFrame([user_input])

    # Extract & scale groups
    doctor_input = doctor_scaler.transform(input_df[doctor_questions].to_numpy())
    nurse_input = nurse_scaler.transform(input_df[nurse_questions].to_numpy())
    staff_input = staff_scaler.transform(input_df[staff_questions].to_numpy())
    patient_input = patient_scaler.transform(input_df[patient_questions].to_numpy())


    # Reshape for CNNs
    doctor_input = doctor_input.reshape((1, doctor_input.shape[1], 1))
    nurse_input = nurse_input.reshape((1, nurse_input.shape[1], 1))
    staff_input = staff_input.reshape((1, staff_input.shape[1], 1))
    patient_input = patient_input.reshape((1, patient_input.shape[1], 1))

    # CNN predictions
    doctor_out = doctor_cnn.predict(doctor_input)
    nurse_out = nurse_cnn.predict(nurse_input)
    staff_out = staff_cnn.predict(staff_input)
    patient_out = patient_cnn.predict(patient_input)

    print("Doctor CNN output shape:", doctor_out.shape)
    print("Nurse CNN output shape:", nurse_out.shape)
    print("Staff CNN output shape:", staff_out.shape)
    print("Patient CNN output shape:", patient_out.shape)


    # Combine CNN outputs
    combined_features = np.concatenate([doctor_out, nurse_out, staff_out, patient_out], axis=1)

    # Random Forest prediction
    final_rating = rf_model.predict(combined_features)

    st.success(f"🏥 Predicted Hospital Rating: {final_rating[0]:.2f} / 10")
