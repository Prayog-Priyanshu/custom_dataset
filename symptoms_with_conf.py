import streamlit as st
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Load the dataset
df = pd.read_csv("disease_dataset.csv")

# Treat each row as a training sample (no grouping)
symptom_df = df.copy()
symptom_df["Symptoms"] = symptom_df["Symptoms"].apply(
    lambda x: list(set(s.lower().strip() for s in x.split(", ")))
)

# Define your controlled symptom list
custom_symptoms = [
    "Fatigue", "Fever", "Chills", "Sweating", "Weight loss", "Weight gain", "Night sweats",
    "Rapid heartbeat", "High blood pressure", "High cholesterol", "Excess body weight",
    "Pale skin", "Jaundice (yellowing of skin/eyes)", "Dark urine", "Headache", "Dizziness",
    "Blurred vision", "Sensitivity to light", "Memory loss", "Confusion", "Difficulty in thinking",
    "Seizures", "Temporary unconsciousness", "Tremors", "Slow movement", "Balance issues",
    "Hallucinations", "Delusions", "Disorganized speech", "Shortness of breath", "Wheezing",
    "Chest tightness", "Cough (with or without mucus)", "Nasal congestion", "Runny nose",
    "Sneezing", "Sore throat", "Loss of taste/smell", "Ear pain", "Hearing loss",
    "Fluid drainage from ear", "Nausea", "Vomiting", "Abdominal pain", "Stomach pain",
    "Bloating", "Burning stomach pain", "Loss of appetite", "Diarrhea", "Constipation",
    "Painful urination", "Discharge", "Joint pain", "Joint stiffness", "Joint swelling",
    "Muscle aches", "Generalized pain", "Facial pain", "Rash", "Itching", "Red eyes",
    "Eye discharge", "Scaly skin patches", "Dry/cracked skin", "Sores", "Swelling (edema)",
    "Chest pain", "Palpitations (rapid heartbeat)", "Frequent urination", "Swelling (genital)",
    "Lower abdominal pain", "Persistent sadness", "Lack of energy", "Sleep issues",
    "Excessive worry", "Restlessness", "Anxiety", "Eye pain", "Itching eyes", "Discharge from eyes"
]

# Standardize list
custom_symptoms = sorted(set([s.lower().strip() for s in custom_symptoms]))

# Binarize symptoms
mlb = MultiLabelBinarizer(classes=custom_symptoms)
X = mlb.fit_transform(symptom_df["Symptoms"])
y = symptom_df["Disease"]

# Train RandomForest with more trees for stronger prediction
clf = RandomForestClassifier(random_state=42, n_estimators=200)
clf.fit(X, y)

# Streamlit UI
st.title("🧠 AI Disease Predictor from Symptoms")
st.markdown("Select the symptoms you're experiencing:")

selected = st.multiselect("Symptoms", custom_symptoms)

if st.button("Predict Disease"):
    if not selected:
        st.warning("Please select at least one symptom.")
    else:
        input_vector = mlb.transform([selected])
        prediction = clf.predict(input_vector)[0]
        probabilities = clf.predict_proba(input_vector)[0]
        confidence = np.max(probabilities) * 100

        st.success(f"🩺 Predicted Disease: **{prediction}**")
        st.markdown(f"### 🤖 AI Confidence: `{confidence:.2f}%` sure about the prediction")
