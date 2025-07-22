import streamlit as st
import pandas as pd
import joblib
import shap
import numpy as np
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import time
import os 
import json
import random
import datetime


# Define the path to the CSV using the current file location
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # Go up from /src to project root
DATA_PATH = os.path.join(BASE_DIR, "data", "student-mat.csv")
MODEL_PATH = os.path.join(BASE_DIR, "artifacts", "model.pkl")
PREPROCESSOR_PATH=os.path.join(BASE_DIR, "artifacts", "preprocessor.pkl")
# ------------------ PAGE STYLE ------------------
st.markdown("""
<style>
.stApp {
    animation: fadeIn 1s ease-in-out;
    background-color: #11151C !important;
    color: #39ff14 !important;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

section[data-testid="stSidebar"] {
    background-color: #2C2A4A !important;
    color: #F7F9F9;
    padding: 2rem 1rem;
    transition: background-color 0.4s ease-in-out;
}

section[data-testid="stSidebar"] h1, 
section[data-testid="stSidebar"] h2, 
section[data-testid="stSidebar"] h3, 
section[data-testid="stSidebar"] label, 
section[data-testid="stSidebar"] span {
    color: #F5F3FF !important;
}

section[data-testid="stSidebar"] .stSlider > div {
    color: #FF4D4D !important;
}

div.stButton > button {
    background-color: #00A896 !important;
    color: white !important;
    border: none;
    border-radius: 10px;
    padding: 0.75rem 1.5rem;
    font-size: 16px;
    font-weight: 600;
    transition: all 0.3s ease-in-out;
    box-shadow: 0px 4px 10px rgba(0, 168, 150, 0.3);
}

div.stButton > button:hover {
    background-color: #028D7E !important;
    transform: scale(1.03);
    box-shadow: 0px 6px 14px rgba(0, 168, 150, 0.5);
    cursor: pointer;
}

h2, h3 {
    color: #38B6FF !important;
}

.css-1cpxqw2 {
    color: #00A896 !important;
}

@media screen and (max-width: 768px) {
    .main {
        padding: 1rem;
    }
    div.stButton > button {
        width: 100%;
    }
}
</style>
""", unsafe_allow_html=True)

# ------------------ AUTH ------------------

USER_DATA_PATH = os.path.join(os.path.dirname(__file__), "users", "user_data.json")
def load_user_data():
    if os.path.exists(USER_DATA_PATH):
        with open(USER_DATA_PATH, "r") as f:
            return pd.DataFrame(json.load(f))
    return pd.DataFrame(columns=[
        "username", "password", "Hours Studied", "Previous Scores",
        "Sleep Hours", "Sample Question Papers Practiced", 
        "Extracurricular Activities", "StudyIntensity", "Prediction"
    ])

def save_user_prediction(username, password, input_data, prediction):
    data = input_data.copy()
    data["username"] = username
    data["password"] = password
    data["Prediction"] = prediction
    df = load_user_data()
    df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
    os.makedirs(os.path.dirname(USER_DATA_PATH), exist_ok=True)
    with open(USER_DATA_PATH, "w") as f:
        json.dump(df.to_dict(orient="records"), f, indent=4)

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.password = ""

if not st.session_state.logged_in:
    st.title("🔐 Login to Student Performance Predictor")

    login_tab, signup_tab = st.tabs(["🔑 Login", "🆕 Signup"])

    with login_tab:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            users_df = load_user_data()
            if not users_df.empty and ((users_df['username'] == username) & (users_df['password'] == password)).any():
                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.password = password
                st.success("✅ Logged in successfully!")
                st.rerun()
            else:
                st.error("❌ Invalid credentials")

    with signup_tab:
        new_username = st.text_input("New Username")
        new_password = st.text_input("New Password", type="password")
        if st.button("Signup"):
            users_df = load_user_data()
            if new_username in users_df["username"].values:
                st.warning("⚠️ Username already exists.")
            else:
                save_user_prediction(new_username, new_password, {}, 0)
                st.success("🎉 Signup successful! You can now login.")
    st.stop()

# ------------------ APP CORE ------------------

# Load the dataset
df = pd.read_csv(DATA_PATH)

# Load model and preprocessor
model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)

# Prepare background data for SHAP
background_raw = df[[
    "Hours Studied",
    "Previous Scores",
    "Sleep Hours",
    "Sample Question Papers Practiced",
    "Extracurricular Activities"
]].copy()
background_raw["StudyIntensity"] = background_raw["Hours Studied"] / (background_raw["Sleep Hours"] + 1)
background_sample = background_raw.sample(50, random_state=42)
background_processed = preprocessor.transform(background_sample)

def model_predict_transformed(X):
    return model.predict(X)

explainer = shap.Explainer(model_predict_transformed, background_processed)

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height or 500)

st.title("🎓 Student Performance Predictor with SHAP")

# Logout button
with st.sidebar:
    st.write(f"👤 Logged in as: `{st.session_state.username}`")
    if st.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.password = ""
        st.success("You’ve been logged out.")
        st.rerun()
    TIPS = [
            "📌 Review your mistakes more than your notes.",
            "⏳ Pomodoro method boosts retention.",
            "🧠 Teach a topic to learn it better.",
            "🌅 Start with hard tasks when energy is high.",
            "😴 Sleep is essential for memory consolidation.",
            "🔁 Use spaced repetition to remember more.",
            "🚫 Limit distractions to improve focus.",
            "💪 Small consistent efforts beat cramming.",
            "📝 Test yourself before the test.",
            "🧘 Take breaks to reset your focus."
        ]

    if 'tip_index' not in st.session_state:
        st.session_state.tip_index = random.randint(0, len(TIPS) - 1)
        st.session_state.last_tip_time = time.time()

     # Rotate every 10 seconds
    current=time.time() 
    if current - st.session_state.last_tip_time >=10:
        st.session_state.tip_index = (st.session_state.tip_index + 1) % len(TIPS)
        st.session_state.last_tip_time = current
        st.rerun()  # re-run app to update tip   

    st.markdown("### 💡 Tip of the Moment")   
    st.info(f"💡 {TIPS[st.session_state.tip_index]}")



st.sidebar.markdown("""
### ℹ️ About the App
This tool predicts a student's expected exam performance based on study habits and prior scores and also flags students who may need academic support.
""")

if st.sidebar.checkbox("📂 Show Original Dataset"):
    num_rows = st.sidebar.slider("Rows to view", 5, len(df), 10)
    st.subheader(f"📑 Showing first {num_rows} rows of the dataset")
    st.dataframe(df.head(num_rows))

st.subheader("📋 Enter Student Details")
col1, col2 = st.columns(2)

with col1:
    hours_studied = st.slider("📚 Hours Studied", 0.0, 10.0, 2.0)
    previous_scores = st.slider("📈 Previous Scores", 0, 100, 75)
    sleep_hours = st.slider("🛌 Sleep Hours", 0.0, 12.0, 7.0)

with col2:
    sample_papers = st.slider("📄 Sample Papers Practiced", 0, 50, 10)
    activities = st.selectbox("🏃‍♀️ Extracurricular Activities", ["yes", "no"])

study_intensity = hours_studied / (sleep_hours + 1)

user_input = {
    "Hours Studied": hours_studied,
    "Previous Scores": previous_scores,
    "Sleep Hours": sleep_hours,
    "Sample Question Papers Practiced": sample_papers,
    "Extracurricular Activities": activities,
    "StudyIntensity": study_intensity
}

input_df = pd.DataFrame([user_input])

if st.button("Predict Performance Index"):
    with st.spinner("Analyzing student's performance..."):
        time.sleep(1.5)
        X_processed = preprocessor.transform(input_df)
        prediction = model.predict(X_processed)[0]

        save_user_prediction(
            username=st.session_state.username,
            password=st.session_state.password,
            input_data=user_input,
            prediction=prediction
        )

        st.subheader(f"🎯 Predicted Performance Index: `{prediction:.2f}`")

        if prediction < 40:
            st.error("⚠️ This student may need additional support based on predicted performance.")
            st.markdown("### 📋 Suggested Interventions:")
            st.markdown("""
            - Review study materials with a mentor or teacher
            - Consider additional practice through mock tests
            - Monitor sleep and study balance
            - Encourage consistent learning schedule
            """)
        elif prediction < 50:
            st.warning("⚠️ Student may need improvement.")
        else:
            st.success("✅ This student is performing well.")

        st.subheader("📊 Feature Influence (SHAP)")
        shap_values = explainer(X_processed)
        fig, ax = plt.subplots()
        shap.plots.bar(shap_values[0], max_display=6, show=False)
        st.pyplot(fig)

        st.subheader("🔍 Interactive SHAP Force Plot")
        st_shap(shap.plots.force(shap_values[0]), height=300)

        st.subheader("📊 Your Prediction History")
        user_data = load_user_data()
        user_history = user_data[user_data['username'] == st.session_state.username]

        if not user_history.empty:
            # Prepare the display dataframe
            display_df = user_history[[
            "Hours Studied", "Previous Scores", "Sleep Hours", 
            "Sample Question Papers Practiced", 
            "Extracurricular Activities", "StudyIntensity", "Prediction"
            ]].sort_index(ascending=False).reset_index(drop=True)

            # Add serial numbers starting from 1
            display_df.index += 1
            display_df.index.name = "S.No."

            # Show the updated table
            st.dataframe(display_df)


            st.line_chart(user_history["Prediction"])

        else:
            st.info("No past predictions found.")
