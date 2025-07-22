# 📚 PrepWise


Welcome to **PrepWise** — your intelligent academic companion. This AI-powered web application is designed to predict students' exam scores as well as assist them in understanding and improving their study outcomes through predictive insights and behavioral analysis.


---


## 🚀 Overview

 **PrepWise** is a machine learning–powered Streamlit app that blends academic metrics, personal habits, and interactive visuals into a unified interface that encourages better study decisions, not just score predictions. This isn’t just a predictor — it’s a **study coach** that gives you:

- 📊 AI-based score predictions based on your study behavior
- 🧠 Smart suggestions to refine your preparation strategy
- 💡 Animated micro-learning tips to keep you inspired and informed
- 📁 Auto-logged prediction history to reflect on progress over time
---

## 🧰 Features & Impact

| Feature                               | Description                                                                 | Impact                                                                 |
|---------------------------------------|-----------------------------------------------------------------------------|------------------------------------------------------------------------|
| 🔐 **Login System**                   | Simple session-based login and registration.                               | Creates a secure, user-specific experience.                           |
| 🎯 **Performance Prediction**         | Predicts final scores based on study habits, sleep, past scores, etc.      | Helps students reflect and adapt behavior.                           |
| 📊 **Trend Visualization**           | Line charts of score predictions over time.                                | Visually tracks academic journey.                                     |
| 💾 **Input History & Logging**        | All inputs are stored in `user_data.json`.                                 | Provides continuity and accountability.                              |
| 🔄 **Study Intensity Metric**         | A derived metric from effort vs. outcome.                                  | Encourages efficient, not just hard, work.                            |
| 💡 **Micro-Learning Tips**            | Shows study tips every 10 seconds in the sidebar.                          | Offers continuous motivation and productivity hacks.                  |
| 📁 **Secure Data Storage**            | Each user’s data is saved in structured JSON format.                       | Ensures privacy, traceability, and reuse.                            |

---

## 📁 Folder Structure

```plaintext
PrepWise/
│
├── artifacts/                       # Saved ML artifacts
│   ├── model.pkl                    # Trained Random Forest model
│   └── preprocessor.pkl             # ColumnTransformer pipeline
│
├── data/
│   └── student-mat.csv             # Dataset (UCI Student Performance)
│
├── notebooks/
│   └── eda.ipynb                   # EDA and visualization notebook
│
├── src/
│   ├── users/
│   │   └── user_data.json          # Stores user credentials & inputs
│   ├── app.py                      # Streamlit web app
│   ├── data_prep.py                # Data cleaning & preprocessing
│   └── train.py                    # Model training logic
│
├── requirements.txt                # Python dependencies
└── README.md                       # Project overview and documentation

```
---
## ⚙️ Technology Stack

| Tool/Library     | Purpose                                                                |
| ---------------- | ---------------------------------------------------------------------- |
| **Python**       | Core programming language                                              |
| **Streamlit**    | Interactive UI for real-time prediction & visualization                |
| **scikit-learn** | ML model training, pipelines, and evaluation                           |
| **pandas**       | Data manipulation and preparation of structured tabular datasets       |
| **numpy**        | Efficient numerical operations and transformations                     |
| **seaborn**      | Beautiful statistical visualizations for EDA                           |
| **SHAP**         | Explaining model predictions with SHAP values (model interpretability) |
| **joblib**       | Model and pipeline serialization (model.pkl, preprocessor.pkl)         |
| **json**         | Lightweight format to store user login data and prediction history     |

---

## 🔑 User Data Management

Every user registers and logs in with a unique **username** and **password**. All associated data is stored in:
```
src/users/user_data.json
```

This includes:
- 🔐 Secure login credentials
- 📊 All historical inputs related to predictions (study hours, sleep, scores, etc.)

The application automatically filters history **based on the active user**, ensuring each student sees only their own personalized performance data.

---

## 🧠 Machine Learning Model

- **Model Used:** `RandomForestRegressor`
- **Preprocessing:** Handled via `ColumnTransformer`, serialized in `preprocessor.pkl`
- **Training Script:** Accessible at:  
```
src/train.py
```

- **Input Features:**
```- 📚 Hours Studied  
- 😴 Sleep Hours  
- 📈 Previous Scores  
- 📝 Sample Paper Practice Count  
- 🏃 Extracurricular Activities  
```
- **Explainability:**  
SHAP (SHapley Additive exPlanations) was used during development to visualize and explain the model’s decision-making process — enhancing trust and interpretability.

---

## 📈 Prediction History

Every prediction session (based on inputs like hours studied, sleep, practice, etc.) is:
- ✅ Automatically saved to the user’s data profile
- 📑 Displayed in a **sortable table** within the main interface
- 📉 Plotted as a **dynamic line chart** to visualize prediction trends over time

This feature enables students to:
- Track academic behavior patterns
- Identify what strategies work
- Continuously iterate and improve their preparation approach

---

## 🧪 Sample Input & Output

| Hours Studied | Sleep | Previous Scores | Extra Activities | Practice Papers | Predicted Score |
|---------------|-------|------------------|------------------|------------------|------------------|
| 2             | 8     | 77               | Yes              | 10               | 82.3             |
| 2             | 6     | 60               | No               | 5                | 68.5             |

---

## 🧾 Requirements
  Contents of `requirements.txt`:
```
pandas
numpy
matplotlib
seaborn
scikit-learn
streamlit
joblib
shap
```
---

## 🚀 How to Run the Project

To get started with PrepWise on your local machine, follow the steps below:

1. **Clone the Repository**

   First, clone the project repository to your machine using your preferred method (e.g., GitHub Desktop or terminal Git commands).

2. **Navigate into the Project Directory**

   ```bash
   cd PrepWise

3. **📥 Download Model Files**

Due to GitHub’s file size limit , the trained machine learning assets are hosted externally.

Please download the following files manually and place them inside the `artifacts/` directory:

- [📦 Download `model.pkl`](https://drive.google.com/file/d/1KRlcmpX_XC5gLpjUnESqytZDSNABM9N_/view?usp=drivesdk)
- [⚙️ Download `preprocessor.pkl`](https://drive.google.com/file/d/1MMtU-PIc7p57z_aGOMtf4Iry2G7CGFr3/view?usp=drivesdk)


After downloading, ensure your folder structure looks like this:
```
PrepWise/
├── artifacts/
│ ├── model.pkl
│ └── preprocessor.pkl
├── src/
│ └── ...
├── app.py
└── README.md
```


4. **Create a Virtual Environment (Optional but Recommended)**
```
python -m venv venv
source venv/bin/activate        # On Linux/macOS
venv\Scripts\activate           # On Windows
```
5. **Install All Dependencies**
```
pip install -r requirements.txt
```


6. **Run the Streamlit Application**

  Launch the app in your default browser using:
```
streamlit run app.py
```
💡 The application will start in your default web browser at `http://localhost:8501`.


7. **Login or Register**

- Once the app is running, either **log in with existing credentials** or **register a new user**.

- All credentials and prediction history are securely stored in the `src/users/user_data.json` file.
---

## 📚 Acknowledgements

- **📊 UCI Machine Learning Repository**  
_Student Performance Dataset_

- **💡 Streamlit**  
For the seamless and interactive app interface

- **🧪 scikit-learn**  
For model training, pipelines, and preprocessing utilities

- **📉 SHAP**  
For powering model explainability and interpretability

---

## 💬 Final Thought

> _“The best way to predict your future is to create it.”_

This app empowers students to **create their future** —  
One study session, one insight, and one prediction at a time.

---


## ✍️ Author
  
Crafted with insight, intent, and a spark of innovation by:

**Debargha Karmakar**  
- 📧 debarghakarmakar853@gmail.com 
- 🌐 https://www.linkedin.com/in/debargha-karmakar-680a77357

🎓 *Driven by curiosity. Shaped by data. Inspired by learning.*

If you found this project helpful or inspiring, feel free to ⭐️ the repo or connect!
> _“Code is poetry when it meets purpose.”_
---








