🫀 Heart Disease Prediction using Machine Learning

This repository contains a machine learning project that predicts the presence of heart disease based on clinical and physiological attributes. The project is implemented in a Jupyter Notebook and explores data preprocessing, visualization, model training, and evaluation.

📂 Project Structure
├── heart_disease_prediction.ipynb   # Main ML notebook
├── README.md                        # Project documentation
└── data/                            # Dataset (if included)

📌 Objective

The goal of this project is to build a predictive model that can classify whether a person is likely to have heart disease based on medical features. This helps support early diagnosis and preventive healthcare.

📊 Dataset

The dataset contains the following commonly used features:

Age

Sex

Chest Pain Type (cp)

Resting Blood Pressure (trestbps)

Serum Cholesterol (chol)

Fasting Blood Sugar (fbs)

Resting ECG Results (restecg)

Maximum Heart Rate Achieved (thalach)

Exercise-Induced Angina (exang)

ST Depression (oldpeak)

Slope of the ST Segment (slope)

Number of Major Vessels (ca)

Thalassemia (thal)

Target (1 = Disease, 0 = No Disease)

🛠️ Technologies Used

Python

Jupyter Notebook

Pandas

NumPy

Matplotlib & Seaborn

Scikit-Learn

🔧 Workflow
✔ 1. Data Preprocessing

Handling missing values

Scaling numerical features

Encoding categorical variables

Splitting data into training & testing sets

✔ 2. Exploratory Data Analysis (EDA)

Correlation heatmaps

Boxplots

Distribution plots

Pairplots

✔ 3. Model Development

Multiple ML algorithms are tested, such as:

Logistic Regression

K-Nearest Neighbors (KNN)

Support Vector Machine (SVM)

Random Forest

Gradient Boosting/XGBoost (if used)

✔ 4. Model Evaluation

Includes metrics like:

Accuracy

Precision, Recall, F1-Score

Confusion Matrix

ROC-AUC Score

The best-performing model is selected for predictions.

🧪 Making Predictions

Custom input prediction example:

input_data = np.array([...])
prediction = model.predict(input_data)
print("Heart Disease Detected" if prediction[0] == 1 else "No Heart Disease")

💾 Saving the Model

Model can be saved for deployment using:

import joblib
joblib.dump(model, "heart_disease_model.pkl")

🚀 How to Run the Project

Clone this repository:

git clone https://github.com/yourusername/heart-disease-prediction.git


Install required dependencies:

pip install -r requirements.txt


Open the notebook:

jupyter notebook


Run all cells in
heart_disease_prediction.ipynb

🛠️ Future Improvements

Deploy using Flask / FastAPI / Streamlit

Add SHAP or LIME explainability

Perform hyperparameter tuning

Add more advanced ensemble models

📜 License

This project is licensed under the MIT License.

👤 Sainath NAik

Feel free to reach out for contributions or suggestions!
