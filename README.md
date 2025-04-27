This project focuses on identifying key cardiovascular risk factors using machine learning techniques. By analyzing a heart disease dataset, we train and evaluate a Random Forest Classifier to predict the likelihood of heart disease in individuals based on various clinical attributes.
🚀 Objective:
To develop a machine learning model that can help predict the presence of heart disease using patient data. This aids in early diagnosis and improves healthcare interventions for at-risk individuals.
🧠 Machine Learning Approach
Algorithm Used: Random Forest Classifier

Problem Type: Binary Classification

Target Variable: target (0 = No Heart Disease, 1 = Heart Disease)

📊 Dataset Overview
Source: UCI Heart Disease Dataset

Rows: ~300

Columns: 13 features + 1 target

Features Include:

age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, and target

🛠️ Libraries Used
pandas, numpy – data manipulation

matplotlib, seaborn – data visualization

sklearn – model building, training, and evaluation

🧪 Steps Performed
Data Loading and Exploration

Dataset preview, shape, and missing value check

Visualization

Correlation heatmap

Target variable distribution

Boxplot of age vs. heart disease presence

Preprocessing

Mapping and encoding of categorical values

Splitting data into training and test sets

Standardizing the features

Model Training

Using a Random Forest Classifier

Evaluation

Accuracy score

Confusion matrix

Classification report

Feature Importance

Identify which features influence the prediction most

📈 Results
Achieved high accuracy on the test data

Key features influencing heart disease predictions were identified

Confusion matrix and classification report provide insights into model performance

📷 Visualizations
Correlation Heatmap

Target Distribution

Age vs Heart Disease Boxplot

Confusion Matrix

Feature Importance Bar Chart

💡 Insights
Certain features such as cp (chest pain type), thalach (maximum heart rate), and oldpeak (ST depression) are strongly associated with heart disease.

Random Forest showed robust performance with minimal overfitting due to proper standardization and train-test split.

📌 Requirements
Install dependencies using:

bash
Copy
Edit
pip install -r requirements.txt
Example requirements.txt:

nginx
Copy
Edit
pandas
numpy
matplotlib
seaborn
scikit-learn
👩‍⚕️ Use Case
This model can serve as a foundational decision-support tool in healthcare analytics to assist practitioners in identifying at-risk individuals based on basic diagnostic attributes.
