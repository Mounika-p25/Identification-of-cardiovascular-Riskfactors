This project focuses on identifying key cardiovascular risk factors using machine learning techniques. By analyzing a heart disease dataset, we train and evaluate a Random Forest Classifier to predict the likelihood of heart disease in individuals based on various clinical attributes.


**🚀 Objective:**

To develop a machine learning model that can help predict the presence of heart disease using patient data. This aids in early diagnosis and improves healthcare interventions for at-risk individuals.


**🧠 Machine Learning Approach**

  Algorithm Used: Random Forest Classifier
  
  Problem Type: Binary Classification
  
  Target Variable: target (0 = No Heart Disease, 1 = Heart Disease)

**📊 Dataset Overview**

Source: UCI Heart Disease Dataset

Rows: ~1026

Columns: 13 features + 1 target

Features Include:age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, and target

**🛠️ Libraries Used**

pandas, numpy – data manipulation

matplotlib, seaborn – data visualization

sklearn – model building, training, and evaluation

**🧪 Steps Performed**

1.Data Loading and Exploration

  ->Dataset preview, shape, and missing value check
  
2.Visualization

  ->Correlation heatmap
  
  ->Target variable distribution
  
  ->Boxplot of age vs. heart disease presence
  
3.Preprocessing

  ->Mapping and encoding of categorical values
  
  ->Splitting data into training and test sets
  
  ->Standardizing the features
  
4.Model Training

  ->Using a Random Forest Classifier
  
5.Evaluation

  ->Accuracy score
  
  ->Confusion matrix
  
  ->Classification report
  
6.Feature Importance

  ->Identify which features influence the prediction most


**📈 Results**

Achieved high accuracy on the test data

Key features influencing heart disease predictions were identified

Confusion matrix and classification report provide insights into model performance


**💡 Insights**

Certain features such as cp (chest pain type), thalach (maximum heart rate), and oldpeak (ST depression) are strongly associated with heart disease.

Random Forest showed robust performance with minimal overfitting due to proper standardization and train-test split.

**👩‍⚕️ Use Case**

This model can serve as a foundational decision-support tool in healthcare analytics to assist practitioners in identifying at-risk individuals based on basic diagnostic attributes.
