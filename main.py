# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load the dataset (make sure the file path is correct)
df = pd.read_csv('heart.csv')


# Display the first few rows of the dataset
df.head()

# Check the shape of the dataset
print(f"Dataset Shape: {df.shape}")

# Check for missing values
print("Missing Values:")
print(df.isnull().sum())

# Correlation matrix heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Heatmap")
plt.show()

# Plot distribution of the target variable (Heart Disease)
sns.countplot(x='target', data=df)
plt.title("Heart Disease Distribution")
plt.show()

# Plot age vs. target variable (Heart Disease)
plt.figure(figsize=(10, 6))
sns.boxplot(x='target', y='age', data=df)
plt.title("Age vs Heart Disease")
plt.show()


# Convert categorical features to numerical values
# For example: 'sex' and 'target' columns (if they're not already numeric)
df['sex'] = df['sex'].map({0: 'female', 1: 'male'})
df['target'] = df['target'].map({0: 'no', 1: 'yes'})

# Convert other categorical columns to numeric using one-hot encoding or label encoding if necessary
# For simplicity, let's assume the dataset columns like 'cp', 'fbs', 'restecg', etc., are already numeric
# However, if they are not, you can use pd.get_dummies() or LabelEncoder for those columns

# Define features and target variable
X = df.drop('target', axis=1)
y = df['target']

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.dtypes)
print(X_test.dtypes)

# If there are categorical columns, convert them to numeric (e.g., using get_dummies)
X_train = pd.get_dummies(X_train, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Make sure 'X_test' is correct

# Initialize the model
model = RandomForestClassifier(random_state=42)

# Train the model
model.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = model.predict(X_test_scaled)

# Accuracy score
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Disease', 'Disease'], yticklabels=['No Disease', 'Disease'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Feature importance
feature_importances = model.feature_importances_
features = X.columns

# Create a dataframe to display the feature importance
feature_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
feature_df = feature_df.sort_values(by='Importance', ascending=False)

# Visualize feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_df)
plt.title('Feature Importance')
plt.show()
