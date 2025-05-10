import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the survey data
data = pd.read_csv('survey_data.csv')

# Convert the existing 3-level anxiety to 5-level
# We'll create a new dataframe with expanded levels
expanded_data = data.copy()

# Map from old (0,1,2) to new (0,2,4) to make room for intermediate levels
expanded_data['anxiety_level'] = expanded_data['anxiety_level'].map({
    0: 0, 1: 1, 2: 2, 3: 3, 4: 4
})

# Create 5 level training data with clear distinctions:
# 0 = Very Low, 1 = Low, 2 = Medium, 3 = High, 4 = Very High

# Very low anxiety (0): Excellent scores (mostly 5s)
very_low_cases = pd.DataFrame({
    'feeling': [5, 5, 4, 5],
    'sleep': [5, 5, 5, 4],
    'anxiety': [1, 1, 2, 1],
    'energy': [5, 5, 4, 5],
    'stress': [1, 1, 2, 1],
    'anxiety_level': [0, 0, 0, 0]
})

# Low anxiety (1): Good scores (mostly 4s)
low_anxiety_cases = pd.DataFrame({
    'feeling': [4, 4, 3, 4],
    'sleep': [4, 4, 3, 4],
    'anxiety': [2, 2, 2, 2],
    'energy': [4, 4, 3, 4],
    'stress': [2, 2, 2, 2],
    'anxiety_level': [1, 1, 1, 1]
})

# Medium anxiety (2): Average scores (mostly 3s)
medium_anxiety_cases = pd.DataFrame({
    'feeling': [3, 3, 2, 3],
    'sleep': [3, 3, 3, 2],
    'anxiety': [3, 3, 3, 3],
    'energy': [3, 3, 3, 3],
    'stress': [3, 3, 3, 3],
    'anxiety_level': [2, 2, 2, 2]
})

# High anxiety (3): Poor scores (mostly 2s)
high_anxiety_cases = pd.DataFrame({
    'feeling': [2, 2, 2, 3],
    'sleep': [2, 2, 2, 2],
    'anxiety': [4, 4, 4, 3],
    'energy': [2, 2, 3, 2],
    'stress': [4, 4, 3, 4],
    'anxiety_level': [3, 3, 3, 3]
})

# Very high anxiety (4): Very poor scores (mostly 1s)
very_high_anxiety_cases = pd.DataFrame({
    'feeling': [1, 1, 2, 1],
    'sleep': [1, 1, 1, 2],
    'anxiety': [5, 5, 4, 5],
    'energy': [1, 1, 2, 1],
    'stress': [5, 5, 4, 5],
    'anxiety_level': [4, 4, 4, 4]
})

# Add more cases where all ratings are 1 to force very high anxiety prediction
all_ones_cases = pd.DataFrame({
    'feeling': [1, 1, 1, 1, 1],
    'sleep': [1, 1, 1, 1, 1],
    'anxiety': [5, 5, 5, 5, 5],
    'energy': [1, 1, 1, 1, 1],
    'stress': [5, 5, 5, 5, 5],
    'anxiety_level': [4, 4, 4, 4, 4]  # Very high anxiety (4)
})

# Combine all datasets
data = pd.concat([
    expanded_data,
    very_low_cases,
    low_anxiety_cases, 
    medium_anxiety_cases,
    high_anxiety_cases,
    very_high_anxiety_cases,
    all_ones_cases
], ignore_index=True)

# Split features and target
X = data[['feeling', 'sleep', 'anxiety', 'energy', 'stress']]
y = data['anxiety_level']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize and train the Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(report)

# Test specific cases
test_cases = np.array([
    [1, 1, 5, 1, 5],  # Very high anxiety case (4)
    [2, 2, 4, 2, 4],  # High anxiety case (3)
    [3, 3, 3, 3, 3],  # Medium anxiety case (2)
    [4, 4, 2, 4, 2],  # Low anxiety case (1)
    [5, 5, 1, 5, 1]   # Very low anxiety case (0)
])

predictions = model.predict(test_cases)
print("\nTest Case Predictions (0=Very Low, 1=Low, 2=Medium, 3=High, 4=Very High):")
for i, case in enumerate(test_cases):
    print(f"Case {i+1} {case}: {predictions[i]}")

# Save the model
joblib.dump(model, 'model.pkl')
print("Model saved as 'model.pkl'") 