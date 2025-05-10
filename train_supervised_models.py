import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import joblib

# Load data
data = pd.read_csv('survey_data.csv')
X = data[['feeling', 'sleep', 'anxiety', 'energy', 'stress']]
y = data['anxiety_level']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 1. Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
joblib.dump(lr, 'models/linear_regression.pkl')

# 2. Logistic Regression
logr = LogisticRegression(max_iter=1000)
logr.fit(X_train, y_train)
joblib.dump(logr, 'models/logistic_regression.pkl')

# 3. Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
joblib.dump(dt, 'models/decision_tree.pkl')

# 4. Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
joblib.dump(rf, 'models/random_forest.pkl')

# 5. Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)
joblib.dump(nb, 'models/naive_bayes.pkl')

# 6. KNN
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
joblib.dump(knn, 'models/knn.pkl')

# 7. SVM
svm = SVC(probability=True)
svm.fit(X_train, y_train)
joblib.dump(svm, 'models/svm.pkl')

# 8. Gradient Boosting
gb = GradientBoostingClassifier()
gb.fit(X_train, y_train)
joblib.dump(gb, 'models/gradient_boosting.pkl')

print("✅ Все supervised модели обучены и сохранены в models/")
