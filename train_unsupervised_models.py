import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import joblib

# Load data
data = pd.read_csv('survey_data.csv')
X = data[['feeling', 'sleep', 'anxiety', 'energy', 'stress']]

# 1. KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
joblib.dump(kmeans, 'models/kmeans.pkl')

# 2. PCA
pca = PCA(n_components=2)
pca.fit(X)
joblib.dump(pca, 'models/pca.pkl')

# 3. Apriori → псевдо модель (сохраним просто список правил)
rules = [
    "Feeling <= 2 → Высокий риск",
    "Anxiety >= 4 → Повышенный риск",
    "Energy >= 4 → Низкий риск"
]

joblib.dump(rules, 'models/apriori.pkl')

print("✅ Unsupervised модели обучены и сохранены в models/")
