import random

def load_dataset(filename="survey_data.csv"):
    X = []
    with open(filename, "r") as file:
        reader = file.readlines()
    
    for row in reader[1:]:  # пропустить заголовок
        parts = row.strip().split(",")
        X.append([int(val) for val in parts[:5]])

    return X

def euclidean_distance(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5

def initialize_centroids(X, k=3):
    random.seed(42)  # фиксируем выбор
    return random.sample(X, k)

def assign_clusters(X, centroids):
    clusters = []
    for point in X:
        distances = [euclidean_distance(point, c) for c in centroids]
        cluster = distances.index(min(distances))
        clusters.append(cluster)
    return clusters

def update_centroids(X, clusters, k=3):
    new_centroids = []
    for cluster_id in range(k):
        cluster_points = [x for x, c in zip(X, clusters) if c == cluster_id]
        if cluster_points:
            centroid = [sum(vals) / len(vals) for vals in zip(*cluster_points)]
        else:
            centroid = [0] * len(X[0])
        new_centroids.append(centroid)
    return new_centroids

def train_kmeans(X, k=3, max_iters=10):
    centroids = initialize_centroids(X, k)

    for _ in range(max_iters):
        clusters = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, clusters, k)

        if new_centroids == centroids:
            break
        centroids = new_centroids

    return centroids

def predict(input_data):
    X = load_dataset()
    centroids = train_kmeans(X)

    # Оцениваем риск по сумме значений в центроиде
    centroid_scores = [(i, sum(c)) for i, c in enumerate(centroids)]
    centroid_scores.sort(key=lambda x: x[1])  # от меньшей суммы к большей

    # Назначаем метки на основе позиции
    cluster_labels = {
        centroid_scores[0][0]: "Cluster 0 (Low Risk)",
        centroid_scores[1][0]: "Cluster 1 (Medium Risk)",
        centroid_scores[2][0]: "Cluster 2 (High Risk)",
    }

    # Определяем ближайший кластер
    distances = [euclidean_distance(input_data, c) for c in centroids]
    cluster = distances.index(min(distances))

    return f"{cluster_labels.get(cluster)} (cluster {cluster})"
