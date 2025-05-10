import random

def load_dataset(filename="survey_data.csv"):
    X = []
    with open(filename, "r") as file:
        reader = file.readlines()
    
    for row in reader[1:]:  # пропускаем заголовок через [1:]
        parts = row.strip().split(",")
        X.append([int(val) for val in parts[:5]])

    return X


def euclidean_distance(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5

def initialize_centroids(X, k=3):
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

    # Определяем ближайший кластер
    distances = [euclidean_distance(input_data, c) for c in centroids]
    cluster = distances.index(min(distances))

    cluster_labels = {
        0: "Cluster 0 (Low Risk)",
        1: "Cluster 1 (Medium Risk)",
        2: "Cluster 2 (High Risk)"
    }

    return f"{cluster_labels.get(cluster, 'Unknown')} (cluster {cluster})"
