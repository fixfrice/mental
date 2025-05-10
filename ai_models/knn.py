import csv
from collections import Counter

def load_dataset(filename="survey_data.csv"):
    X = []
    y = []
    with open(filename, "r") as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            X.append([int(val) for val in row[:5]])
            y.append(int(row[5]))
    return X, y

# Евклидово расстояние
def euclidean_distance(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5

# Основной алгоритм
def predict(input_data, k=3):
    X, y = load_dataset()

    distances = []
    for features, label in zip(X, y):
        dist = euclidean_distance(input_data, features)
        distances.append((dist, label))

    distances.sort(key=lambda x: x[0])
    nearest_labels = [label for (_, label) in distances[:k]]

    most_common = Counter(nearest_labels).most_common(1)[0][0]

    labels = {
        0: "Very Low",
        1: "Low",
        2: "Medium",
        3: "High",
        4: "Very High"
    }

    return f"{labels.get(most_common, 'Unknown')} (class {most_common})"
