import math
import os
import csv

# Загрузка датасета
def load_dataset(filename="survey_data.csv"):
    X = []
    y = []
    with open(filename, "r") as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            X.append([float(val) for val in row[:5]])
            y.append(float(row[5]))
    return X, y

# Сигмоида
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# Тренировка
def train(X, y, lr=0.01, epochs=2000):
    n_samples = len(X)
    n_features = len(X[0])

    weights = [0.0] * n_features
    bias = 0.0

    for _ in range(epochs):
        for xi, target in zip(X, y):
            z = sum(w * xij for w, xij in zip(weights, xi)) + bias
            pred = sigmoid(z)
            error = pred - target

            # обновление
            for j in range(n_features):
                weights[j] -= lr * error * xi[j]
            bias -= lr * error

    return weights, bias

# Предсказание
def predict(input_data):
    X, y = load_dataset()
    weights, bias = train(X, y)

    z = sum(w * x for w, x in zip(weights, input_data)) + bias
    probability = sigmoid(z)

    # Интерпретация
    if probability <= 0.2:
        return f"Very Low ({probability:.2f})"
    elif probability <= 0.4:
        return f"Low ({probability:.2f})"
    elif probability <= 0.6:
        return f"Medium ({probability:.2f})"
    elif probability <= 0.8:
        return f"High ({probability:.2f})"
    else:
        return f"Very High ({probability:.2f})"
