# Простая реализация Linear Regression без библиотек
import os
import csv

# Функция для загрузки датасета
def load_dataset(filename="survey_data.csv"):
    X = []
    y = []
    if not os.path.exists(filename):
        raise FileNotFoundError("Dataset not found!")

    with open(filename, "r") as file:
        reader = csv.reader(file)
        next(reader)  # Пропустить заголовок
        for row in reader:
            X.append([float(r) for r in row[:5]])
            y.append(float(row[5]))
    return X, y

# Функция для тренировки модели
def train(X, y):
    n_samples = len(X)
    n_features = len(X[0])

    # Инициализация весов и смещения
    weights = [0.0] * n_features
    bias = 0.0
    lr = 0.01
    epochs = 1000

    # Градиентный спуск
    for _ in range(epochs):
        for xi, target in zip(X, y):
            prediction = sum(w * xij for w, xij in zip(weights, xi)) + bias
            error = prediction - target

            # Обновление весов
            for j in range(n_features):
                weights[j] -= lr * error * xi[j]

            bias -= lr * error

    return weights, bias

# Функция для предсказания
def predict(input_data):
    # Загрузка и тренировка
    X, y = load_dataset()
    weights, bias = train(X, y)

    # Предсказание
    result = sum(w * x for w, x in zip(weights, input_data)) + bias

    # Классификация результата
    if result <= 0.5:
        return "Very Low"
    elif result <= 1.5:
        return "Low"
    elif result <= 2.5:
        return "Medium"
    elif result <= 3.5:
        return "High"
    else:
        return "Very High"
