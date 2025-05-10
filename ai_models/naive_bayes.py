import csv
import math

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

def calculate_class_probabilities(X, y):
    class_counts = {}
    feature_counts = {}

    # Считаем сколько классов
    for features, label in zip(X, y):
        if label not in class_counts:
            class_counts[label] = 0
            feature_counts[label] = [{} for _ in range(len(features))]

        class_counts[label] += 1

        for i, val in enumerate(features):
            if val not in feature_counts[label][i]:
                feature_counts[label][i][val] = 0
            feature_counts[label][i][val] += 1

    total_samples = len(y)
    class_probs = {}
    for label in class_counts:
        class_probs[label] = class_counts[label] / total_samples

    return class_probs, feature_counts, class_counts

def predict(input_data):
    X, y = load_dataset()
    class_probs, feature_counts, class_counts = calculate_class_probabilities(X, y)

    best_label = None
    best_prob = -1

    for label in class_probs:
        prob = math.log(class_probs[label])  # log вероятности чтобы избежать переполнения

        for i, val in enumerate(input_data):
            count = feature_counts[label][i].get(val, 0) + 1  # +1 сглаживание
            total = class_counts[label] + len(feature_counts[label][i])
            prob += math.log(count / total)

        if best_label is None or prob > best_prob:
            best_label = label
            best_prob = prob

    # Переводим класс в текст
    labels = {
        0: "Very Low",
        1: "Low",
        2: "Medium",
        3: "High",
        4: "Very High"
    }

    return f"{labels.get(best_label, 'Unknown')} (class {best_label})"
