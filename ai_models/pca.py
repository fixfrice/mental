import csv

def load_dataset(filename="survey_data.csv"):
    X = []
    with open(filename, "r") as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            X.append([int(val) for val in row[:5]])
    return X

def calculate_means(X):
    n_features = len(X[0])
    means = [0] * n_features

    for row in X:
        for i in range(n_features):
            means[i] += row[i]

    means = [val / len(X) for val in means]
    return means

def predict(input_data):
    X = load_dataset()
    means = calculate_means(X)

    # Вычисляем отклонения
    deviations = []
    for val, mean in zip(input_data, means):
        deviations.append(abs(val - mean))

    # Находим самые важные признаки (top 2)
    important_indexes = sorted(range(len(deviations)), key=lambda i: deviations[i], reverse=True)[:2]

    features = ["Feeling", "Sleep", "Anxiety", "Energy", "Stress"]
    important_features = [features[i] for i in important_indexes]

    return f"Most important features: {important_features[0]} and {important_features[1]}"
