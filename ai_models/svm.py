import csv

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

# Простейший мультиклассовый SVM (One-vs-Rest)
def train(X, y, lr=0.001, epochs=1000):
    n_samples = len(X)
    n_features = len(X[0])
    classes = set(y)

    # Для каждого класса отдельные w и b
    models = {}
    for cls in classes:
        weights = [0.0] * n_features
        bias = 0.0

        for _ in range(epochs):
            for xi, target in zip(X, y):
                yi = 1 if target == cls else -1
                result = sum(w * xij for w, xij in zip(weights, xi)) + bias

                if yi * result < 1:
                    # Ошибка → обновляем
                    for j in range(n_features):
                        weights[j] += lr * (yi * xi[j])
                    bias += lr * yi

        models[cls] = (weights, bias)

    return models

def predict(input_data):
    X, y = load_dataset()
    models = train(X, y)

    # Получаем "оценку" от каждого класса
    scores = {}
    for cls, (weights, bias) in models.items():
        score = sum(w * x for w, x in zip(weights, input_data)) + bias
        scores[cls] = score

    # Выбираем класс с максимальной оценкой
    best_class = max(scores, key=scores.get)

    labels = {
        0: "Very Low",
        1: "Low",
        2: "Medium",
        3: "High",
        4: "Very High"
    }

    return f"{labels.get(best_class, 'Unknown')} (class {best_class})"

