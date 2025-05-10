def predict(input_data):
    feeling, sleep, anxiety, energy, stress = input_data

    rules = []

    # Частые правила
    if anxiety >= 5 and stress >= 5:
        rules.append("Very High")
    elif anxiety >= 4 and stress >= 4:
        rules.append("High")
    elif anxiety <= 2 and stress <= 2:
        rules.append("Low")
    elif anxiety <= 2:
        rules.append("Low")
    elif stress >= 4:
        rules.append("High")

    # Анализ правил
    if not rules:
        return "Medium"

    # Голосование
    count = {"Very Low": 0, "Low": 0, "Medium": 0, "High": 0, "Very High": 0}
    for r in rules:
        count[r] += 1

    # Выбираем самое частое
    result = max(count, key=count.get)
    return result
