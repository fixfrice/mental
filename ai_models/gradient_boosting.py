def tree1(data):
    feeling, sleep, anxiety, energy, stress = data
    score = 0

    if anxiety >= 4:
        score += 1
    elif anxiety <= 2:
        score -= 1

    return score

def tree2(data):
    feeling, sleep, anxiety, energy, stress = data
    score = 0

    if stress >= 4:
        score += 1
    elif stress <= 2:
        score -= 1

    return score

def tree3(data):
    feeling, sleep, anxiety, energy, stress = data
    score = 0

    if sleep <= 2 or energy <= 2:
        score += 1
    elif feeling >= 4:
        score -= 1

    return score

def predict(input_data):
    # Сумма всех деревьев
    total_score = tree1(input_data) + tree2(input_data) + tree3(input_data)

    # Преобразуем в класс
    if total_score <= -2:
        return "Very Low"
    elif total_score == -1:
        return "Low"
    elif total_score == 0:
        return "Medium"
    elif total_score == 1:
        return "High"
    else:
        return "Very High"
