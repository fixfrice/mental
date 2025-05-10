# ai_models/random_forest.py

from collections import Counter

def tree1(data):
    feeling, sleep, anxiety, energy, stress = data
    if anxiety >= 5 or stress >= 5:
        return "Very High"
    if anxiety >= 4:
        return "High"
    if sleep <= 2 and energy <= 2:
        return "Low"
    return "Medium"

def tree2(data):
    feeling, sleep, anxiety, energy, stress = data
    if stress >= 4 and anxiety >= 4:
        return "High"
    if feeling <= 2:
        return "Low"
    if anxiety <= 2 and sleep >= 3:
        return "Very Low"
    return "Medium"

def tree3(data):
    feeling, sleep, anxiety, energy, stress = data
    score = 0
    if anxiety >= 3: score += 1
    if stress >= 3: score += 1
    if sleep <= 2: score += 1
    if energy <= 2: score += 1
    if score >= 3:
        return "High"
    elif score == 2:
        return "Medium"
    elif score == 1:
        return "Low"
    else:
        return "Very Low"

def predict(input_data):
    predictions = [
        tree1(input_data),
        tree2(input_data),
        tree3(input_data)
    ]

    # Подсчитываем голоса
    vote_count = Counter(predictions)
    most_common = vote_count.most_common(1)[0][0]  # [(‘Medium’, 2), …]

    return f"{most_common} (by voting: {dict(vote_count)})"
