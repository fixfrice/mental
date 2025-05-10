# ai_models/decision_tree.py

def predict(input_data):
    # Распаковываем входные данные
    feeling, sleep, anxiety, energy, stress = input_data

    # Уровень тревожности определяется в первую очередь по anxiety и stress
    if anxiety >= 5 and stress >= 5:
        return "Very High"
    
    if anxiety >= 4 and stress >= 4:
        if sleep <= 2:
            return "Very High"
        else:
            return "High"
    
    # Если стресс высокий, но тревожность умеренная
    if stress >= 4:
        if feeling <= 2:
            return "High"
        else:
            return "Medium"
    
    # Если тревожность умеренная
    if anxiety == 3:
        if energy <= 2:
            return "Medium"
        else:
            return "Low"

    # Если тревожность низкая, но плохой сон или усталость
    if anxiety <= 2:
        if sleep <= 2 and energy <= 2:
            return "Low"
        elif feeling <= 1:
            return "Low"
        else:
            return "Very Low"
    
    # В остальных случаях
    return "Medium"
