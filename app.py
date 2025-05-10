import os
import sqlite3
import joblib
import numpy as np
import pandas as pd
import spacy
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from datetime import datetime
import ai_models.linear_regression as linear
import ai_models.logistic_regression as logistic
import ai_models.decision_tree as tree
import ai_models.random_forest as forest
import ai_models.naive_bayes as bayes
import ai_models.knn as knn
import ai_models.svm as svm
import ai_models.gradient_boosting as gb
import ai_models.kmeans as kmeans
import ai_models.apriori as apriori
import ai_models.pca as pca
import ai_models.yolo_detector as yolo

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'svetlyiden_secret_key'

# Load spaCy model - fallback to simple tokenizer if loading fails
try:
    nlp = spacy.load('en_core_web_sm')
    print("Successfully loaded spaCy model")
except Exception as e:
    print(f"Error loading spaCy model: {e}")
    # Create a simple tokenizer as fallback
    from spacy.lang.en import English
    nlp = English()
    print("Using fallback English tokenizer")

# Load ML model
try:
    model = joblib.load('model.pkl')
    print("Successfully loaded model.pkl")
except Exception as e:
    print(f"Error loading model: {e}")
    # We'll handle prediction failures in the code

# Setup database
def init_db():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    c.execute('''
    CREATE TABLE IF NOT EXISTS survey_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        feeling INTEGER,
        sleep INTEGER,
        anxiety INTEGER,
        energy INTEGER,
        stress INTEGER,
        anxiety_level INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')
    c.execute('''
    CREATE TABLE IF NOT EXISTS journal_entries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        content TEXT,
        sentiment_score REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')
    conn.commit()
    conn.close()

init_db()

# Helper functions
def get_user_id():
    if 'user_id' not in session:
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        c.execute("INSERT INTO users (name) VALUES ('Anonymous')")
        user_id = c.lastrowid
        conn.commit()
        conn.close()
        session['user_id'] = user_id
    return session['user_id']

def find_all(text, substring):
    """Find all occurrences of substring in text"""
    start = 0
    while True:
        start = text.find(substring, start)
        if start == -1: return
        yield start
        start += 1  # Move past the current match

def analyze_text(text):
    # We'll use a simple approach with word stems that works for English text
    
    # Count positive and negative words
    positive_words = [
        # English positive words
        'happy', 'good', 'better', 'joy', 'excited', 'pleasant', 'peaceful',
        'great', 'wonderful', 'excellent', 'amazing', 'fantastic', 'brilliant',
        'delighted', 'glad', 'pleased', 'satisfied', 'cheerful', 'content',
        'optimistic', 'positive', 'energetic', 'enthusiastic', 'hopeful',
        'grateful', 'thankful', 'blessed', 'lucky', 'love', 'enjoy', 'smile'
    ]
    
    negative_words = [
        # English negative words
        'sad', 'bad', 'worse', 'anxious', 'depressed', 'unhappy', 'worry', 
        'stress', 'tired', 'angry', 'upset', 'disappointed', 'frustrated',
        'annoyed', 'irritated', 'miserable', 'gloomy', 'terrible', 'horrible',
        'awful', 'scared', 'afraid', 'fearful', 'nervous', 'tense', 'exhausted',
        'lonely', 'hurt', 'pain', 'hate', 'dislike', 'disappointed', 'failure'
    ]
    
    positive_count = 0
    negative_count = 0
    
    # Convert to lowercase for case-insensitive matching
    text_lower = text.lower()
    
    # Check for each positive word stem
    for word in positive_words:
        positive_count += sum(1 for _ in find_all(text_lower, word))
    
    # Check for each negative word stem
    for word in negative_words:
        negative_count += sum(1 for _ in find_all(text_lower, word))
    
    # Calculate sentiment score (-1 to 1)
    total = positive_count + negative_count
    if total > 0:
        sentiment_score = (positive_count - negative_count) / total
    else:
        sentiment_score = 0
    
    print(f"Text analysis: Positive words: {positive_count}, Negative words: {negative_count}, Score: {sentiment_score}")
    return sentiment_score

def get_recommendations(anxiety_level, sentiment_score=None):
    recommendations = []
    
    if anxiety_level == 0:  # Very Low anxiety
        recommendations = [
            "Excellent! Continue maintaining your daily routine and healthy habits.",
            "You're doing great! Your anxiety level is very low.",
            "Share your stress management methods with friends - you're good at it!",
            "Keep focusing on self-care, it's bringing results."
        ]
    elif anxiety_level == 1:  # Low anxiety
        recommendations = [
            "Continue maintaining your daily routine and healthy habits.",
            "Try practicing mindfulness to maintain emotional balance.",
            "Engage in activities you enjoy to strengthen your positive mood.",
            "Stay connected with friends and family."
        ]
    elif anxiety_level == 2:  # Medium anxiety
        recommendations = [
            "Take short breaks throughout the day for deep breathing.",
            "Consider limiting caffeine and alcohol consumption.",
            "Engage in moderate physical activity like walking or swimming.",
            "Try to maintain a regular sleep schedule.",
            "Consider talking to a friend about your feelings."
        ]
    elif anxiety_level == 3:  # High anxiety
        recommendations = [
            "Spend a few minutes in the morning and evening for breathing exercises.",
            "Write down your anxious thoughts and try to analyze them objectively.",
            "Pay special attention to sleep and diet.",
            "Make time for regular physical exercise.",
            "Consider consulting with a specialist if anxiety interferes with daily life."
        ]
    else:  # Very High anxiety (4)
        recommendations = [
            "Practice deep breathing exercises 2-3 times a day.",
            "Use progressive muscle relaxation techniques.",
            "Make time for enjoyable activities and self-care.",
            "If possible, limit exposure to stressful situations.",
            "Consider reaching out to a mental health professional.",
            "Remember that asking for help is a sign of strength."
        ]
    
    # Additional recommendations based on sentiment if available
    if sentiment_score is not None:
        if sentiment_score < -0.5:
            recommendations.append("Your journal shows signs of negative thoughts. Write down 3 positive things that happened today.")
            recommendations.append("Spend 10 minutes a day on meditation or breathing exercises.")
            recommendations.append("Call someone close to you - social connection can significantly improve your mood.")
        elif sentiment_score < 0:
            recommendations.append("Try reframing negative thoughts into more balanced ones.")
            recommendations.append("Take a short walk outdoors - it can help clear your mind.")
        elif sentiment_score > 0.5:
            recommendations.append("Great! Write down what made your day so good to repeat this experience in the future.")
            recommendations.append("Share your positive experience with loved ones - it will enhance positive emotions.")
        elif sentiment_score > 0:
            recommendations.append("Continue noting positive moments in your life.")
            recommendations.append("Think about what other activities might improve your mood and add them to your daily routine.")
    
    return recommendations

def get_progress_data():
    user_id = get_user_id()
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    # Get survey results
    c.execute("SELECT anxiety_level, created_at FROM survey_results WHERE user_id = ? ORDER BY created_at", (user_id,))
    survey_data = c.fetchall()
    
    # Get journal sentiment scores
    c.execute("SELECT sentiment_score, created_at FROM journal_entries WHERE user_id = ? ORDER BY created_at", (user_id,))
    journal_data = c.fetchall()
    
    conn.close()
    
    # Return empty data if no records found
    if not survey_data and not journal_data:
        return None
    
    # Format dates for display
    def format_date(date_str):
        try:
            # Handle bytes object
            if isinstance(date_str, bytes):
                date_str = date_str.decode('utf-8')
                
            dt = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S.%f')
        except ValueError:
            try:
                dt = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
            except ValueError:
                # If all else fails, return the string representation
                return str(date_str)
        return dt.strftime('%d.%m %H:%M')
    
    # Create data structure for Chart.js
    chart_data = {
        'labels': [],
        'datasets': [
            {
                'label': 'Anxiety Level (0-4)',
                'data': [],
                'borderColor': '#e74c3c',
                'backgroundColor': 'rgba(231, 76, 60, 0.2)',
                'borderWidth': 2,
                'pointRadius': 4,
                'tension': 0.1,
                'yAxisID': 'y-anxiety'
            },
            {
                'label': 'Journal Mood (-1 to 1)',
                'data': [],
                'borderColor': '#3498db',
                'backgroundColor': 'rgba(52, 152, 219, 0.2)',
                'borderWidth': 2,
                'pointRadius': 4,
                'tension': 0.1,
                'yAxisID': 'y-mood'
            }
        ]
    }
    
    # Process survey data
    if survey_data:
        for row in survey_data:
            # Safely convert anxiety_level to integer
            anxiety_level = None
            try:
                if row['anxiety_level'] is not None:
                    # Try to handle different types of data
                    if isinstance(row['anxiety_level'], bytes):
                        # For binary data, try to extract a valid integer or default to None
                        try:
                            # First try to decode as string then convert to int
                            anxiety_level = int(row['anxiety_level'].decode('utf-8').strip())
                        except (UnicodeDecodeError, ValueError):
                            # If it's not valid UTF-8 or not a number, try to interpret first byte as integer
                            if len(row['anxiety_level']) > 0 and 0 <= row['anxiety_level'][0] <= 4:
                                anxiety_level = row['anxiety_level'][0]
                            else:
                                anxiety_level = None
                    else:
                        # For non-binary data, convert directly
                        anxiety_level = int(row['anxiety_level'])
            except (ValueError, TypeError):
                # If conversion fails, use None
                anxiety_level = None
                
            # Format date and ensure it's a string
            date = format_date(row['created_at'])
            
            # Only add valid data points
            if anxiety_level is not None and 0 <= anxiety_level <= 4:
                chart_data['labels'].append(date)
                chart_data['datasets'][0]['data'].append(anxiety_level)
    
    # Process journal data - generate matching labels for consistent x-axis
    if journal_data:
        journal_dates = []
        journal_scores = []
        
        for row in journal_data:
            # Format the date
            date = format_date(row['created_at'])
            
            # Safely convert sentiment score to float
            sentiment_score = None
            try:
                if row['sentiment_score'] is not None:
                    if isinstance(row['sentiment_score'], bytes):
                        try:
                            sentiment_score = float(row['sentiment_score'].decode('utf-8').strip())
                        except (UnicodeDecodeError, ValueError):
                            sentiment_score = None
                    else:
                        sentiment_score = float(row['sentiment_score'])
            except (ValueError, TypeError):
                sentiment_score = None
                
            # Only add valid data points
            if sentiment_score is not None and -1 <= sentiment_score <= 1:
                journal_dates.append(date)
                journal_scores.append(sentiment_score)
        
        # If we already have labels from survey data, align journal data with those labels
        if chart_data['labels']:
            # Fill with null for dates that don't have journal entries
            for date in chart_data['labels']:
                if date in journal_dates:
                    idx = journal_dates.index(date)
                    chart_data['datasets'][1]['data'].append(journal_scores[idx])
                else:
                    chart_data['datasets'][1]['data'].append(None)
            
            # Add any journal dates not already in labels
            for i, date in enumerate(journal_dates):
                if date not in chart_data['labels']:
                    chart_data['labels'].append(date)
                    # Add null for anxiety data at this date
                    chart_data['datasets'][0]['data'].append(None)
                    chart_data['datasets'][1]['data'].append(journal_scores[i])
        else:
            # No survey data, just use journal dates as labels
            chart_data['labels'] = journal_dates
            chart_data['datasets'][1]['data'] = journal_scores
    
    # Sort all data by date
    if chart_data['labels']:
        # Create tuples of (date, anxiety, mood) to sort by date
        combined = list(zip(chart_data['labels'], 
                         chart_data['datasets'][0]['data'], 
                         chart_data['datasets'][1]['data']))
        combined.sort(key=lambda x: x[0])
        
        # Unpack back into chart structure
        chart_data['labels'] = [item[0] for item in combined]
        chart_data['datasets'][0]['data'] = [item[1] for item in combined]
        chart_data['datasets'][1]['data'] = [item[2] for item in combined]
    
    return chart_data

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/form', methods=['GET', 'POST'])
def form():
    if request.method == 'POST':
        feeling = int(request.form['feeling'])
        sleep = int(request.form['sleep'])
        anxiety = int(request.form['anxiety'])
        energy = int(request.form['energy'])
        stress = int(request.form['stress'])
        
        # Predict anxiety level using the model
        features = np.array([[feeling, sleep, anxiety, energy, stress]])
        anxiety_level = model.predict(features)[0]
        
        # Save to database
        user_id = get_user_id()
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        c.execute(
            "INSERT INTO survey_results (user_id, feeling, sleep, anxiety, energy, stress, anxiety_level) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (user_id, feeling, sleep, anxiety, energy, stress, anxiety_level)
        )
        conn.commit()
        conn.close()
        
        # Get recommendations
        recommendations = get_recommendations(anxiety_level)
        
        return render_template('results.html', 
                              anxiety_level=anxiety_level, 
                              recommendations=recommendations)
    
    return render_template('form.html')
@app.route('/ai-lab', methods=['GET', 'POST'])
def ai_lab():
    results = {}
    summary = None
    chart_labels = []
    chart_data = []
    algo_chart_data = {}
    uploaded_image = None

    # Алгоритмы
    import ai_models.linear_regression as linear
    import ai_models.logistic_regression as logistic
    import ai_models.decision_tree as tree
    import ai_models.random_forest as forest
    import ai_models.naive_bayes as bayes
    import ai_models.knn as knn
    import ai_models.svm as svm
    import ai_models.gradient_boosting as gb
    import ai_models.kmeans as kmeans
    import ai_models.apriori as apriori
    import ai_models.pca as pca
    import ai_models.yolo_detector as yolo

    algos = {
        "Linear Regression": linear.predict,
        "Logistic Regression": logistic.predict,
        "Decision Tree": tree.predict,
        "Random Forest": forest.predict,
        "Naive Bayes": bayes.predict,
        "KNN": knn.predict,
        "SVM": svm.predict,
        "Gradient Boosting": gb.predict,
        "KMeans": kmeans.predict,
        "Apriori": apriori.predict,
        "PCA": pca.predict,
    }

    if request.method == 'POST':
        feeling = int(request.form['feeling'])
        sleep = int(request.form['sleep'])
        anxiety = int(request.form['anxiety'])
        energy = int(request.form['energy'])
        stress = int(request.form['stress'])
        input_data = [feeling, sleep, anxiety, energy, stress]

        # Определяем выбранные алгоритмы
        selected_algos = request.form.getlist('selected_algos')
        action = request.form.get('action')

        # Если нажата кнопка Compare или не выбрано вообще → использовать все алгоритмы
        if action == "compare" or not selected_algos:
            selected_algos = list(algos.keys())

        # Прогнозирование выбранных алгоритмов
        for name in selected_algos:
            prediction = algos[name](input_data)
            results[name] = prediction
            algo_chart_data[name] = extract_numeric_level(prediction)

        # YOLO всегда обрабатываем отдельно
        if 'yolo_image' in request.files:
            file = request.files['yolo_image']
            if file and file.filename != '':
                import os
                os.makedirs('static/uploads', exist_ok=True)
                file_path = f"static/uploads/{file.filename}"
                file.save(file_path)
                uploaded_image = file_path
                prediction = yolo.predict(file_path)
                results['YOLO (CV)'] = prediction
            else:
                results['YOLO (CV)'] = "No image"
        else:
            results['YOLO (CV)'] = "No image"

        # График общий
        chart_labels = list(results.keys())
        chart_data = [extract_numeric_level(value) for value in results.values()]

        # Summary
        low = sum(1 for res in results.values() if "Low" in res or "Very Low" in res)
        medium = sum(1 for res in results.values() if "Medium" in res)
        high = sum(1 for res in results.values() if "High" in res)

        if high >= 4:
            summary = "Overall risk is High"
        elif medium >= 4:
            summary = "Overall risk is Medium"
        else:
            summary = "Overall risk is Low"

    return render_template('ai_lab.html',
                           results=results,
                           summary=summary,
                           chart_labels=chart_labels,
                           chart_data=chart_data,
                           algo_chart_data=algo_chart_data,
                           uploaded_image=uploaded_image)




def extract_numeric_level(prediction_text):
    if "Very High" in prediction_text:
        return 4
    elif "High" in prediction_text:
        return 3
    elif "Medium" in prediction_text:
        return 2
    elif "Low" in prediction_text:
        return 1
    elif "Very Low" in prediction_text:
        return 0
    else:
        return 0




@app.route('/journal', methods=['GET', 'POST'])
def journal():
    if request.method == 'POST':
        content = request.form['content']
        
        # Analyze text sentiment
        sentiment_score = analyze_text(content)
        
        # Save to database
        user_id = get_user_id()
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        c.execute(
            "INSERT INTO journal_entries (user_id, content, sentiment_score) VALUES (?, ?, ?)",
            (user_id, content, sentiment_score)
        )
        conn.commit()
        conn.close()
        
        # Get the most recent anxiety level for this user
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        c.execute("SELECT anxiety_level FROM survey_results WHERE user_id = ? ORDER BY created_at DESC LIMIT 1", (user_id,))
        result = c.fetchone()
        conn.close()
        
        if result:
            anxiety_level = result[0]
        else:
            # Default to medium if no survey completed
            anxiety_level = 1
        
        # Get recommendations based on both anxiety level and sentiment
        recommendations = get_recommendations(anxiety_level, sentiment_score)
        
        # Use English mood descriptions
        mood_text = "neutral"
        if sentiment_score > 0.3:
            mood_text = "positive"
        elif sentiment_score < -0.3:
            mood_text = "negative"
            
        return render_template('results.html', 
                              sentiment_score=sentiment_score,
                              mood_text=mood_text,
                              recommendations=recommendations)
    
    return render_template('journal.html')

@app.route('/progress')
def progress():
    chart_data = get_progress_data()
    return render_template('progress.html', chart_data=chart_data)

@app.route('/api/progress-data')
def api_progress_data():
    """API endpoint to get progress data for Chart.js"""
    chart_data = get_progress_data()
    if chart_data:
        return jsonify(chart_data)
    return jsonify({"error": "No data available"}), 404

if __name__ == '__main__':
    app.run(debug=True) 





