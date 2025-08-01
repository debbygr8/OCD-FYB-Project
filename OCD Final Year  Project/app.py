from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
from sklearn.cluster import Birch
import os

app = Flask(__name__)

# Load trained Birch model based on user dataset
def load_model():
    data = pd.read_csv("cleaned_ocd_dataset.csv")
    features = data[[
        'Age',
        'Family History of OCD',
        'Duration of Symptoms (months)',
        'Depression Diagnosis',
        'Anxiety Diagnosis'
    ]].dropna()
    # Convert categorical columns to numerical if needed
    for col in features.columns:
        if features[col].dtype == 'object':
            features[col] = features[col].astype('category').cat.codes

    model = Birch(n_clusters=3)
    model.fit(features.values)
    return model

model = load_model()

def map_severity(label):
    return {0: "Mild", 1: "Moderate", 2: "Severe"}.get(label, "Unknown")

@app.route('/')
def home():
    return render_template('landing.html', image_url="static/images/ocdcare_banner.jpg",)

@app.route('/form', methods=['GET', 'POST'])
def form():
    if request.method == 'POST':
        # Extract user input
        age = int(request.form['age'])
        history = 1 if request.form['history'] == 'Yes' else 0
        duration = int(request.form['duration'])
        depression = 1 if request.form['depression'] == 'Yes' else 0
        anxiety = 1 if request.form['anxiety'] == 'Yes' else 0
        
        selected_subtypes = request.form.getlist('subtypes')

        # Combine data for prediction
        input_data = np.array([[age, history, duration, depression, anxiety]])
        label = model.predict(input_data)[0]
        severity = map_severity(label)
        subtype_result = ",".join(selected_subtypes) 

        return redirect(url_for('result', severity=severity,subtype=subtype_result))
    return render_template('form.html')

#Quiz Questions
@app.route('/quiz', methods=['GET', 'POST'])
def quiz():
    if request.method == 'POST':
        # Collect all the quiz answers from the form
        answers = [
            request.form.get('q1', ''),
            request.form.get('q2', ''),
            request.form.get('q3', ''),
            request.form.get('q4', ''),
            request.form.get('q5', ''),
            request.form.get('q6', ''),
            request.form.get('q7', ''),
            request.form.get('q8', ''),
            request.form.get('q9', ''),
            request.form.get('q10', ''),
            request.form.get('q11', ''),
            request.form.get('q12', ''),
        ]
        score, feedback = calculate_ocd_score(answers)
        return render_template('quiz_result.html', score=score, feedback=feedback)
    return render_template('quiz.html')
def calculate_ocd_score(answers):
    # Define the scoring system
    scoring = {
        'q1': 1, 'q2': 1, 'q3': 1, 'q4': 1,
        'q5': 1, 'q6': 1, 'q7': 1, 'q8': 1,
        'q9': 1, 'q10': 1, 'q11': 1, 'q12': 1
    }
    
    score = sum(scoring.get(q, 0) for q in answers)
    
    if score <= 4:
        feedback = "Your score indicates a low likelihood of OCD. However, if you have concerns, consider consulting a mental health professional."
    elif score <= 8:
        feedback = "Your score suggests moderate OCD symptoms. It may be beneficial to seek professional advice."
    else:
        feedback = "Your score indicates significant OCD symptoms. We strongly recommend consulting a mental health professional."
    
    return score, feedback


@app.route('/result')
def result():
    severity = request.args.get('severity', 'Unknown')
    return render_template('result.html', severity=severity, subtype=request.args.get('subtype', 'None'))

@app.route('/recommendation')
def recommendation():
    return render_template('recommendation.html')


@app.route('/about')
def about():
    return render_template('about.html')



if __name__ == '__main__':
    app.run(debug=True)
