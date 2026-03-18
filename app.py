from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Load the model
model_path = os.path.join('model', 'final_model.pkl')
model = joblib.load(model_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data and convert to proper types
        input_data = {
            'gender': str(request.form['gender']),
            'ssc_percentage': float(request.form['ssc_percentage']),
            'ssc_board': str(request.form['ssc_board']),
            'hsc_percentage': float(request.form['hsc_percentage']),
            'hsc_board': str(request.form['hsc_board']),
            'hsc_stream': str(request.form['hsc_stream']),
            'degree_percentage': float(request.form['degree_percentage']),
            'degree_field': str(request.form['degree_field']),
            'work_experience_months': int(request.form['work_experience_months']),
            'mba_percentage': float(request.form['mba_percentage']),
            'specialization': str(request.form['specialization']),
            'city_tier': str(request.form['city_tier']),  # Changed to string
            'backlogs': int(request.form['backlogs']),
            'internships_count': int(request.form['internships_count']),
            'projects_count': int(request.form['projects_count']),
            'certifications_count': int(request.form['certifications_count']),
            'technical_skills_score': float(request.form['technical_skills_score']),
            'communication_score': float(request.form['communication_score']),
            'soft_skills_score': float(request.form['soft_skills_score']),
            'leadership_roles': int(request.form['leadership_roles']),
            'extracurricular_activities': int(request.form['extracurricular_activities']),
            'aptitude_score': float(request.form['aptitude_score']),
            'age': int(request.form['age'])
        }
        
        # Create DataFrame with explicit dtypes
        df = pd.DataFrame([input_data])
        
        # Ensure correct dtypes for categorical columns
        cat_cols = ['gender', 'city_tier', 'ssc_board', 'hsc_board', 
                   'hsc_stream', 'degree_field', 'specialization']
        
        for col in cat_cols:
            df[col] = df[col].astype(str)
        
        # Ensure numeric columns are float
        num_cols = ['ssc_percentage', 'hsc_percentage', 'degree_percentage', 
                   'mba_percentage', 'technical_skills_score', 'communication_score',
                   'soft_skills_score', 'aptitude_score']
        
        for col in num_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Make prediction
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]
        
        result = "Placed" if prediction == 1 else "Not Placed"
        
        return render_template('index.html', 
                             prediction_text=f'Prediction: {result}',
                             probability_text=f'Placement Probability: {probability:.2%}')
    
    except Exception as e:
        return render_template('index.html', 
                             prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
