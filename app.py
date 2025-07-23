from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

model = joblib.load(r'C:/Users/user/Desktop/Ict-webapp casstudy/train-model.pkl')
le_country = joblib.load('train-model.pkl')
le_continent = joblib.load('model/le_continent.pkl')

df = pd.read_csv('data/drinks.csv')
countries = sorted(df['country'].unique())
continents = sorted(df['continent'].dropna().unique())

@app.route('/')
def home():
    return render_template('index.html', countries=countries, continents=continents)

@app.route('/predict', methods=['POST'])
def predict():
    country = request.form['country']
    continent = request.form['continent']
    beer = float(request.form['beer'])
    spirit = float(request.form['spirit'])
    wine = float(request.form['wine'])

    # Encode inputs
    country_encoded = le_country.transform([country])[0]
    continent_encoded = le_continent.transform([continent])[0]

    features = np.array([[country_encoded, beer, spirit, wine, continent_encoded]])
    prediction = model.predict(features)[0]
    
    return render_template('index.html', prediction_text=f"Predicted pure alcohol consumption: {prediction:.2f} litres", 
                           countries=countries, continents=continents)

if __name__ == "__main__":
    app.run(debug=True)