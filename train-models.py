import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os
# Load dataset
df = pd.read_csv("beer-servings.csv")

# Drop missing values
df = df.dropna()

# Features and target
X = df[['country', 'beer_servings', 'spirit_servings', 'wine_servings', 'continent']]
y = df['total_litres_of_pure_alcohol']

# OneHotEncoder for categorical data
preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore'), ['country', 'continent'])
], remainder='passthrough')

# Define models
models = {
    'LR_model': LinearRegression(),
    'Random Forest': RandomForestRegressor(random_state=42)
}

# Track best model
best_model = None
best_score = -1
best_name = ""


# Train and evaluate each model
for name, model in models.items():
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"{name} R² Score: {r2:.2f}")

     # Track best model
    if r2 > best_score:
        best_model = pipeline
        best_score = r2
        best_name = name

# Save the best model as a .pkl file
os.makedirs("model", exist_ok=True)
joblib.dump(best_model, f"model/{best_name}_pipeline.pkl")
print(f"\n✅ Best model '{best_name}' saved as 'model/{best_name}_pipeline.pkl'")