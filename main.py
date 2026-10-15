import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# 1. The Raw Studio Data (Messy, missing data, text and numbers)
data = pd.DataFrame({
    'Budget': [150, 200, 50, 10, 80, 120, 250, 5],
    'Runtime': [120, np.nan, 90, 85, 100, 140, np.nan, 80], # Lost files!
    'Genre': ['Action', 'Action', 'Comedy', 'Drama', 'Comedy', 'Sci-Fi', 'Action', 'Drama'], # Text!
    'Hit': [1, 1, 0, 0, 0, 1, 1, 0] # 1 = Hit, 0 = Flop
})

X = data[['Budget', 'Runtime', 'Genre']]
y = data['Hit']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ==========================================
# YOUR TURN: DIRECT THE PIPELINE
# ==========================================

# 2. Build the Numeric Assembly Line (Fix NaN with mean -> Scale)
numeric_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# 3. Build the Categorical Assembly Line (Fix NaN with most_frequent -> One-Hot Encode)
categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# 4. The Router (Merge the two assembly lines)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_pipeline, ['Budget', 'Runtime']),
        ('cat', categorical_pipeline, ['Genre'])
    ])

# 5. The Master Pipeline (Router -> Engine)
master_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('engine', LogisticRegression())
])

# 6. Train the machine
master_pipeline.fit(X_train, y_train)

# 7. Take the Final Exam
predictions = master_pipeline.predict(X_test)
print("Predictions:", predictions)

# 8. Grade the Exam
print("--- ADVANCED METRICS ---")
print(classification_report(y_test, predictions))
print("--- THE 4 ROOMS ---")
print(confusion_matrix(y_test, predictions))