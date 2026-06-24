import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
df = pd.read_csv("data/Telco-Customer-Churn.csv")
df.drop('customerID', axis=1, inplace=True)
df['TotalCharges'] = pd.to_numeric(
    df['TotalCharges'],
    errors='coerce'
)
df['TotalCharges'] = df['TotalCharges'].fillna(
    df['TotalCharges'].median()
)
df.drop_duplicates(inplace=True)
X = df.drop('Churn', axis=1)
y = df['Churn']

categorical_cols = X.select_dtypes(
    include=['object', 'string']
).columns

X = pd.get_dummies(
    X,
    columns=categorical_cols,
    drop_first=True
)

y = y.map({
    'No':0,
    'Yes':1
})
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("Accuracy:", accuracy)

joblib.dump(
    rf_model,
    "models/churn_model.pkl"
)
joblib.dump(
    X.columns.tolist(),
    "models/feature_columns.pkl"
)