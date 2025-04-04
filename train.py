import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Thiết lập MLFlow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Iris Classification")

# Load data
data = pd.read_csv("data/iris.csv")
X = data.drop("species", axis=1)
y = data["species"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tham số model
params = {
    "n_estimators": 100,
    "max_depth": 3,
    "random_state": 42
}

with mlflow.start_run():
    # Log parameters
    mlflow.log_params(params)
    
    # Train model
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    
    # Evaluate
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    mlflow.log_metric("accuracy", accuracy)
    
    # Log model
    mlflow.sklearn.log_model(model, "model", registered_model_name="IrisRF")
    
    print(f"Model trained with accuracy: {accuracy:.2f}")