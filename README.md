# 🌸 E2E MLOps Demo with MLflow – Iris Classification

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking%2C%20Registry-orange.svg)](https://mlflow.org/)
[![Status](https://img.shields.io/badge/Status-Completed-brightgreen.svg)]()

This is a simple end-to-end MLOps project demonstrating how to train, track, register, and deploy a machine learning model using **MLflow** and **Flask**. The model classifies iris flowers based on four input features.

---

## 🧠 Project Goals

- Train a `RandomForestClassifier` on the classic Iris dataset.
- Track training experiments with MLflow Tracking.
- Register and store the model in MLflow Model Registry.
- Deploy the model using a Flask API for inference.
- Showcase a simplified MLOps pipeline from training to deployment.

---

## 📁 Project Structure

```bash
.
├── data/
│   └── iris.csv                  # Training data
├── artifacts/                   # Stored model artifacts
├── mlflow.db                    # SQLite DB used by MLflow Tracking
├── train.py                     # Script to train and log model
├── serve.py                     # Flask API for serving model
└── README.md
```

---

## ⚙️ Installation

```bash
# 1. (Optional) Create virtual environment
python -m venv .venv
source .venv/bin/activate     # On Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

Required packages:
- Python 3.8+
- MLflow
- scikit-learn
- pandas
- Flask

---

## 🚀 Running the Project

Use **three terminals** to run different components:

### Terminal 1 – Run MLflow Tracking UI

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Access the MLflow UI at: [http://localhost:5000](http://localhost:5000)

---

### Terminal 2 – Train and log the model

```bash
python train.py
```

What happens:
- Loads and splits the Iris dataset.
- Trains a Random Forest model.
- Logs parameters and accuracy metric to MLflow.
- Saves the model to the Model Registry with the name `IrisRF`.

---

### Terminal 3 – Run the Flask API Server

```bash
python serve.py
```

The API will be available at: [http://localhost:5001/predict](http://localhost:5001/predict)

---

## 🔍 Code Explanation

### `train.py`
- Loads Iris dataset and trains a `RandomForestClassifier`.
- Uses `mlflow.log_params()` and `mlflow.log_metric()` to track hyperparameters and accuracy.
- Logs and registers the model via:
```python
mlflow.sklearn.log_model(model, "model", registered_model_name="IrisRF")
```

### `serve.py`
- Loads the latest or promoted model (`@champion`) from MLflow Model Registry:
```python
model = mlflow.pyfunc.load_model("models:/IrisRF@champion")
```
- Runs a simple Flask server to expose the model as a REST API.

---

## 📬 API Request Guide

### Endpoint: `POST /predict`

#### ✅ Request Payload (JSON)
```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

| Field         | Description                   |
|---------------|-------------------------------|
| sepal_length  | Sepal length in cm            |
| sepal_width   | Sepal width in cm             |
| petal_length  | Petal length in cm            |
| petal_width   | Petal width in cm             |

---

### 🧾 Response Example

```json
{
  "species": 0
}
```

Where the predicted `species` corresponds to:
- `0`: Iris-setosa
- `1`: Iris-versicolor
- `2`: Iris-virginica

---

## 🧪 Test with `curl`

```bash
curl -X POST http://localhost:5001/predict \
-H "Content-Type: application/json" \
-d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```

---

## 🔄 Possible Extensions

- 🧪 Add automated tests and CI/CD integration.
- 🐳 Dockerize the full pipeline.
- 📊 Add model monitoring for drift detection.
- ☁️ Deploy to AWS/GCP/Azure for production.

---

## ✅ Summary

This mini-project demonstrates a simple but complete MLOps flow:
1. Train & track models with MLflow
2. Register and manage models
3. Serve models via REST API
4. Predict with real-time JSON input

> "Garbage in, garbage out" — ensure data quality for effective ML models.

---

⭐️ Star this repo if you find it useful!
