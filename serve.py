import mlflow
from flask import Flask, request, jsonify
import pandas as pd

app = Flask(__name__)

# Set up MLFlow
mlflow.set_tracking_uri("http://localhost:5000")

try:
    # Option 1: Load the promoted model (replacing the Production stage)
    model = mlflow.pyfunc.load_model("models:/IrisRF@champion")
except:
    try:
        # Option 2: Load the latest version of the model
        model = mlflow.pyfunc.load_model("models:/IrisRF/latest")
        print("Using latest version instead of promoted model")
    except Exception as e:
        print(f"Could not load model: {e}")
        raise

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        input_data = pd.DataFrame([[
            data["sepal_length"],
            data["sepal_width"],
            data["petal_length"],
            data["petal_width"]
        ]], columns=["sepal length (cm)", "sepal width (cm)", 
                    "petal length (cm)", "petal width (cm)"])
        
        prediction = model.predict(input_data)
        # Convert prediction to species name
        label_map = {0: "setosa", 1: "versicolor", 2: "virginica"}
        return jsonify({"species": label_map[int(prediction[0])]})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)