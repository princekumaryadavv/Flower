from flask import Flask, render_template, request
import joblib
import numpy as np

# Load trained model, scaler, and label encoder
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Initialize Flask app
app = Flask(__name__)

# Define home route
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        try:
            # Get user input from form
            sepal_length = float(request.form["sepal_length"])
            sepal_width = float(request.form["sepal_width"])
            petal_length = float(request.form["petal_length"])
            petal_width = float(request.form["petal_width"])

            # Convert input into numpy array
            input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

            # Scale the input data
            import pandas as pd  # Add this at the top if not already imported

            # Define feature names as they were during training
            feature_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

            # Convert input_data to a DataFrame with column names
            input_data_df = pd.DataFrame(input_data, columns=feature_names)

            # Scale the input data
            input_data_scaled = scaler.transform(input_data_df)


            # Make prediction
            prediction = model.predict(input_data_scaled)
            predicted_species = label_encoder.inverse_transform(prediction)[0]

            return render_template("index.html", prediction=predicted_species)

        except Exception as e:
            return render_template("index.html", error="Invalid input! Please enter valid numbers.")

    return render_template("index.html", prediction=None, error=None)

# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)
