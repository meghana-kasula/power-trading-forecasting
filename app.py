from pathlib import Path

import pandas as pd
from flask import Flask, flash, render_template, request

from src.prediction import ForecastService


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

app = Flask(__name__)
app.config["SECRET_KEY"] = "power-price-dashboard"

forecast_service = ForecastService(MODEL_DIR)


def load_default_data():
    input_df = pd.read_csv(DATA_DIR / "flasksample.csv")
    actual_df = pd.read_csv(DATA_DIR / "ActualMCPValues.csv")
    return input_df, actual_df


def read_uploaded_csv(file_storage):
    if not file_storage or file_storage.filename == "":
        return None
    if not file_storage.filename.lower().endswith(".csv"):
        raise ValueError("Please upload a CSV file.")
    return pd.read_csv(file_storage)


@app.route("/", methods=["GET", "POST"])
def index():
    using_sample = True
    uploaded_name = None

    try:
        if request.method == "POST":
            uploaded_df = read_uploaded_csv(request.files.get("file"))
            if uploaded_df is None:
                flash("No file selected. Showing the sample forecast instead.")
                input_df, actual_df = load_default_data()
            else:
                input_df = uploaded_df
                actual_df = pd.read_csv(DATA_DIR / "ActualMCPValues.csv")
                using_sample = False
                uploaded_name = request.files["file"].filename
        else:
            input_df, actual_df = load_default_data()

        result = forecast_service.run_forecast(input_df, actual_df=actual_df)
    except Exception as exc:
        flash(str(exc))
        input_df, actual_df = load_default_data()
        result = forecast_service.run_forecast(input_df, actual_df=actual_df)
        using_sample = True
        uploaded_name = None

    return render_template(
        "dashboard.html",
        result=result,
        required_columns=forecast_service.required_columns,
        using_sample=using_sample,
        uploaded_name=uploaded_name,
    )


if __name__ == "__main__":
    app.run(debug=True)
