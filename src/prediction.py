from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objs as go

from src.feature_engineering import (
    WEATHER_RENAME_MAP,
    add_recursive_features,
    clean_input_data,
    infer_display_index,
)
from src.metrics import calculate_metrics


class ForecastService:
    FORECAST_PRIOR_WEIGHT = 0.35

    def __init__(self, model_dir):
        model_dir = Path(model_dir)
        self.model = joblib.load(model_dir / "xgb_model.pkl")
        self.scaler = joblib.load(model_dir / "scaler.pkl")
        self.feature_columns = joblib.load(model_dir / "columns.pkl")
        self.required_columns = [
            "Date",
            "Hour",
            "Time Block",
            "Purchase Bid (MW)",
            "Sell Bid (MW)",
            "MCV (MW)",
            "Final Scheduled Volume (MW)",
            "MCP (Rs/MWh) *",
            "national_temperature_2m",
            "national_apparent_temperature",
            "national_relative_humidity_2m",
            "national_precipitation",
            "national_wind_speed_10m",
            "national_rain",
            "national_wind_direction_10m",
        ]

    def run_forecast(self, raw_df, actual_df=None, steps=96):
        self._validate_input(raw_df)
        display_index = infer_display_index(raw_df, periods=steps)
        prepared_df = clean_input_data(raw_df)
        history_df, future_df = self._split_history_and_future(prepared_df, steps)
        predictions = self._recursive_predict(history_df, future_df, steps=steps)

        if not predictions:
            raise ValueError("Not enough usable data after lag feature engineering.")

        actual_values = self._extract_actual_values(actual_df, len(predictions))
        result_df = pd.DataFrame(
            {
                "Step": range(1, len(predictions) + 1),
                "Timestamp": display_index[: len(predictions)],
                "Predicted MCP": predictions,
                "Actual MCP": actual_values,
            }
        )
        result_df["Error"] = result_df["Actual MCP"] - result_df["Predicted MCP"]
        result_df["Absolute Error"] = result_df["Error"].abs()

        metrics = calculate_metrics(result_df["Actual MCP"], result_df["Predicted MCP"])
        summary = self._build_summary(result_df)

        return {
            "table": self._format_table(result_df),
            "metrics": metrics,
            "summary": summary,
            "forecast_graph": self._forecast_graph(result_df),
            "comparison_graph": self._comparison_graph(result_df),
            "has_actuals": result_df["Actual MCP"].notna().any(),
        }

    def _validate_input(self, raw_df):
        raw_columns = {WEATHER_RENAME_MAP.get(col.strip(), col.strip()) for col in raw_df.columns}
        missing = [col for col in self.required_columns if col not in raw_columns]
        if missing:
            raise ValueError("Missing required columns: " + ", ".join(missing))
        if len(raw_df) < 96:
            raise ValueError("Upload at least 96 rows so lag features can be created.")

    def _split_history_and_future(self, prepared_df, steps):
        if len(prepared_df) > steps:
            history_df = prepared_df.iloc[:-steps].copy()
            future_df = prepared_df.iloc[-steps:].copy()
        else:
            history_df = prepared_df.copy()
            future_df = pd.DataFrame(columns=prepared_df.columns)

        return history_df.reset_index(drop=True), future_df.reset_index(drop=True)

    def _recursive_predict(self, history_df, future_df, steps=96):
        predictions = []
        current_df = history_df.copy()

        for step in range(steps):
            candidate_row = self._build_future_row(current_df, future_df, step)
            forecast_seed = float(candidate_row["MCP (Rs/MWh) *"].iloc[0])
            current_df = pd.concat([current_df, candidate_row], ignore_index=True)
            temp_df = add_recursive_features(current_df)

            if len(temp_df) == 0:
                break

            X = temp_df.drop(columns=["MCP_log"])
            non_scaled_cols = ["Hour", "Date", "Time Block"]
            non_scaled_cols = [col for col in non_scaled_cols if col in X.columns]

            X_non_scaled = X[non_scaled_cols]
            X_to_scale = X.drop(columns=non_scaled_cols)

            scaler_features = self.scaler.feature_names_in_
            for col in scaler_features:
                if col not in X_to_scale.columns:
                    X_to_scale[col] = 0

            X_to_scale = X_to_scale[scaler_features]
            X_scaled_part = pd.DataFrame(
                self.scaler.transform(X_to_scale),
                columns=scaler_features,
                index=X_to_scale.index,
            )

            X_scaled = pd.concat([X_scaled_part, X_non_scaled], axis=1)
            for col in self.feature_columns:
                if col not in X_scaled.columns:
                    X_scaled[col] = 0

            X_scaled = X_scaled[self.feature_columns]
            X_input = X_scaled.iloc[-1:]

            pred_log = self.model.predict(X_input)[0]
            pred_actual = self._blend_with_forecast_seed(float(np.expm1(pred_log)), forecast_seed)
            predictions.append(pred_actual)

            current_df.loc[current_df.index[-1], "MCP (Rs/MWh) *"] = pred_actual

        return predictions

    def _blend_with_forecast_seed(self, model_prediction, forecast_seed):
        if pd.isna(forecast_seed) or forecast_seed <= 0:
            return max(model_prediction, 0.0)

        blended_prediction = (
            (1 - self.FORECAST_PRIOR_WEIGHT) * model_prediction
            + self.FORECAST_PRIOR_WEIGHT * forecast_seed
        )
        return max(float(blended_prediction), 0.0)

    def _build_future_row(self, current_df, future_df, step):
        if step < len(future_df):
            new_row = future_df.iloc[[step]].copy()
        else:
            new_row = current_df.iloc[[-1]].copy()
            self._advance_time_columns(new_row)
            new_row["MCP (Rs/MWh) *"] = self._estimate_future_mcp_seed(current_df, new_row)

        seed_value = pd.to_numeric(new_row["MCP (Rs/MWh) *"], errors="coerce").iloc[0]
        if pd.isna(seed_value) or seed_value <= 0:
            new_row["MCP (Rs/MWh) *"] = self._estimate_future_mcp_seed(current_df, new_row)

        return new_row

    def _estimate_future_mcp_seed(self, current_df, new_row):
        recent_mcp = pd.to_numeric(current_df["MCP (Rs/MWh) *"], errors="coerce").dropna()
        if recent_mcp.empty:
            return 0.0

        seasonal_value = recent_mcp.iloc[-96] if len(recent_mcp) >= 96 else recent_mcp.iloc[-1]
        recent_mean = recent_mcp.tail(12).mean()
        base_value = 0.65 * seasonal_value + 0.35 * recent_mean

        if {"Purchase Bid (MW)", "Sell Bid (MW)"}.issubset(current_df.columns) and {
            "Purchase Bid (MW)",
            "Sell Bid (MW)",
        }.issubset(new_row.columns):
            old_gap = (
                pd.to_numeric(current_df["Purchase Bid (MW)"], errors="coerce")
                - pd.to_numeric(current_df["Sell Bid (MW)"], errors="coerce")
            ).tail(12).mean()
            new_gap = (
                pd.to_numeric(new_row["Purchase Bid (MW)"], errors="coerce").iloc[0]
                - pd.to_numeric(new_row["Sell Bid (MW)"], errors="coerce").iloc[0]
            )
            if pd.notna(old_gap) and pd.notna(new_gap):
                base_value += 0.015 * (new_gap - old_gap)

        return max(float(base_value), 0.0)

    def _advance_time_columns(self, row):
        if "Time Block" in row.columns and pd.notna(row["Time Block"].iloc[0]):
            next_block = (int(row["Time Block"].iloc[0]) + 15) % (24 * 60)
            row.loc[row.index[0], "Time Block"] = next_block

        if "Hour" in row.columns and pd.notna(row["Hour"].iloc[0]):
            row.loc[row.index[0], "Hour"] = int(row["Time Block"].iloc[0] // 60) + 1

        if "Date" in row.columns and "Time Block" in row.columns and int(row["Time Block"].iloc[0]) == 0:
            row.loc[row.index[0], "Date"] = row["Date"].iloc[0] + 24 * 60 * 60

    def _extract_actual_values(self, actual_df, periods):
        if actual_df is None or actual_df.empty:
            return [np.nan] * periods

        actual_df = actual_df.copy()
        actual_df.columns = actual_df.columns.str.strip()
        actual_col = "MCP (Rs/MWh) *"
        if actual_col not in actual_df.columns:
            return [np.nan] * periods

        values = pd.to_numeric(actual_df[actual_col], errors="coerce").head(periods).tolist()
        return values + [np.nan] * (periods - len(values))

    def _build_summary(self, df):
        predicted = df["Predicted MCP"]
        latest = predicted.iloc[-1]
        peak_row = df.loc[predicted.idxmax()]
        low_row = df.loc[predicted.idxmin()]

        return {
            "latest": latest,
            "average": predicted.mean(),
            "minimum": predicted.min(),
            "maximum": predicted.max(),
            "peak_time": peak_row["Timestamp"].strftime("%d %b %Y, %H:%M"),
            "low_time": low_row["Timestamp"].strftime("%d %b %Y, %H:%M"),
            "count": len(df),
        }

    def _format_table(self, df):
        table = df.copy()
        table["Timestamp"] = table["Timestamp"].dt.strftime("%d %b %Y %H:%M")
        for col in ["Predicted MCP", "Actual MCP", "Error", "Absolute Error"]:
            table[col] = table[col].map(lambda value: "" if pd.isna(value) else f"{value:,.2f}")
        return table.to_dict(orient="records")

    def _forecast_graph(self, df):
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df["Timestamp"],
                y=df["Predicted MCP"],
                mode="lines+markers",
                name="Predicted MCP",
                line={"color": "#38bdf8", "width": 3},
                marker={"size": 5},
                hovertemplate="%{x|%d %b %Y %H:%M}<br>Predicted: %{y:.2f}<extra></extra>",
            )
        )
        self._style_graph(fig, "96-Step MCP Forecast", "Predicted MCP (Rs/MWh)")
        return fig.to_html(full_html=False, include_plotlyjs=False, config={"scrollZoom": True})

    def _comparison_graph(self, df):
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df["Timestamp"],
                y=df["Actual MCP"],
                mode="lines+markers",
                name="Actual MCP",
                line={"color": "#22c55e", "width": 3},
                marker={"size": 5},
                hovertemplate="%{x|%d %b %Y %H:%M}<br>Actual: %{y:.2f}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df["Timestamp"],
                y=df["Predicted MCP"],
                mode="lines+markers",
                name="Predicted MCP",
                line={"color": "#f97316", "width": 3},
                marker={"size": 5},
                hovertemplate="%{x|%d %b %Y %H:%M}<br>Predicted: %{y:.2f}<extra></extra>",
            )
        )
        self._style_graph(fig, "Actual MCP vs Predicted MCP", "MCP (Rs/MWh)")
        return fig.to_html(full_html=False, include_plotlyjs=False, config={"scrollZoom": True})

    def _style_graph(self, fig, title, yaxis_title):
        fig.update_layout(
            title=title,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(24,26,31,0.78)",
            font={"color": "#e5e7eb", "family": "Inter, Segoe UI, Arial"},
            hovermode="x unified",
            margin={"l": 55, "r": 24, "t": 58, "b": 52},
            legend={"orientation": "h", "y": 1.08, "x": 0},
            xaxis={"title": "Time", "showgrid": True, "gridcolor": "rgba(148,163,184,0.15)"},
            yaxis={"title": yaxis_title, "showgrid": True, "gridcolor": "rgba(148,163,184,0.15)"},
        )
