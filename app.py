from flask import Flask, render_template, request
import io
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

LOOKBACK = 30
DEFAULT_HORIZON = 14

def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(LOOKBACK, 1)),
        tf.keras.layers.LSTM(32, return_sequences=True),
        tf.keras.layers.Dropout(0.15),
        tf.keras.layers.LSTM(16),
        tf.keras.layers.Dense(8, activation="relu"),
        tf.keras.layers.Dense(1)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"]
    )
    return model

def make_sequences(values):
    X, y = [], []
    for i in range(LOOKBACK, len(values)):
        X.append(values[i - LOOKBACK:i, 0])
        y.append(values[i, 0])
    return np.array(X), np.array(y)

def demo_data(store=1, item=1):
    rng = np.random.default_rng(42 + store * 10 + item)
    dates = pd.date_range("2025-01-01", periods=260, freq="D")
    t = np.arange(len(dates))
    trend = 0.035 * t
    weekly = 5 * np.sin(2 * np.pi * t / 7)
    monthly = 3 * np.sin(2 * np.pi * t / 30)
    noise = rng.normal(0, 2.2, len(t))
    sales = 45 + trend + weekly + monthly + noise
    sales = np.maximum(np.round(sales), 0)

    return pd.DataFrame({
        "date": dates,
        "store": store,
        "item": item,
        "sales": sales
    })

def forecast_series(series, horizon):
    series = series.sort_values("date").copy()
    values = series["sales"].astype("float32").values.reshape(-1, 1)

    if len(values) < LOOKBACK + 20:
        raise ValueError(
            f"At least {LOOKBACK + 20} observations are recommended for the demo."
        )

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values)

    X, y = make_sequences(scaled)
    X = X.reshape((-1, LOOKBACK, 1))

    model = build_model()
    model.fit(
        X, y,
        epochs=8,
        batch_size=16,
        validation_split=0.15,
        verbose=0,
        shuffle=False
    )

    window = scaled[-LOOKBACK:].copy()
    future_scaled = []

    for _ in range(horizon):
        x = window.reshape(1, LOOKBACK, 1)
        next_value = float(model.predict(x, verbose=0)[0, 0])
        future_scaled.append(next_value)
        window = np.vstack([window[1:], [[next_value]]])

    future_sales = scaler.inverse_transform(
        np.array(future_scaled).reshape(-1, 1)
    ).flatten()
    future_sales = np.maximum(future_sales, 0)

    future_dates = pd.date_range(
        series["date"].iloc[-1] + pd.Timedelta(days=1),
        periods=horizon,
        freq="D"
    )

    return pd.DataFrame({
        "date": future_dates,
        "forecast": np.round(future_sales, 2)
    })

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    mode = "Demo data"
    selected_store = 1
    selected_item = 1
    horizon = DEFAULT_HORIZON

    if request.method == "POST":
        try:
            selected_store = int(request.form.get("store", 1))
            selected_item = int(request.form.get("item", 1))
            horizon = min(max(int(request.form.get("horizon", DEFAULT_HORIZON)), 1), 60)

            uploaded = request.files.get("file")

            if uploaded and uploaded.filename:
                mode = "Uploaded Kaggle CSV"
                raw = uploaded.read()
                data = pd.read_csv(io.BytesIO(raw), parse_dates=["date"])

                required = {"date", "store", "item", "sales"}
                missing = required.difference(data.columns)
                if missing:
                    raise ValueError(
                        "CSV is missing required columns: "
                        + ", ".join(sorted(missing))
                    )

                series = data[
                    (data["store"] == selected_store) &
                    (data["item"] == selected_item)
                ][["date", "sales"]].dropna()
            else:
                series = demo_data(selected_store, selected_item)[["date", "sales"]]

            result = forecast_series(series, horizon)

        except Exception as exc:
            error = str(exc)

    return render_template(
        "index.html",
        result=result,
        error=error,
        mode=mode,
        selected_store=selected_store,
        selected_item=selected_item,
        horizon=horizon
    )

if __name__ == "__main__":
    app.run(debug=True)
