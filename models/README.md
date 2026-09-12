# Models

The notebooks can save trained Keras models and scalers here.

Typical generated files are:

```text
demand_lstm.keras
scaler.pkl
```

Generated model artifacts are ignored by Git so that the repository remains lightweight. The Flask application can train a compact model from an uploaded CSV, so a binary model file is not required just to demonstrate the application.
