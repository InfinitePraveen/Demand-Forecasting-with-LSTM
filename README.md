# Demand Forecasting with LSTM

A practical time-series forecasting project that uses a Long Short-Term Memory (LSTM) neural network to forecast daily store-item demand.

The project uses the **Store Item Demand Forecasting Challenge** dataset from Kaggle. The dataset contains daily sales for 10 stores and 50 items from 2013 through 2017, with `date`, `store`, `item`, and `sales` fields.

## Project goal

Given the recent sales history of a selected store and item, predict future daily demand. The project demonstrates:

- Time-series exploration
- Trend and seasonality analysis
- Time-aware train/validation/test splitting
- Sliding-window sequence creation
- Min-Max scaling
- LSTM model building with TensorFlow/Keras
- MAE, RMSE and sMAPE evaluation
- Recursive multi-step forecasting
- A simple Flask web application for an interview-ready demonstration

## Dataset

Download `train.csv` from the Kaggle Store Item Demand Forecasting Challenge and place it at:

```text
data/train.csv
```

Kaggle dataset:
https://www.kaggle.com/competitions/demand-forecasting-kernels-only/data

The original competition asks participants to forecast three months of item sales for 50 items across 10 stores.

## Repository structure

```text
Demand-Forecasting-with-LSTM/
│
├── app.py
├── requirements.txt
├── README.md
├── CONTRIBUTE.md
├── CHANGELOG.md
├── .gitignore
│
├── data/
│   └── README.md
│
├── models/
│   └── README.md
│
├── notebooks/
│   ├── 01_Exploratory_Data_Analysis.ipynb
│   ├── 02_Time_Series_Preparation.ipynb
│   ├── 03_LSTM_Model_Training.ipynb
│   └── 04_Future_Demand_Forecasting.ipynb
│
├── templates/
│   └── index.html
│
└── static/
    └── style.css
```

There is intentionally **no `src/` directory**, no preprocessing module, and no separate utility scripts. The learning workflow stays inside the notebooks, while `app.py` contains the small Flask demo.

## Quick start

### 1. Create an environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add the dataset

Download `train.csv` and put it here:

```text
data/train.csv
```

### 4. Run the notebooks

Start Jupyter:

```bash
jupyter notebook
```

Recommended order:

1. `01_Exploratory_Data_Analysis.ipynb`
2. `02_Time_Series_Preparation.ipynb`
3. `03_LSTM_Model_Training.ipynb`
4. `04_Future_Demand_Forecasting.ipynb`

The notebooks use a selected store-item series for the LSTM experiment. This keeps the project understandable in an interview instead of hiding the logic behind a large production-style codebase.

### 5. Run the Flask app

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

The web app supports two practical modes:

- **Demo mode:** generates a small realistic demand series so the UI can be demonstrated without downloading the full dataset.
- **CSV mode:** upload the Kaggle `train.csv`, choose a store and item, and train a compact LSTM directly in the app before generating a forecast.

## Model approach

The main experiment follows a simple sequence-to-one forecasting design:

```text
Previous 30 days of demand
          ↓
      LSTM layer
          ↓
      Dropout
          ↓
      Dense(16)
          ↓
      Dense(1)
          ↓
Next-day demand
```

For multi-day forecasting, the predicted value is appended to the input window and the oldest value is removed. The process is repeated for the requested forecast horizon.

TensorFlow's time-series guidance similarly describes windowing sequential observations and using `tf.keras.layers.LSTM` for recurrent forecasting.

## Evaluation

The notebooks report:

- MAE
- RMSE
- sMAPE

The validation period is later in time than the training period. This avoids randomly mixing future observations into the training set, which would be inappropriate for a forecasting problem.

## Interview talking points

You can explain the project as:

> "I built an LSTM-based demand forecasting system using five years of store-item sales data. I converted the sales history into fixed-length rolling windows, scaled the target series, trained an LSTM using TensorFlow, evaluated it with MAE, RMSE and sMAPE using time-based validation, and exposed the forecasting workflow through a Flask application."

Be ready to explain:

- Why a random train/test split is avoided
- What an LSTM remembers through its hidden/cell state
- Why scaling helps neural-network training
- How a sliding window becomes supervised learning data
- Why recursive forecasting can accumulate error
- Why a simple seasonal baseline is useful before claiming the LSTM is better
- Why model accuracy can differ between individual store-item series

## Author

**Praveen Kumar**

Data Scientist | Open Source Learner | IBM Certified

GitHub: https://github.com/InfinitePraveen  
LinkedIn: https://www.linkedin.com/in/infinitepraveen/

## Disclaimer

This is an educational/interview portfolio project. Forecast quality depends on the selected store-item series, training period, hyperparameters and available historical information.
