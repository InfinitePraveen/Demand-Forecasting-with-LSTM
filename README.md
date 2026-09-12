# Demand Forecasting with LSTM

<p align="center"> <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.12"> <img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow"> <img src="https://img.shields.io/badge/Keras-3.x-D00000?style=for-the-badge&logo=keras&logoColor=white" alt="Keras"> <img src="https://img.shields.io/badge/Pandas-2.x-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas"> </p>

<p align="center"> <img src="https://img.shields.io/badge/NumPy-2.x-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy"> <img src="https://img.shields.io/badge/Scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-learn"> <img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter"> <img src="https://img.shields.io/badge/Flask-Web_App-000000?style=for-the-badge&logo=flask&logoColor=white" alt="Flask"> </p>

<p align="center"> <img src="https://img.shields.io/badge/LSTM-Time_Series-8A2BE2?style=for-the-badge" alt="LSTM"> <img src="https://img.shields.io/badge/Time_Series-Forecasting-2E8B57?style=for-the-badge" alt="Time Series Forecasting"> <img src="https://img.shields.io/badge/License-Educational-lightgrey?style=for-the-badge" alt="Educational"> </p>

A practical time-series forecasting project that uses a **Long Short-Term Memory (LSTM)** neural network to forecast daily store demand.

The project demonstrates how historical store-item sales data can be transformed into time-series sequences, used to train an LSTM model with TensorFlow, evaluated using forecasting metrics, and finally presented through a Flask web application.

---

## Project Overview

Demand forecasting is an important problem in retail and supply-chain analytics. Accurate demand forecasts can help businesses make better decisions about inventory, purchasing, staffing, and resource planning.

In this project, an LSTM neural network is used to learn patterns from historical daily store-item sales and forecast future demand.

### Skills Demonstrated

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* TensorFlow
* Keras
* LSTM Neural Networks
* Time-Series Forecasting
* Feature Scaling
* Sliding-Window Sequences
* Model Evaluation
* Recursive Forecasting
* Flask
* Jupyter Notebook

---

## Dataset

This project uses the **Store Item Demand Forecasting Challenge** dataset.

The dataset contains daily sales information for:

* 10 stores
* 50 items
* Multiple years of historical sales
* Daily demand observations

The main columns are:

```text
date
store
item
sales
```

The dataset can be downloaded from Kaggle:

https://www.kaggle.com/competitions/demand-forecasting-kernels-only/data

After downloading the dataset, place the files inside the `data/` directory.

---

## Important: Python Version

**Python 3.12 is required to execute this project.**

Please install **Python 3.12.x** before creating the virtual environment.

Check your Python version:

```bash
python --version
```

or:

```bash
py --version
```

The expected version is:

```text
Python 3.12.x
```

Using another Python version may cause compatibility problems with the TensorFlow and other project dependencies.

---

# Installation

## 1. Install Python 3.12

Install Python **3.12.x** on your system.

After installation, verify:

```bash
python --version
```

You should see something similar to:

```text
Python 3.12.10
```

---

## 2. Clone the Repository

```bash
git clone https://github.com/InfinitePraveen/Demand-Forecasting-with-LSTM.git
```

Move into the project directory:

```bash
cd Demand-Forecasting-with-LSTM
```

---

## 3. Create a Virtual Environment

Create the environment using Python 3.12:

```bash
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

### Linux/macOS

```bash
source .venv/bin/activate
```

---

## 4. Verify the Virtual Environment

Run:

```bash
python --version
```

Make sure it reports:

```text
Python 3.12.x
```

---

## 5. Install Dependencies

Install all required packages:

```bash
pip install -r requirements.txt
```

---

# Dataset Setup

Download the Store Item Demand Forecasting dataset from Kaggle.

Place the dataset files inside:

```text
data/
```

The expected files are:

```text
data/
├── README.md
├── train.csv
└── test.csv
```

The `train.csv` file is used for model development and forecasting experiments.

The `test.csv` file contains the test-period records supplied with the original competition dataset.

---

# Repository Structure

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
│   ├── README.md
│   ├── train.csv
│   └── test.csv
│
├── models/
│   ├── README.md
│   ├── demand_lstm.keras
│   └── scaler.pkl
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

The project intentionally does **not** contain a `src/` directory, separate preprocessing modules, or unnecessary utility scripts. The main data-science workflow remains inside the Jupyter notebooks, while `app.py` contains the Flask application.

---

# Project Workflow

The project follows this workflow:

```text
Historical Store Sales
          │
          ▼
Exploratory Data Analysis
          │
          ▼
Time-Series Preparation
          │
          ▼
Feature Scaling
          │
          ▼
Sliding-Window Sequences
          │
          ▼
LSTM Neural Network
          │
          ▼
Model Evaluation
          │
          ▼
Future Demand Forecasting
          │
          ▼
Flask Web Application
```

---

# Notebooks

## 01 - Exploratory Data Analysis

File:

```text
notebooks/01_Exploratory_Data_Analysis.ipynb
```

This notebook explores the demand dataset and analyzes:

* Dataset structure
* Missing values
* Store distribution
* Item distribution
* Sales statistics
* Overall demand trends
* Monthly demand
* Store-item demand
* Weekly demand patterns

---

## 02 - Time Series Preparation

File:

```text
notebooks/02_Time_Series_Preparation.ipynb
```

This notebook prepares the data for the LSTM model.

The main steps include:

* Selecting a store-item time series
* Sorting observations chronologically
* Creating training, validation and test periods
* Applying Min-Max scaling
* Creating sliding-window sequences
* Preparing LSTM input tensors
* Creating a seasonal-naive baseline

The model uses recent historical observations to predict the next demand value.

---

## 03 - LSTM Model Training

File:

```text
notebooks/03_LSTM_Model_Training.ipynb
```

This notebook builds and trains the TensorFlow/Keras LSTM model.

The model follows a simple architecture:

```text
30 Days of Historical Demand
            │
            ▼
        LSTM Layer
            │
            ▼
         Dropout
            │
            ▼
        LSTM Layer
            │
            ▼
       Dense Layer
            │
            ▼
        Output
            │
            ▼
     Next-Day Demand
```

The model is evaluated using:

* MAE
* RMSE
* sMAPE

The trained model is saved as:

```text
models/demand_lstm.keras
```

The fitted scaler is saved as:

```text
models/scaler.pkl
```

---

## 04 - Future Demand Forecasting

File:

```text
notebooks/04_Future_Demand_Forecasting.ipynb
```

This notebook demonstrates multi-step forecasting.

The model predicts one future value at a time and then uses that prediction as part of the next input window.

For example:

```text
Previous 30 Days
       ↓
Prediction 1
       ↓
Previous 29 Days + Prediction 1
       ↓
Prediction 2
       ↓
Previous 28 Days + Predictions
       ↓
...
```

This approach allows the model to generate forecasts for multiple future days.

---

# Model

The primary model used in this project is an **LSTM neural network** implemented with TensorFlow/Keras.

LSTM networks are particularly useful for sequential data because they can learn relationships between observations across time.

The model receives a sequence of historical demand observations and predicts the next demand value.

The input shape is approximately:

```text
(samples, 30 time steps, 1 feature)
```

where:

* `samples` = number of training sequences
* `30` = historical days used as the input window
* `1` = sales/demand feature

---

# Model Files

The `models/` directory contains the trained model artifacts.

```text
models/
├── README.md
├── demand_lstm.keras
└── scaler.pkl
```

### `demand_lstm.keras`

Contains the trained TensorFlow/Keras LSTM model.

### `scaler.pkl`

Contains the fitted `MinMaxScaler` used to scale demand values before they are passed to the neural network.

Keeping the scaler is important because future prediction data must be transformed using the same scaling process used during model training.

---

# Evaluation

The project evaluates the forecasting model using:

### Mean Absolute Error

```text
MAE
```

Measures the average absolute difference between actual and predicted demand.

### Root Mean Squared Error

```text
RMSE
```

Penalizes larger forecasting errors more heavily.

### Symmetric Mean Absolute Percentage Error

```text
sMAPE
```

Provides a percentage-based view of forecasting error while treating over- and under-predictions more symmetrically.

A seasonal-naive baseline is also included so the LSTM is not evaluated in isolation.

---

# Running the Notebooks

After activating the Python 3.12 virtual environment, start Jupyter:

```bash
jupyter notebook
```

Run the notebooks in this order:

```text
01_Exploratory_Data_Analysis.ipynb
        ↓
02_Time_Series_Preparation.ipynb
        ↓
03_LSTM_Model_Training.ipynb
        ↓
04_Future_Demand_Forecasting.ipynb
```

---

# Running the Flask Web Application

The repository also contains an interactive Flask application for demonstrating the forecasting system.

Start the application:

```bash
python app.py
```

The application will run locally at:

```text
http://127.0.0.1:5000
```

Open that address in your browser.

---

# Web Application Features

The Flask application provides an interview-friendly demonstration of the project.

It supports:

* Store selection
* Item selection
* Forecast horizon selection
* CSV dataset upload
* LSTM-based demand forecasting
* Forecast result table
* Demo mode
* Simple explanation of the forecasting workflow

The application also displays the project author's professional profiles.

### GitHub

https://github.com/InfinitePraveen

### LinkedIn

https://www.linkedin.com/in/infinitepraveen/

---

# Demo Mode

The Flask application can also run without uploading the complete Kaggle dataset.

If no CSV file is uploaded, the application generates a small demonstration demand series and trains a compact LSTM model.

This makes it easier to demonstrate the project during an interview or presentation.

For actual store-item forecasting, upload the appropriate `train.csv` file.

---

# Interview Explanation

A concise way to explain this project during an interview is:

> I built an LSTM-based demand forecasting system using historical store-item sales data. I performed time-series analysis, created chronological training and validation sets, converted the sales history into rolling sequences, scaled the data using Min-Max scaling, trained an LSTM model with TensorFlow, evaluated it using MAE, RMSE and sMAPE, and deployed an interactive forecasting demonstration using Flask.

---

# Important Interview Concepts

Be prepared to explain:

* Why LSTM is suitable for sequential data
* How LSTM differs from a standard RNN
* What the forget, input and output gates do
* Why time-series data should not normally use a random train/test split
* Why Min-Max scaling is useful for neural networks
* How sliding windows convert time-series data into supervised learning data
* Why the lookback window is 30 days
* How recursive forecasting works
* Why recursive forecasts can accumulate errors
* Why a seasonal-naive baseline is useful
* The difference between MAE and RMSE
* What sMAPE measures
* How the Flask application uses the forecasting model

---

# Limitations

This is an educational and interview portfolio project.

The forecasting performance can vary depending on:

* Store
* Item
* Historical period
* Lookback window
* Model architecture
* Training parameters
* Demand patterns
* Forecast horizon

The LSTM should therefore not automatically be considered better than simpler forecasting methods without comparing their results.

---

# Future Improvements

Possible improvements include:

* Hyperparameter tuning
* Longer historical windows
* Additional LSTM layers
* Bidirectional experiments where appropriate
* GRU comparison
* CNN-LSTM comparison
* Direct multi-step forecasting
* Adding holidays and calendar features
* Adding promotional information
* Adding store and item embeddings
* Comparing against ARIMA/SARIMA
* Comparing against Prophet
* Comparing against tree-based forecasting models
* Automated model retraining
* Forecast confidence intervals
* Model monitoring

---

# Repository Design

This repository intentionally keeps the structure simple.

There is:

* No `src/` directory
* No separate preprocessing package
* No unnecessary utility modules
* No complex application architecture

The notebooks contain the main data-science workflow, while the Flask application provides a practical demonstration of the final forecasting concept.

This makes the repository easier to understand, explain, and present during technical interviews.

---

# Author

## Praveen Kumar

**Data Scientist | Open Source Learner | IBM Certified**

### GitHub

https://github.com/InfinitePraveen

### LinkedIn

https://www.linkedin.com/in/infinitepraveen/

---

# License

This project is intended for educational and portfolio purposes.

Please follow the applicable terms of the original dataset when downloading, using, or redistributing the Kaggle data.

---

## Python Requirement

**Python 3.12.x is required for this project.**

Before running the notebooks or Flask application, verify:

```bash
python --version
```

Expected:

```text
Python 3.12.x
```

Using **Python 3.12** is recommended to maintain compatibility with the dependencies specified in `requirements.txt`.
