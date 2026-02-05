# Hospital Triage and Staffing Forecast System

Streamlit app with:
- Triage prediction from patient intake and vital signs
- Next-hour staffing forecast for four departments

## Project Path

`C:\Users\willg\HospitalModule`

## Current Files

- `app.py`: Streamlit dashboard with `Triage` and `Staffing` tabs
- `model.ipynb`: model training and evaluation notebook
- `cleaned.csv`: processed dataset for model development and analysis
- `MSE 436 Dataset.csv`: original dataset
- `best_triage_model.pkl`: trained model loaded by `app.py`
- `requirements.txt`: Python dependencies

## Setup

```bash
pip install -r requirements.txt
```

## Train / Rebuild Model

Run all cells in `model.ipynb` to train models and regenerate:

- `best_triage_model.pkl`

## Run the Web App

Start Streamlit from the project root:

```bash
streamlit run app.py
```

The app provides:

- `Triage` tab: enter patient features and predict KTAS level
- `Staffing` tab: next-hour staffing forecast and extra staff recommendations by department

## Staffing Forecast Details

- Departments: `Emergency`, `Surgery`, `Critical Care`, `Step Down`
- User inputs:
  - Free workers per department
  - Waiting patients per department
  - Simulation runs
  - Coverage percentile (for staffing recommendation)
- Recommendation rule:
  - Target patient-to-worker ratio is `1:1`
  - Recommended extra staff is computed per department from the selected forecast percentile
- Flow assumptions (next hour):
  - Arrivals:
    - Emergency: bounded normal in `[1, 9]`, mean `4`
    - Surgery/Critical Care/Step Down: bounded normal in `[0, 2]`, mean `0.3`
  - Transfers:
    - Emergency -> any other department: `[0, 2]`, mean `0.5`
    - Critical Care -> Surgery or Step Down: `[0, 1]`, mean `0.3`
    - Surgery -> Step Down or Critical Care: `[0, 1]`, mean `0.3`
    - Step Down -> Critical Care or Surgery: `[0, 1]`, mean `0.3`
  - Exits:
    - Emergency: `[0, 4]`, mean `2`
    - Other departments: `[0, 2]`, mean `0.3`
