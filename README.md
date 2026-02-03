# Hospital Triage Prediction System

This project uses machine learning to predict triage levels for hospital patients based on their vital signs and other clinical features taken upon hospital entry.

## Run the Script

**Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
**Run All Sections of model.ipynb**

This will:
- Load the hospital dataset
- Create clinical threshold features (blood pressure, heart rate, etc.)
- Train XGBoost, SVM and Random Forest models
- Select the best performing model
- Save the model as `best_triage_model.pkl`

**Run the Web Application**

Start the Streamlit web interface:

```bash
streamlit run old/triage_app.py
```

This will open a web browser where you can:
- Enter patient vital signs and information
- Get predicted triage levels (1-5, where 1 is most urgent)
- View prediction probabilities
