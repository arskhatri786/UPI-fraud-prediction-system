## UPI Fraud Detection System
MCA Final Year Project | GBU Noida | Semester 4

## Project Overview
An AI/ML-powered system to detect fraudulent UPI transactions in real-time using machine learning algorithms including Random Forest, Gradient Boosting, and Logistic Regression.

## How to Run

Step 1 — Setup virtual environment
```bash
python -m venv myenv
myenv\Scripts\activate        # Windows
source myenv/bin/activate     # Mac/Linux
```

Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

Step 3 — Run the app
```bash
streamlit run app.py
```

## ML Models Used
| Model | Accuracy | AUC-ROC |
|-------|----------|---------|
| Random Forest | ~95% | ~0.98 |
| Gradient Boosting | ~94% | ~0.97 |
| Logistic Regression | ~88% | ~0.93 |

## Features
- Real-time fraud prediction
- Interactive dashboard with charts
- 3 ML models comparison
- Dataset explorer with filters
- Risk score gauge

## Input Features
- Transaction Amount
- Hour of Transaction
- Transaction Frequency
- Location Distance
- Device Type
- Failed Attempts
- Account Age
- PIN Change Status

## Project Structure
```
upi_fraud_detection/
├── app.py                 # Main Streamlit app
├── train_model.py         # Model training script
├── generate_dataset.py    # Dataset generator
├── requirements.txt       # Dependencies
├── models/                # Saved ML models (auto-generated)
└── upi_transactions.csv   # Dataset (auto-generated)
```
