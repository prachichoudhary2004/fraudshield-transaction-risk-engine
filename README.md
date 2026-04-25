# FraudShield AI — Real-Time Transaction Risk Analysis

An end-to-end machine learning system for real-time fraud detection, transaction risk scoring, and intelligent payment security analysis.

![FraudShield AI](https://img.shields.io/badge/Status-ML%20Project-brightgreen)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Flask](https://img.shields.io/badge/Flask-2.3+-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![SMOTE](https://img.shields.io/badge/Imbalance-SMOTE-purple)

## ✨ Key Highlights

- End-to-end ML fraud detection pipeline
- SMOTE handling for severe class imbalance
- Fintech-inspired real-time risk dashboard
- Transaction risk scoring with action recommendations

## 🚀 Features

- **Real-Time Risk Analysis**: Instant fraud detection on credit card transactions
- **Modern Dashboard**: Stripe/PayPal-inspired UI with KPI metrics
- **SMOTE Oversampling**: Handles imbalanced datasets effectively
- **Risk Scoring**: 0-100 risk score with confidence levels
- **Dynamic KPIs**: Real-time model performance metrics (Precision, Recall, AUC)
- **Sample Data**: One-click demo transaction loading
- **Advanced Risk Signals**: Collapsible PCA component inputs for detailed analysis

## 📊 Model Performance

- **Recall**: 89.8%
- **ROC-AUC**: 0.977
- **Precision**: 11.8%
- **Accuracy**: 98.8%
- **Imbalance Handling**: SMOTE Oversampling

## 🛠️ Tech Stack

- **Backend**: Flask (Python)
- **Machine Learning**: scikit-learn (Logistic Regression)
- **Data Processing**: pandas, numpy
- **Imbalance Handling**: imbalanced-learn (SMOTE)
- **Frontend**: Bootstrap 5.3, Inter Font
- **Visualization**: matplotlib, seaborn

## 📋 Prerequisites

- Python 3.8 or higher
- pip package manager

## 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/prachichoudhary2004/fraudshield-transaction-risk-engine.git
cd fraudshield-transaction-risk-engine
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the dataset:
   - Go to [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
   - Download `creditcard.csv`
   - Create a `dataset` folder in the project root
   - Place `creditcard.csv` inside the `dataset` folder

   Or use Kaggle API:
```bash
pip install kaggle
kaggle datasets download -d mlg-ulb/creditcardfraud
unzip creditcardfraud.zip
mkdir dataset
mv creditcard.csv dataset/
```

## 🚀 Running the Application

1. Start the Flask server:
```bash
python app.py
```

2. Open your browser and navigate to:
```
http://127.0.0.1:5000
```

The application will automatically:
- Load the dataset
- Train the model (if not already trained)
- Start the web server

## 📖 Usage

### Analyze a Transaction

1. Enter the **Transaction Amount** and **Timestamp**
2. Click **"Use Demo Transaction"** to load a sample, or enter custom values
3. Click **"Analyze Transaction"** to get risk assessment
4. View the **Risk Score**, **Confidence**, and **Recommended Action**

### Advanced Analysis

1. Click **"Risk Signals (Advanced)"** to expand PCA component inputs
2. Enter values for V1-V28 features for detailed analysis
3. Analyze to get comprehensive risk assessment

### Retrain Model

1. Scroll to **Model Information** section
2. Click **"Retrain Model"** to retrain with current dataset

## 📁 Project Structure

```
fraudshield-transaction-risk-engine/
├── app.py                      # Flask application and ML model
├── requirements.txt             # Python dependencies
├── README.md                   # This file
├── .gitignore                  # Git ignore rules
├── templates/
│   └── index.html             # Frontend dashboard
├── dataset/
│   └── creditcard.csv         # Dataset (not included in git)
├── fraud_model.pkl            # Trained model (auto-generated)
├── model_metrics.pkl          # Model metrics (auto-generated)
└── confusion_matrix_plot.pkl  # Confusion matrix plot (auto-generated)
```

## 🎯 How It Works

1. **Data Loading**: Loads credit card transaction dataset
2. **Preprocessing**: Splits data into train/test sets with stratification
3. **SMOTE Oversampling**: Balances the imbalanced dataset
4. **Model Training**: Trains Logistic Regression classifier
5. **Prediction**: Real-time fraud detection on new transactions
6. **Risk Scoring**: Calculates risk score based on prediction confidence

## 🔬 Model Details

- **Algorithm**: Logistic Regression
- **Features**: Time, Amount, and 28 PCA components (V1-V28)
- **Target**: Class (0 = Legitimate, 1 = Fraud)
- **Training**: 80% train, 20% test split
- **Imbalance Handling**: SMOTE (Synthetic Minority Over-sampling Technique)

## 📈 KPI Metrics Explained

- **Fraud Rate**: Percentage of fraudulent transactions in the dataset
- **Precision**: Of all predicted fraud cases, how many were actually fraud
- **Recall**: Of all actual fraud cases, how many were correctly predicted
- **AUC Score**: Area Under ROC Curve - model's ability to distinguish classes

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Dataset: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Inspired by modern fintech risk management systems

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

**Built with ❤️ for fraud detection and financial security**
