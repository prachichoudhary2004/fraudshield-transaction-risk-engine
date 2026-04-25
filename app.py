from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import pickle
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io

app = Flask(__name__)

# Store the trained model and feature names globally
model = None
feature_names = None

# Cache dataset for faster random samples
dataset_cache = None

def download_kaggle_dataset():
    """
    Download the credit card fraud dataset from Kaggle.
    Falls back to local file if Kaggle API isn't available.
    """
    try:
        import kaggle
        print("Downloading dataset from Kaggle...")
        kaggle.api.dataset_download_files('mlg-ulb/creditcardfraud', path='./', unzip=True)
        print("Dataset downloaded successfully!")
        return True
    except ImportError:
        print("Kaggle API not installed. Install with: pip install kaggle")
        return False
    except Exception as e:
        print(f"Kaggle download failed: {e}")
        return False

def train_model():
    """
    Train the fraud detection model using credit card transaction data.
    Uses SMOTE for better handling of imbalanced dataset.
    Includes comprehensive evaluation metrics.
    """
    global model, feature_names
    
    # Try to load the credit card dataset
    try:
        card_dataset = pd.read_csv("dataset/creditcard.csv")
    except FileNotFoundError:
        print("Couldn't find creditcard.csv - trying to download from Kaggle...")
        if download_kaggle_dataset():
            try:
                card_dataset = pd.read_csv("dataset/creditcard.csv")
            except FileNotFoundError:
                print("Still couldn't find dataset. Please download manually or check Kaggle API setup.")
                return False
        else:
            print("Please download the dataset from Kaggle and place creditcard.csv in the project folder.")
            return False
    
    # Prepare features (X) and target (y) - use full dataset
    X = card_dataset.drop(["Class"], axis=1)
    Y = card_dataset["Class"]
    feature_names = X.columns.tolist()
    
    # Split data for training and testing (before SMOTE to avoid data leakage)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)
    
    print(f"Original dataset - Fraud cases: {sum(Y_train)}, Legit cases: {len(Y_train) - sum(Y_train)}")
    
    # Apply SMOTE to balance the training data
    try:
        smote = SMOTE(random_state=42)
        X_train_resampled, Y_train_resampled = smote.fit_resample(X_train, Y_train)
        print(f"After SMOTE - Fraud cases: {sum(Y_train_resampled)}, Legit cases: {len(Y_train_resampled) - sum(Y_train_resampled)}")
    except ImportError:
        print("SMOTE not available, falling back to random undersampling...")
        # Fallback to improved undersampling
        fraud_indices = Y_train[Y_train == 1].index
        legit_indices = Y_train[Y_train == 0].sample(n=len(fraud_indices), random_state=42).index
        balanced_indices = fraud_indices.union(legit_indices)
        X_train_resampled = X_train.loc[balanced_indices]
        Y_train_resampled = Y_train.loc[balanced_indices]
    
    # Train logistic regression model
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_resampled, Y_train_resampled)
    
    # Make predictions on test set
    Y_train_pred = model.predict(X_train_resampled)
    Y_test_pred = model.predict(X_test)
    Y_test_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate comprehensive metrics
    train_accuracy = accuracy_score(Y_train_resampled, Y_train_pred)
    test_accuracy = accuracy_score(Y_test, Y_test_pred)
    precision = precision_score(Y_test, Y_test_pred)
    recall = recall_score(Y_test, Y_test_pred)
    f1 = f1_score(Y_test, Y_test_pred)
    roc_auc = roc_auc_score(Y_test, Y_test_proba)
    
    # Generate confusion matrix
    cm = confusion_matrix(Y_test, Y_test_pred)
    
    print(f"Model trained successfully!")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    
    # Save metrics and confusion matrix for web display
    model_metrics = {
        'accuracy': test_accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm.tolist(),
        'classification_report': classification_report(Y_test, Y_test_pred, output_dict=True)
    }
    
    # Save confusion matrix plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Legitimate', 'Fraud'], 
                yticklabels=['Legitimate', 'Fraud'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    # Save plot as base64 string
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
    img_buffer.seek(0)
    cm_plot = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close()
    
    # Save model and metrics
    with open('fraud_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('model_metrics.pkl', 'wb') as f:
        pickle.dump(model_metrics, f)
    with open('confusion_matrix_plot.pkl', 'wb') as f:
        pickle.dump(cm_plot, f)
    
    return True

def load_model():
    """
    Load a previously trained model from disk.
    If the original dataset isn't available, use default feature names.
    """
    global model, feature_names
    
    # Check if we have a saved model
    if os.path.exists('fraud_model.pkl'):
        with open('fraud_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Try to get feature names from the original dataset
        try:
            card_dataset = pd.read_csv("dataset/creditcard.csv")
            feature_names = card_dataset.drop(["Class"], axis=1).columns.tolist()
        except FileNotFoundError:
            # Fallback to default feature names if dataset isn't available
            feature_names = [f'V{i}' for i in range(1, 29)] + ['Time', 'Amount']
        
        return True
    return False

@app.route('/')
def home():
    """Main page for the fraud detection web app"""
    return render_template('index.html')

@app.route('/train')
def train():
    """Train the fraud detection model via API call"""
    if train_model():
        return jsonify({"status": "success", "message": "Model trained successfully!"})
    else:
        return jsonify({"status": "error", "message": "Failed to train model. Check if creditcard.csv exists."})

@app.route('/predict', methods=['POST'])
def predict():
    """
    Make fraud prediction on transaction data.
    Expects JSON with all transaction features.
    """
    if model is None:
        if not load_model():
            return jsonify({"error": "Model not trained. Please train the model first."})
    
    try:
        # Get transaction data from the request
        data = request.json
        
        # Build feature array in the correct order
        features = []
        for feature in feature_names:
            features.append(float(data.get(feature, 0)))
        
        input_data = np.array(features).reshape(1, -1)
        
        # Use the model to predict if this transaction is fraud
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0]
        
        # Format the results for the frontend
        result = {
            "prediction": int(prediction),
            "prediction_label": "Fraud" if prediction == 1 else "Legitimate",
            "confidence": float(max(prediction_proba)),
            "fraud_probability": float(prediction_proba[1]),
            "legitimate_probability": float(prediction_proba[0])
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/model_info')
def model_info():
    """Return information about the trained model"""
    if model is None:
        return jsonify({
            "model_type": "Logistic Regression",
            "features": [f'V{i}' for i in range(1, 29)] + ['Time', 'Amount'],
            "num_features": 30,
            "metrics": {}
        })
    
    # Load metrics if available
    metrics = {}
    if os.path.exists('model_metrics.pkl'):
        with open('model_metrics.pkl', 'rb') as f:
            metrics = pickle.load(f)
    
    return jsonify({
        "model_type": "Logistic Regression",
        "features": feature_names,
        "num_features": len(feature_names) if feature_names else 0,
        "metrics": metrics
    })

@app.route('/model_metrics')
def model_metrics():
    """Return detailed model performance metrics"""
    if not os.path.exists('model_metrics.pkl'):
        return jsonify({"error": "Model not trained yet"})
    
    try:
        with open('model_metrics.pkl', 'rb') as f:
            metrics = pickle.load(f)
        
        # Load confusion matrix plot if available
        cm_plot = None
        if os.path.exists('confusion_matrix_plot.pkl'):
            with open('confusion_matrix_plot.pkl', 'rb') as f:
                cm_plot = pickle.load(f)
        
        return jsonify({
            "metrics": metrics,
            "confusion_matrix_plot": cm_plot
        })
        
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/random_sample')
def random_sample():
    """Return a random transaction sample from the dataset"""
    global dataset_cache
    
    try:
        # Load dataset into cache if not already loaded
        if dataset_cache is None:
            dataset_cache = pd.read_csv("creditcard.csv")
        
        card_dataset = dataset_cache
        
        # Randomly choose between fraud and legitimate (50/50 chance)
        import random
        choose_fraud = random.choice([True, False])
        
        if choose_fraud:
            # Get a random fraud transaction
            fraud_data = card_dataset[card_dataset['Class'] == 1]
            if len(fraud_data) > 0:
                sample = fraud_data.sample(n=1).iloc[0]
                label = "Fraudulent"
            else:
                # Fallback to legitimate if no fraud data
                sample = card_dataset[card_dataset['Class'] == 0].sample(n=1).iloc[0]
                label = "Legitimate"
        else:
            # Get a random legitimate transaction
            legit_data = card_dataset[card_dataset['Class'] == 0]
            sample = legit_data.sample(n=1).iloc[0]
            label = "Legitimate"
        
        # Convert to dictionary, excluding the Class column
        sample_dict = sample.drop('Class').to_dict()
        
        return jsonify({
            "sample": sample_dict,
            "label": label
        })
        
    except FileNotFoundError:
        return jsonify({"error": "Dataset not found. Please ensure creditcard.csv exists."})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/kpi_metrics')
def kpi_metrics():
    """Return KPI metrics for dashboard display"""
    global dataset_cache
    
    try:
        # Load metrics if available
        if os.path.exists('model_metrics.pkl'):
            with open('model_metrics.pkl', 'rb') as f:
                metrics = pickle.load(f)
            
            # Calculate fraud rate from dataset
            if dataset_cache is None:
                dataset_cache = pd.read_csv("dataset/creditcard.csv")
            
            fraud_count = len(dataset_cache[dataset_cache['Class'] == 1])
            total_count = len(dataset_cache)
            fraud_rate = (fraud_count / total_count) * 100
            
            return jsonify({
                "fraud_rate": round(fraud_rate, 2),
                "precision": round(metrics.get('precision', 0) * 100, 1),
                "recall": round(metrics.get('recall', 0) * 100, 1),
                "auc": round(metrics.get('roc_auc', 0), 3)
            })
        else:
            # Return default values if model not trained
            return jsonify({
                "fraud_rate": 0.17,
                "precision": 0,
                "recall": 0,
                "auc": 0
            })
            
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    # Try to load existing model, if not available, don't auto-train (train via web UI)
    load_model()
    
    app.run(debug=True, port=5000)
