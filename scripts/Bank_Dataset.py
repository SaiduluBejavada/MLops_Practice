#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# Importing libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
from flask import Flask, request, jsonify
import os
import logging


# In[ ]:


# Set up logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)


# In[ ]:


# Set the model path to save in the same directory as this script
MODEL_PATH = os.getenv("MODEL_PATH", "bank_model.pkl")
#DATA_URL = os.getenv("DATA_URL", "/workspaces/MLops_Practice/scripts/bank.csv")  # Path to your CSV file
DATA_URL = os.getenv("DATA_URL", "./bank.csv")


# In[ ]:


def load_model():
    """Load the trained model from a pickle file."""
    try:
        with open(MODEL_PATH, 'rb') as file:
            model = pickle.load(file)
        logging.info("Model loaded successfully.")
        return model
    except Exception as e:
        logging.error("Error loading model: %s", e)
        return None


# In[ ]:


def load_data(url):
    """Load dataset from a given URL."""
    try:
        logging.info(f"Loading data from {url}")
        
        # Check if the file exists and log its size
        if not os.path.exists(url):
            logging.error("Data file does not exist.")
            return None
        
        file_size = os.path.getsize(url)
        logging.info(f"File size: {file_size} bytes")

        data = pd.read_csv(url, sep=',')  # Use comma as the separator
        if data.empty:
            logging.error("Loaded data is empty.")
            return None
        
        logging.info("Data loaded successfully.")
        logging.info(f"Columns in loaded data: {data.columns.tolist()}")  # Log the columns
        return data
    except Exception as e:
        logging.error("Error loading data: %s", e)
        return None


# In[ ]:


def preprocess_data(data):
    """Preprocess the data (if needed)."""
    # Save the target variable
    target = data['y']
    
    # Convert categorical variables to dummy variables, dropping the target
    data = pd.get_dummies(data.drop('y', axis=1), drop_first=True)
    
    # Add the target variable back
    data['y'] = target
    return data


# In[ ]:


def train_model(data):
    """Train a machine learning model."""
    if 'y' not in data.columns:
        raise ValueError("Target column 'y' not found in the dataset.")

    X = data.drop('y', axis=1)  # Features
    y = data['y']                # Target variable

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"Model accuracy: {accuracy:.2f}")

    return model


# In[ ]:


def save_model(model):
    """Save the trained model as a pickle file."""
    with open(MODEL_PATH, 'wb') as file:
        pickle.dump(model, file)
    logging.info(f"Model saved as {MODEL_PATH}")


# In[ ]:


@app.route('/')
def home():
    """Home route."""
    return jsonify({"message": "Welcome to the Bank Model API. Use /train to train the model and /predict to make predictions."})


# In[ ]:


@app.route('/predict', methods=['POST'])
def predict():
    """Predict endpoint."""
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400

    df = pd.DataFrame(data)
    try:
        df = preprocess_data(df)
        model = load_model()
        predictions = model.predict(df)
        return jsonify(predictions.tolist())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# In[ ]:


@app.route('/train', methods=['POST'])
def train():
    """Train endpoint."""
    data = load_data(DATA_URL)
    
    if data is not None:
        processed_data = preprocess_data(data)
        model = train_model(processed_data)
        save_model(model)
        return jsonify({"message": "Model trained and saved."})
    else:
        return jsonify({"error": "Data loading failed."}), 500


# In[ ]:


if __name__ == "__main__":
    # Train the model when the script is run
    data = load_data(DATA_URL)
    if data is not None:
        processed_data = preprocess_data(data)
        model = train_model(processed_data)
        save_model(model)
    else:
        logging.error("Initial data loading failed.")

    # Run the Flask app
    app.run(host='0.0.0.0', port=50702)

