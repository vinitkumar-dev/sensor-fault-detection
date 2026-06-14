🚨 Fault Detection System
🌐 Live Demo

👉 https://sensor-fault-detection-1-51br.onrender.com

📌 Overview

This project is an end-to-end Machine Learning system for detecting faults in sensors or industrial systems based on input data.

It predicts whether a machine/sensor is faulty or healthy, and demonstrates a complete ML pipeline with production-ready project structure.

⚡ Features
Data Ingestion pipeline
Data Validation
Data Transformation
Model Training using multiple algorithms
Model Evaluation & selection of best model
Prediction pipeline (real-time inference)
Modular and scalable code structure
Logging and custom exception handling
🧠 Technologies Used
Python
NumPy
Pandas
Scikit-learn
XGBoost
Matplotlib
Seaborn
Flask (for deployment)
Render (for hosting)
🏗️ Project Structure
fault_detection/
│
├── notebook/               # Experiments and EDA
├── src/                    # Core source code
│   ├── components/         # ML pipeline components
│   ├── pipeline/           # Training & prediction pipeline
│   ├── exception.py        # Custom exception handling
│   ├── logger.py           # Logging system
│   └── utils.py            # Helper functions
│
├── artifacts/              # Saved models & outputs
├── logs/                   # Log files
├── requirements.txt
├── setup.py
├── app.py                  # Flask application
├── README.md
└── .gitignore
