# Fake News Detector

A full-stack machine learning application that identifies misinformation in news articles using Natural Language Processing (NLP) and the Passive Aggressive Classifier algorithm.

## Project Overview
This project was developed to combat the spread of "fake news" by providing a tool that can analyze text patterns and predict the authenticity of an article. It features a trained ML model and a web-based interface for real-time testing.

## Tech Stack
- **Backend:** Python, Flask
- **Machine Learning:** Scikit-learn, Pandas, NumPy
- **Frontend:** HTML5, CSS3, JavaScript
- **Deployment-Ready:** Includes `requirements.txt` for easy environment setup.

## Repository Structure
- `app.py`: The Flask web application logic.
- `train_model.py`: The script used to process the dataset and train the ML model.
- `model.pkl`: The serialized, pre-trained model ready for production.
- `data/`: Contains the datasets used for training and testing.(Fake.csv/Real.csv) are excluded from the repository for size/privacy but are used locally for training)
- `static/` & `templates/`: Frontend assets and UI layout.
