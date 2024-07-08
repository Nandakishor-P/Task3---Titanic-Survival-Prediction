# Titanic Survival Predictor

This project implements a machine learning model to predict survival on the Titanic, along with a simple GUI for making predictions.

## Overview

The Titanic Survival Predictor consists of two main components:
1. A Random Forest Classifier trained on the Titanic dataset
2. A Tkinter-based GUI for inputting passenger details and getting survival predictions

## Features

- Data preprocessing and feature engineering
- Model training using Random Forest Classifier
- K-Fold Cross-validation
- Model evaluation using accuracy, precision, and recall metrics
- GUI for easy prediction input

## Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn
- joblib
- tkinter

## Installation

1. Clone this repository:
```bash
  git clone https://github.com/Nandakishor-P/Task3---Titanic-Survival-Prediction
```
2. Install the required packages:
```bash
  python -r requirements.txt
```
## Usage

1. Train the model:
This will create `titanic_model.pkl` and `model_columns.pkl`.
```bash
  python train.py
```
3. Run the GUI:
```bash
  python load.py
```
3. Enter the passenger details in the GUI and click "Predict" to see the survival prediction.

## File Descriptions

- `train_model.py`: Script for data preprocessing, model training, and evaluation
- `prediction_gui.py`: GUI implementation for making predictions
- `Dataset/train.csv`: Training data (not included in repo)
- `Dataset/test.csv`: Test data (not included in repo)
- `titanic_model.pkl`: Saved trained model
- `model_columns.pkl`: Saved model columns for preprocessing new data

## Model Performance

The model achieves the following performance on the test set:
- Accuracy: [0.8076923076923077]
- Precision: [0.696969696969697]
- Recall: [0.8214285714285714]

## Future Improvements

- Feature importance analysis
- Hyperparameter tuning
- Deployment as a web application

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)
