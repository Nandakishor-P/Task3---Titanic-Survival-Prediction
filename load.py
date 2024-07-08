import pandas as pd
import numpy as np
import joblib
import tkinter as tk
from tkinter import messagebox

# Load the trained model
model = joblib.load('titanic_model.pkl')
model_columns = joblib.load('model_columns.pkl')

def predict_survival():
    # Get user inputs
    pclass = int(pclass_var.get())
    sex = sex_var.get()
    age = float(age_var.get())
    sibsp = int(sibsp_var.get())
    parch = int(parch_var.get())
    fare = float(fare_var.get())
    embarked = embarked_var.get()
    
    # Create a DataFrame with the inputs
    input_data = pd.DataFrame({'Pclass': [pclass], 'Sex': [sex], 'Age': [age], 'SibSp': [sibsp], 'Parch': [parch], 'Fare': [fare], 'Embarked': [embarked]})
    input_data = pd.get_dummies(input_data)
    input_data = input_data.reindex(columns=model_columns, fill_value=0)
    
    # Make prediction
    prediction = model.predict(input_data)
    result = 'Survived' if prediction[0] == 1 else ' not survived'
    
    # Show the result
    messagebox.showinfo("Prediction Result", f"The passenger would have {result}.")

# Create the main window
root = tk.Tk()
root.title("Titanic Survival Prediction")

# Add input fields
tk.Label(root, text="Pclass:").grid(row=0, column=0)
pclass_var = tk.StringVar()
tk.Entry(root, textvariable=pclass_var).grid(row=0, column=1)

tk.Label(root, text="Sex:").grid(row=1, column=0)
sex_var = tk.StringVar()
tk.Entry(root, textvariable=sex_var).grid(row=1, column=1)

tk.Label(root, text="Age:").grid(row=2, column=0)
age_var = tk.StringVar()
tk.Entry(root, textvariable=age_var).grid(row=2, column=1)

tk.Label(root, text="SibSp:").grid(row=3, column=0)
sibsp_var = tk.StringVar()
tk.Entry(root, textvariable=sibsp_var).grid(row=3, column=1)

tk.Label(root, text="Parch:").grid(row=4, column=0)
parch_var = tk.StringVar()
tk.Entry(root, textvariable=parch_var).grid(row=4, column=1)

tk.Label(root, text="Fare:").grid(row=5, column=0)
fare_var = tk.StringVar()
tk.Entry(root, textvariable=fare_var).grid(row=5, column=1)

tk.Label(root, text="Embarked:").grid(row=6, column=0)
embarked_var = tk.StringVar()
tk.Entry(root, textvariable=embarked_var).grid(row=6, column=1)

# Add predict button
tk.Button(root, text="Predict", command=predict_survival).grid(row=7, columnspan=2)

# Run the application
root.mainloop()
