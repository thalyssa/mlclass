import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('diabetes_dataset.csv')
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
scaler = StandardScaler()

filledData = imputer.fit_transform(data)
filledData = scaler.fit_transform(filledData)

new_data = pd.DataFrame(filledData, columns=["Pregnancies", "Glucose", "BloodPressure",
                                             "SkinThickness", "Insulin", "BMI",
                                             "DiabetesPedigreeFunction", "Age", "Outcome"])

new_data.to_csv('diabetes_dataset.csv')
