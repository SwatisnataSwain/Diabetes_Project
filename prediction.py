from joblib import load
import pandas as pd

loaded_model=load("model_dir\diabetes_model.joblib")

data_dict = pd.DataFrame({
    "Pregnancies":[3],
    "Glucose":[103],
    "BloodPressure" :[72],
    "SkinThickness":[33],
    "Insulin" :[160],
    "BMI":[26.7],
    "DiabetesPedigreeFunction" :[0.121],
    "Age":[22]
})
if (loaded_model.predict(data_dict))==0 :
    print("He/She doesn't have diabetes.")
else :
    print("He/She has diabetes and should see a doctor.")