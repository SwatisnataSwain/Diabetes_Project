import pandas as pd
from sklearn.model_selection import train_test_split ,KFold,StratifiedKFold , cross_val_score
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.preprocessing import StandardScaler , MinMaxScaler
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
import matplotlib as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from joblib import dump,load


df = pd.read_csv(r"C:\Users\Swati\OneDrive\Desktop\PYTHON\Diabetes_Project\diabetes.csv")

cols = ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]
for i in cols:
    print(i,(df[i]==0).sum())

df[cols] = df[cols].replace(0,np.nan)
print(df.isnull().sum())

print(df.fillna(df.mean(numeric_only=True),inplace=True))


X = df.drop(columns="Outcome")
y = df["Outcome"]
print("Shape Dataset -> ", df.shape)
print("Shape X -> ", X.shape)
print("Shape y -> ", y.shape)

def check_ratio(y):
    d = {
        "count" : y.value_counts(),
        "percent" : round(y.value_counts(normalize=True), 3)
    }
    y_ratio = pd.DataFrame(d)
    return y_ratio
print(check_ratio(y))


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

d = {
    "y_percent" : round(y.value_counts(normalize=True), 4),
    "y_train_percent" : round(y_train.value_counts(normalize=True), 4),
    "y_test_percent" : round(y_test.value_counts(normalize=True), 4)
}
y_ratio = pd.DataFrame(d)
print(y_ratio)

pipeline = Pipeline([
    ("Scaler",StandardScaler()),
    ("model",LogisticRegression())
])

print(pipeline.fit(X_train,y_train))

dump(pipeline,"model_dir/diabetes_model.joblib")