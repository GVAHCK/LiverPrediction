from flask import Flask, flash, request, redirect, url_for, render_template, send_file
import os
import warnings
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
warnings.filterwarnings("ignore")

PEOPLE_FOLDER = os.path.join('static','images')
app = Flask(__name__)

app.config['SECRET_KEY'] = 'secret-key-goes-here'

@app.route("/", methods=['GET'])
def index():
    return render_template('home.html')

@app.route("/contact", methods=['GET'])
def Contact():
    return render_template('contact.html')

def Model():
    data = pd.read_csv("./indian_liver_patient.csv")
    data['Dataset'] = data['Dataset'].map({2:0,1:1})
    data['Albumin_and_Globulin_Ratio'].fillna(value=0.94, inplace=True)
    data_features = data.drop(['Dataset'],axis=1)
    data_num_features = data.drop(['Gender','Dataset'],axis=1)
    scaler = StandardScaler()
    cols = list(data_num_features.columns)
    data_features_scaled = pd.DataFrame(data=data_features)
    data_features_scaled[cols] = scaler.fit_transform(data_features[cols])
    data_exp = pd.get_dummies(data_features_scaled)
    X=data_exp
    y=data['Dataset'] 
    X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size = 0.20,random_state = 42)
    knn = KNeighborsClassifier(n_neighbors = 22)
    knn.fit(X_train, Y_train)
    return knn

@app.route("/check-up", methods=['GET', 'POST'])
def Check_Up():
    if request.method == 'POST':
        loaded_model = Model()
        dt = {}
        gender = int(request.form.get('gender'))
        dt['Age'] = int(request.form.get('age'))
        dt['Total_Bilirubin'] = float(request.form.get('tb'))
        dt['Direct_Bilirubin'] = float(request.form.get('db'))
        dt['Alkaline_Phosphotase'] = int(request.form.get('ap'))
        dt['Alamine_Aminotransferase'] = int(request.form.get('aa'))
        dt['Aspartate_Aminotransferase'] = int(request.form.get('asa'))
        dt['Total_Protiens'] = float(request.form.get('tp'))
        dt['Albumin'] = float(request.form.get('ab'))
        dt['Albumin_and_Globulin_Ratio'] = float(request.form.get('ag'))
        if gender == 1:
            dt['Gender_Female'] = 0
            dt['Gender_Male'] = 1
        else:
            dt['Gender_Female'] = 1
            dt['Gender_Male'] = 0
        testData = pd.DataFrame({'x':dt}).transpose()
        prediction = loaded_model.predict(testData)[0]
        if(int(request.form.get('age')) < 25):
            return render_template('check.html', result=0)
        return render_template('check.html', result=prediction)
    return render_template('check.html')

#JUST DO IT!!!
if __name__=="__main__":
    app.run(port="9000")