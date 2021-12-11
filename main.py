# pip3 install virtualenv
# virtualenv env
# pip3 install flask flask-sqlalchemy
# Flask-SQLAlchemy installation (pip install Flask-SQLAlchemy)

from flask import Flask, render_template, request, abort
from flask_sqlalchemy import SQLAlchemy

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score

import numpy as np
import pandas as pd
import os

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///database/data_heart_predicted.sqlite3"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db1 = SQLAlchemy(app)
db2 = SQLAlchemy(app)

df = pd.read_excel("static/resources/files/Coeur.xlsx")


# CLASS --------------------------------------------------------

# CREATE TABLE 
class Heart_dataset(db1.Model):
    __tablename__ = 'heart_dataset'
    id = db1.Column(db1.Integer, primary_key=True)
    age = db1.Column(db1.Float, nullable=False)
    sexe = db1.Column(db1.Float, nullable=False)
    tdt = db1.Column(db1.Integer, nullable=False)
    par = db1.Column(db1.Float, nullable=False)
    cholesterol = db1.Column(db1.Float, nullable=False)
    gaj = db1.Column(db1.Integer, nullable=False)
    ecg = db1.Column(db1.Integer, nullable=False)
    fcmax = db1.Column(db1.Float, nullable=False)
    angine = db1.Column(db1.Integer, nullable=False)
    depression = db1.Column(db1.Float, nullable=False)
    pente = db1.Column(db1.Integer, nullable=False)
    coeur = db1.Column(db1.Integer, nullable=False)


# CREATE TABLE 
class Heart_predicted(db2.Model):
    __tablename__ = 'heart_predicted'
    id = db2.Column(db2.Integer, primary_key=True)
    age = db2.Column(db2.Float, nullable=False)
    sexe = db2.Column(db2.Integer, nullable=False)
    tdt = db2.Column(db2.Integer, nullable=False)
    par = db2.Column(db2.Float, nullable=False)
    cholesterol = db2.Column(db2.Float, nullable=False)
    gaj = db2.Column(db2.Integer, nullable=False)
    ecg = db2.Column(db2.Integer, nullable=False)
    fcmax = db2.Column(db2.Float, nullable=False)
    angine = db2.Column(db2.Integer, nullable=False)
    depression = db2.Column(db2.Float, nullable=False)
    pente = db2.Column(db2.Integer, nullable=False)
    coeur = db2.Column(db2.Integer, nullable=False)   



class Memory_recoder():
    data_recoder = {}
    
# CLASS --------------------------------------------------------


memory_recoder = Memory_recoder()


# METHODES -----------------------------------------------------

def quantitatives_variables_recoder(df):
    data_Of_Memory = {} 
    
    # Make data memory
    for i in df.columns[df.dtypes == object]:
        k = 0
        for j in df[i].unique():
            data_Of_Memory[j] = k
            k = k + 1
        
        memory_recoder.data_recoder[i] = data_Of_Memory.copy()
        data_Of_Memory.clear()   
    # --------------------------

    # Recode categorial values
    for key, value in memory_recoder.data_recoder.items():
        df[key] = df[key].replace(value)
        
    
    return df, memory_recoder
    # -------------------------


def makePreprocessing(df):
    # Suppression des doublons
    df.drop_duplicates()
    # Suppression des valeurs manquantes
    df = df.dropna()
    
    #print(df.head())
    #print(df.describe())
    
    # Vérification des constantes
    print("Constantes: ", (df.nunique() == 1).sum())
    
    # Suppression des doublons
    df.drop_duplicates()
    
    # Standardisation des variables quantitatives
    for col in df.drop('CŒUR', axis = 1).select_dtypes(exclude='object').columns:
      df[col] = df[col] / df[col].max()
    
    df, memory_recoder = quantitatives_variables_recoder(df)

    #print(df, memory_recoder.data_recoder)
    return df, memory_recoder


# Use dataset to make a database with dataset values
def makeDatabase_from_dataset(df):
    # if database file is empty
    if (os.stat("database/data_heart_predicted.sqlite3").st_size) == 0:
        # create database if not exists
        db1.create_all()
        db2.create_all()
    
        for elt in df.index:
            print(elt,"\n")
            age = df["AGE"][elt]
            sexe = df["SEXE"][elt]
            tdt = df["TDT"][elt]
            par = df["PAR"][elt]
            cholesterol = df["CHOLESTEROL"][elt]
            gaj = df["GAJ"][elt]
            ecg = df["ECG"][elt]
            fcmax = df["FCMAX"][elt]
            angine = df["ANGINE"][elt]
            depression = df["DEPRESSION"][elt]
            pente = df["PENTE"][elt]
            coeur = df["CŒUR"][elt]

            new_patient = Heart_dataset(age=float(age), sexe=int(sexe), tdt=int(tdt), par=float(par), cholesterol=float(cholesterol), gaj=int(gaj), ecg=int(ecg), fcmax=float(fcmax), angine=int(angine), depression=float(depression), pente=int(pente), coeur=int(coeur))
        
            try:
                db1.session.add(new_patient)
                db1.session.commit()
            except:
                return "errueur d'insertion"



def predict(dataElement):
    # get values
    patient = pd.DataFrame.from_dict(dataElement)
 
    y = df["CŒUR"].values
    x = df.drop("CŒUR", axis=1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
    
    # Entrainement du model
    model = LogisticRegression(random_state=0).fit(x_train, y_train)

    #prédiction
    return model.predict(patient)
    
# METHODES -----------------------------------------------------


# VIEWS --------------------------------------------------------

# Call App home page
@app.route("/")
def index():
    return render_template('pages/index.html', page="home")


# Call App home page
@app.route("/home")
def home():
    return render_template('pages/index.html', page="home")


# Call App home page
@app.route("/heart_register")
def heart_register():
    return render_template('pages/heart_register.html', error=False, page="register", memory_recoder=memory_recoder.data_recoder)


@app.route("/registerSummary", methods = ["POST", "GET"])
def registerSummary():
   
    if request.method == "POST":
        list_of_values = { "age": request.form["age"], "sexe": request.form["sexe"],  "tdt": request.form["tdt"], "par": request.form["par"], "cholesterol": request.form["cholesterol"], "gaj": request.form["gaj"], "ecg": request.form["ecg"], "fcmax": request.form["fcmax"], "angine": request.form["angine"], "depression": request.form["depression"], "pente": request.form["pente"] }
        list_of_values_for_prediction = { "age": [request.form["age"]], "sexe": [request.form["sexe"]],  "tdt": [request.form["tdt"]], "par": [request.form["par"]], "cholesterol": [request.form["cholesterol"]], "gaj": [request.form["gaj"]], "ecg": [request.form["ecg"]], "fcmax": [request.form["fcmax"]], "angine": [request.form["angine"]], "depression": [request.form["depression"]], "pente": [request.form["pente"]] }
        variables_quantitatives = ["age","par","cholesterol","fcmax","depression"]

        # verification of float values
        for key, value in list_of_values.items():
            if key in variables_quantitatives:
                try:
                    list_of_values[key] = float(value)
                except:
                    return render_template('pages/heart_register.html', error="badEntryFloat", page="register", memory_recoder=memory_recoder.data_recoder)

        # prediction
        prediction = predict(list_of_values_for_prediction)
        
        try:
            new_patient = Heart_predicted(age=list_of_values["age"], sexe=list_of_values["sexe"], tdt=list_of_values["tdt"], par=list_of_values["par"], cholesterol=list_of_values["cholesterol"], gaj=list_of_values["gaj"], ecg=list_of_values["ecg"], fcmax=list_of_values["fcmax"], angine=list_of_values["angine"], depression=list_of_values["depression"], pente=list_of_values["pente"], coeur=int(prediction[0]))         
            db2.session.add(new_patient)
            db2.session.commit()
            return render_template("pages/register_summary.html", page="register", list_of_values=list_of_values, original_list=refactor_list(list_of_values), prediction=prediction)
        except:
            abort(404)
    
    else:
        return render_template('pages/heart_register.html', error=False, page="register", memory_recoder=memory_recoder.data_recoder)


# Refactor list of value
# to introduce normal quantitative value
def refactor_list(listElement):
    
    # get data_recoder
    memory = memory_recoder.data_recoder
    listElement2 = listElement.copy()
   
    for keyMemory, valueMemory in memory.items():
        for key, value in listElement.items():
           
            if keyMemory.lower() == key.lower():

                for k, v in valueMemory.items():
                    if int(v) == int(value):
                        listElement2[key] = k
    
    return listElement2


# Call administration page with all data
def callAdminSpace(supression=False, supressionCalled=False):
    patients = Heart_predicted.query.all()
    return render_template('pages/administration.html', page="administration", patients=patients, supression=supression, supressionCalled=supressionCalled, columns=df.columns.values)


# Call admin page
# Using to manage database
@app.route("/administration")
def adminSpace():
    try:
        patients = Heart_predicted.query.all()
        return render_template('pages/administration.html', page="administration", patients=patients, columns=df.columns.values)
    except:
        pass


# App route
# Go to dataset page
@app.route("/dataset-page")
def dataset_page():
    
    try:
        # get all value
        database_of_dataset = Heart_dataset.query.all()
        return render_template("pages/dataset_page.html", page="dataset_page", data=database_of_dataset, columns=df.columns.values)
    except:
        abort(404)
        #return abort(404)


# Call delete page
@app.route("/delete/<int:id>")
def delete(id):

    patientDelete = Heart_predicted.query.get_or_404(id)
  
    try:
        db2.session.delete(patientDelete)
        db2.session.commit()
        return callAdminSpace(supression=True, supressionCalled=True)
    except:
        return callAdminSpace(supression=False, supressionCalled=True)


# Call Error page 404
@app.errorhandler(404)
def page_not_found(error):
    return render_template('pages/errors/404.html'), 404

# VIEWS --------------------------------------------------------


# App Launcher
if __name__ == "__main__":
    # Preprocessing
    df, memory_recoder = makePreprocessing(df)
    
    # insert dataset in database
    makeDatabase_from_dataset(df)
  
    app.run(debug=True)
