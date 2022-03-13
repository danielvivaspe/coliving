import flask
from flask import jsonify
import pickle
import numpy as np
import pandas as pd

#Cargamos Modelo
def load_model():
    with open(r'C:\Users\oscar\OneDrive\Escritorio\Oscar\Curso The Bridge Data Science\thebridge_ft_nov21\Desafio Tripulaciones\DEFIANT_RECOMMENDER.model', "rb") as archivo_entrada:
        modelo_recomendador = pickle.load(archivo_entrada)
    #print(list_models)
    return modelo_recomendador
#Predecimos con el modelo cargado:

def prediction(model=load_model()):
    recomendacion=model.predict()
    return recomendacion

load_model()


#Utilizamos Flask para subir el modelo a una web

app = flask.Flask(__name__)
app.config["DEBUG"] = True

#Ponemos el título del home:

@app.route('/')
def home():
    return "<h1>MODELO RECOMENDADOR</h1><p>modelo recomendador de actividades a realizar en un cohousing</p>"

if __name__ == '__main__':
    app.run()

# Devolvemos en un json los resultados del modelo recomendador:

@app.route('/api/model/recomendador', methods=['GET'])
def recomendador():
    modelo = load_model()  
    prediccion_model1 = prediction()
    return jsonify(prediccion_model1)


