# Importaciones
import flask
from flask import jsonify

from Recommender import Recommender

# Importando funciones trabajadas de ML:

##from Defiant_Recommender_notebook import preprocess_form
# from Defiant_Recommender_notebook import defiant_recommender
# from Defiant_Recommender_notebook import check_activities_user


recommender = Recommender()

app = flask.Flask(__name__)
app.config["DEBUG"] = True


# Ponemos el título del home:

@app.route('/')
def home():
    return "<h1>Recommender API</p>"


# Devolvemos en un json los resultados del modelo recomendador:
@app.route('/api/model/recommender/<string:userID>', methods=['GET'])
def recomendacion(userID):
    rec = recommender.get_recommendations(userID)
    return jsonify(rec)


# Devolvemos en un json el top 5 de actividades mejor puntuadas por el usuario X:

@app.route('/api/model/predictor/user/<string:userID>/activity/<string:actID>', methods=['GET'])
def actividades(userID, actID):
    prediction = recommender.get_prediction(userID, actID)
    return jsonify(prediction)


if __name__ == '__main__':
    app.run()
