import pickle

import pandas as pd


class Recommender:
    dict_activities = {
        1: 'YOGA',
        2: 'NATACION',
        3: 'BAILE',
        4: 'GOLF',
        5: 'GIMNASIO',
        6: 'TIRO CON ARCO',
        7: 'ZUMBA',
        8: 'TENIS',
        9: 'CLUB DE LECTURA',
        10: 'CLUB DE ESCRITURA',
        11: 'PINTURA',
        12: 'MUSICA',
        13: 'MACRAME',
        14: 'INFORMATICA',
        15: 'JARDINERIA',
        16: 'MANUALIDADES',
        17: 'IDIOMAS',
        18: 'COCINA',
        19: 'COCTELERIA',
        20: 'CERVECERIA ARTESANAL',
        21: 'CATAS DE COMIDA Y BEBIDA',
        22: 'BINGO',
        23: 'PARCHIS',
        24: 'AJEDREZ',
        25: 'TEATRO'
    }

    def __init__(self):
        self.data = self.load_data()
        self.model = self.load_model()
        self.triplets = self.preprocess_form()

    def load_model(self):
        file = open('model/model_prueba.model', 'rb')
        return pickle.load(file)

    # Importamos Datos:
    def load_data(self):
        return pd.read_csv("data/coliving_TEMP.csv")

    def get_prediction(self, userID, activityID):
        '''
        This function predcits the expected rating for a given activityId and userId

        Output (float): a value from 1 to 5 representing the expected rating
        '''
        return self.model.predict(userID, activityID).est

    def get_recommendations(self, userId, n_recommendations=-1):
        """
        This functions will use a trained algorithm to find the n top list of recommended items for a given userID.
        Parameters
        -----------
        userId (int): the user ID of the person that we want recommendations for.
        n_recommendations (int): the number of items recommended.
        return
        ------
        List of ID of items that an specific user will like.
        """
        item_ids = self.triplets['itemId'].to_list()
        items_finished = self.triplets[self.triplets['userId'] == userId]['itemId']

        items_no_finished = []
        for item in item_ids:
            if item not in items_finished:
                items_no_finished.append(item)

        preds = []
        for item in items_no_finished:
            preds.append(self.model.predict(uid=userId, iid=item))

        recommendations_rating = {pred[1]: pred[3] for pred in preds}

        order_dict = {k: v for k, v in sorted(recommendations_rating.items(), key=lambda item: item[1])}

        top_predictions = list(order_dict.keys())

        if n_recommendations != -1:
            top_predictions = top_predictions[:n_recommendations]

        return top_predictions

    def check_recommended_item_name(self, list):
        """
        This functions will show the names of the n top rated items for a given userID.
        Parameters
        -----------
        list (object): the list of n recommended itemId.
        return
        ------
        A list with the n names of the itemId recommended to the given userId.
        """

        return [self.dict_activities[i] for i in list]

    def check_activities_user(self, userId, n):
        """
        This functions will show the n top rated items for a given userID.
        Parameters
        -----------
        userId (int): the user ID of the person that we want recommendations for.
        dataframe (object): the DataFrame containing three columns; userID, itemID and rating.
        n (int): number of top rated items to show.
        column_rating (string): name of the column containing the item rating.
        column_uid (string): name of the column containing the user ID.
        return
        ------
        A dataframe with the n top rated items by that given user.
        """
        dataframe = self.data[self.data['userId'] == userId].sort_values('rating', ascending=False)[:n]

        # we create a dictionary to map the name of activities and change it for their activity code

        dataframe['itemName'] = dataframe['itemId'].map(self.dict_activities)

        return dataframe

    def preprocess_form(self):
        """
        This functions will preprocess the original csv from the Google form and get it ready for the model.
        Parameters
        -----------
        dataframe (object): the DataFrame containing three columns; userID, itemID and rating.
        return
        ------
        A new dataframe ready for the model.
        """
        # we clean the df from only nan answers and we erase the first two columns as they are not answers for the algorithm
        self.data.drop(['Timestamp', '¿Qué edad tienes?'], axis=1, inplace=True)
        self.data.dropna(how='all', inplace=True)

        # we need to change the answers from the form from str to int
        dicc_formulario = {
            '1 No me gusta': 1,
            '2': 2,
            '3': 3,
            '4': 4,
            '5 Me encanta': 5
        }

        for i in self.data.columns:
            self.data[i] = self.data[i].map(dicc_formulario)

        # we transorm the dataframe to get the triplet column format we need to feed our algorithm
        df_new = pd.DataFrame([(1, 1, 1)])
        for i in self.data.index:
            for j in self.data.columns:
                if self.data[j][i] == 'NaN':
                    pass
                else:
                    df_new = df_new.append([(i, j, self.data[j][i])])

        df_new.dropna(how='any', inplace=True)
        df_new = df_new[1:]

        # we cahange the columns names
        df_new.columns = ['userId', 'itemId', 'rating']

        # we create a dictionary to map the name of activities and change it for their activity code
        dict_activities = {
            'Lista de actividades: [YOGA]': 1,
            'Lista de actividades: [NATACIÓN ]': 2,
            'Lista de actividades: [BAILE ]': 3,
            'Lista de actividades: [GOLF ]': 4,
            'Lista de actividades: [GIMNASIO ]': 5,
            'Lista de actividades: [TIRO CON ARCO ]': 6,
            'Lista de actividades: [ZUMBA ]': 7,
            'Lista de actividades: [TENIS]': 8,
            'Lista de actividades: [CLUB DE LECTURA]': 9,
            'Lista de actividades: [CLUB DE ESCRITURA]': 10,
            'Lista de actividades: [PINTURA]': 11,
            'Lista de actividades: [MÚSICA ]': 12,
            'Lista de actividades: [MACRAMÉ]': 13,
            'Lista de actividades: [INFORMÁTICA]': 14,
            'Lista de actividades: [JARDINERÍA]': 15,
            'Lista de actividades: [MANUALIDADES]': 16,
            'Lista de actividades: [IDIOMAS]': 17,
            'Lista de actividades: [COCINA]': 18,
            'Lista de actividades: [COCTELERÍA]': 19,
            'Lista de actividades: [CERVECERÍA ARTESANAL]': 20,
            'Lista de actividades: [CATAS DE COMIDA Y BEBIDA]': 21,
            'Lista de actividades: [BINGO]': 22,
            'Lista de actividades: [PARCHIS]': 23,
            'Lista de actividades: [AJEDREZ]': 24,
            'Lista de actividades: [TEATRO]': 25
        }
        df_new['itemId'] = df_new['itemId'].map(dict_activities)

        return df_new
