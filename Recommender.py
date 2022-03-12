class Recommender:

    def __init__(self):
        self.model = self.load_model()

    def load_model(self):
        with open('model/coliving_v1.model', 'rb') as model:
            return model

    def get_prediction(self, userID, activityID):
        return self.model.predict(userID, activityID)

    def get_recommendations(self, userId, dataframe, n_recommendations, column_iid= None, column_uid= None):
        """
        This functions will use a trained algorithm to find the n top list of recommended items for a given userID.

        Parameters
        -----------

        userId (int): the user ID of the person that we want recommendations for.

        dataframe (object): the DataFrame containing three columns; userID, itemID and rating.

        n_rcommendations (int): the number of items recommended.

        column_iid (string): name of the column containing the item ID.

        column_uid (string): name of the column containing the user ID.


        return
        ------

        List of ID of items that an specific user will like.

        """
        item_ids = dataframe[column_iid].to_list()
        items_finished = dataframe[dataframe[column_uid] == userId][column_iid]

        items_no_finished = []
        for item in item_ids:
            if item not in items_finished:
                items_no_finished.append(item)

        preds = []
        for item in items_no_finished:
            preds.append(self.model.predict(uid=userId, iid=item))

        recommendations_rating = {pred[1]:pred[3] for pred in preds}

        order_dict = {k: v for k, v in sorted(recommendations_rating.items(), key=lambda item: item[1])}

        top_predictions = list(order_dict.keys())[:n_recommendations]
        
        return top_predictions
        
    def check_activities_user(userId, dataframe, n, column_rating= None, column_uid= None):
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
        return dataframe[dataframe[column_uid] ==userId].sort_values(column_rating, ascending=False)[:n]