# full imports
import os
import argparse
import time
import gc

# spark imports
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import col, lower, udf, lit
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS


class AlsRecommender:
    """
    This a collaborative filtering recommender based on the Alternating Least
    Square
    Matrix Factorization implemented by Spark.
    """

    def __init__(self, spark_session, business_path, ratings_path,
                 checkpoint_dir=None):
        """
        :param spark_session: Instance of pyspark.sql.SparkSession.
        :param business_path: [str] path to business file of yelp dataset.
        :param ratings_path:  [str] path to review file of yelp dataset.
        :param checkpoint_dir: [str] checkpoint directory used by the ALS model.
        """
        self.spark = spark_session
        self.sc = spark_session.sparkContext
        self._set_checkpoint_dir(checkpoint_dir)
        self.user_col = "user_id_numeric"
        self.item_col = "business_id_numeric"
        self.rating_col = "rating"

        # Load and prepare data
        print('Loading data')
        business_df = self._load_file(business_path)
        ratings_df = self._load_file(ratings_path)
        business_df, ratings_df = self._add_numeric_ids(business_df,
                                                        ratings_df)
        self.business_df, self.ratings_df = \
            self._rename_and_select_columns(business_df,
                                            ratings_df)
        # instantiate ALS model
        self.model = ALS(
            userCol=self.user_col,
            itemCol=self.item_col,
            ratingCol=self.rating_col,
            coldStartStrategy="drop",
            seed=42)

    def tune_model(self, maxIter, regParams, ranks,
                   split_ratio=(0.6, 0.2, 0.2)):
        """

        :param maxIter: [int] maxIter parameter of Sparks ALS model.
        :param regParams: [list[float]] List of values of regParam parameter of
        Sparks ALS model that represent one dimension of the search grid.
        :param ranks: [list[int]] List of values of regParam parameter of
        Sparks ALS model that represent one dimension of the search grid.
        :param split_ratio: [tuple] Relative size of training, validation and
        test set.
        :return: -
        """
        train, val, test = self.ratings_df.randomSplit(split_ratio, seed=42)
        print("Starting grid search to find best ALS model")
        self.model = tune_ALS(self.model, train, val,
                              maxIter, regParams, ranks)
        predictions = self.model.transform(test)
        evaluator = RegressionEvaluator(metricName="rmse",
                                        labelCol="rating",
                                        predictionCol="prediction")
        rmse = evaluator.evaluate(predictions)
        print('The out-of-sample RMSE of the best tuned model is:', rmse)
        # clean up
        del train, val, test, predictions, evaluator
        gc.collect()

    def set_model_params(self, maxIter, regParam, rank):
        """
        Set model params for pyspark.ml.recommendation.ALS

        Parameters
        :param maxIter: [int] Max number of learning iterations.
        :param regParam: [float] Regularization parameter.
        :param rank: [float] Number of latent factors.
        :return:
        """
        self.model = self.model \
            .setMaxIter(maxIter) \
            .setRank(rank) \
            .setRegParam(regParam)

    def make_recommendations(self, fav_business, n_recommendations):
        """
        Appends user with highest rating for fav_business to user-item-rating
        matrix and fits ALS model on it.
        Prints business recommendations of the model for user.

        :param fav_business: [str] Favorite business of user.
        :param n_recommendations: [int] number of recommendations.
        :return: -
        """
        ## Compute recommendations
        print('Fitting model and making predictions...')
        t0 = time.time()
        recommended_buisness_ids = self._inference(self.model,
                                                   fav_business,
                                                   n_recommendations)

        print('Done in {:.2f}s  \n'.format(time.time() - t0))
        ## Print recommendations
        business_info = self.business_df \
            .filter(col('business_id_numeric').isin(recommended_buisness_ids)) \
            .drop('business_id', 'business_id_numeric') \
            .rdd.map(self._join_row_fields()) \
            .collect()
        print("Top-{0} recommendations because you liked {1}:"
              .format(n_recommendations, fav_business))
        for i in range(len(business_info)):
            print('\t{0}: {1}'.format(i + 1, business_info[i]))

    def _load_file(self, filepath):
        """
        load json file as spark dataframe.
        """
        return self.spark.read.json(filepath)

    def _set_checkpoint_dir(self, check_point_dir=None):
        """
        Sets checkpoint directory for Sparks ALS model.
        If not set, no checkpointing will take place.
        """
        if checkpoint_dir:
            self.sc.setCheckpointDir(checkpoint_dir)

    def _add_numeric_ids(self, business_df, ratings_df):
        """
        Deduces for both dataframes numeric ids from their string id columns and
        appends them to them as a new column.
        :param business_df: Spark dataframe containing information stored in
        the business file of yelp dataset.
        :param ratings_df: Spark dataframe containing information stored in
        the review file of yelp dataset.
        :return: DataFrames with numeric id columns appended to them.
        """
        b_business_id_dic = self._create_b_id_dic(business_df, "business_id")
        b_user_id_dic = self._create_b_id_dic(ratings_df, "user_id")

        business_df = self._append_numeric_ids(business_df,
                                               string_id_col="business_id",
                                               new_col=self.item_col,
                                               b_id_dic=b_business_id_dic)
        ratings_df = self._append_numeric_ids(ratings_df,
                                              string_id_col="user_id",
                                              new_col=self.user_col,
                                              b_id_dic=b_user_id_dic)
        ratings_df = self._append_numeric_ids(ratings_df,
                                              string_id_col="business_id",
                                              new_col=self.item_col,
                                              b_id_dic=b_business_id_dic)
        return business_df, ratings_df

    def _create_b_id_dic(self, df, string_id_col):
        """
        Creates a unique numeric id for each unique string in id_col.
        Then builds a dictionary that maps each numeric id to its string id
        equivalent. Finally returns the broadcast version of this dictionary.

        :param df: Spark dataframes that contains column id_col.
        :param string_id_col: [str] Name of column that contains id strings.
        :return: Broadcast dictionary
        """
        id_strings = self._get_column_values(df, string_id_col)
        return self.sc.broadcast(self.build_id_dic(id_strings))

    def _get_column_values(self, df, col_name):
        """Collects values of the specified column."""
        return df.select(col_name).rdd \
            .map(self._get_row_value(col_name)).collect()

    def _get_row_value(self, col_name):
        """ Wrapper to pass self and col_name"""

        def _inner_get_row_value(row):
            return row[col_name]

        return _inner_get_row_value

    def build_id_dic(self, id_strings):
        """
        Creates a unique id for each unique string in id_strings and builds a
        dictionary that maps identic id strings to identic ids.

        :param id_strings: [list[str]] List containing the id strings.
        :return: [dict] Dictionary that maps id strings to numeric ids.
        """
        unique_strings = set(id_strings)
        return dict(zip(unique_strings, range(len(unique_strings))))

    def _append_numeric_ids(self, df, string_id_col, new_col, b_id_dic):
        """
        Iterates through the rows of string_id_col and uses b_id_dic to
        look up the numeric id that corresponds to the current string id.
        Then adds the numeric ids as new_col to df.
        :param df: Spark dataframe with string_id_col.
        :param string_id_col: [str] Name of the column of df that contains
        string ids.
        :param new_col: [str] Name of the numeric id column that is created.
        :param b_id_dic: Brodcast variable with dicitonary, that maps from
        string ids to numeric ids, as value.
        :return: df with new_col appended to it.
        """
        lookup_in_id_dic = udf(self._lookup_in_dic(dic=b_id_dic.value),
                               returnType=IntegerType())
        return df.withColumn(new_col,
                             lookup_in_id_dic(string_id_col))

    def _lookup_in_dic(self, dic):
        """Wrapper to pass self & dic"""

        def _inner_lookup_in_dic(key):
            return dic[key]

        return _inner_lookup_in_dic

    def _rename_and_select_columns(self, business_df, ratings_df):
        """
        Selects and renames columns from the inputted Dataframes, that are
        relevant for fitting the ALS model and/or for describing the items it
        predicts.
        """
        business_df = business_df.select(self.item_col,
                                         'name',
                                         'address',
                                         'postal_code',
                                         'city')

        ratings_df = ratings_df.select(self.user_col,
                                       self.item_col,
                                       col("stars").alias(self.rating_col))
        return business_df, ratings_df

    def _inference(self, model, fav_business, n_recommendations):
        """
        Predict the rating the user would give every business, based on
        his/her favorite business. Return the business ids of the
        n_recommendations businesses with highest predicted ratings.

        Parameters.
        :param model: Instance of pyspark.ml.recommendation.ALS.
        :param fav_business: [str] Favorite business of user.
        :param n_recommendations: [int] number of recommendations.
        :return: [list] Business ids with highest predicted ratings
        """
        user_ratings_df, user_id, fav_business_ids = \
            self._create_user_df(fav_business)
        # Append user to ratings_df
        self.ratings_df = self.ratings_df.union(user_ratings_df)
        # Fit ALS model
        model = model.fit(self.ratings_df)
        # Choose for which businesses predicitons should be made for the user
        user_inference_df = self._create_user_inference_df(user_id,
                                                           fav_business_ids)
        # Make predictions
        return model.transform(user_inference_df) \
            .select('business_id_numeric', 'prediction') \
            .orderBy('prediction', ascending=False) \
            .drop('prediction') \
            .rdd.map(lambda r: r[0]) \
            .take(n_recommendations)

    def _create_user_df(self, fav_business):
        """
        Create a new user that rated fav_business with highest rating.
        :param fav_business: [str] favorite business of the created user
        """
        user_id = self.ratings_df.agg({"user_id_numeric": "max"}) \
                      .collect()[0][0] + 1  # create id for user
        fav_business_ids = self._regex_matching(fav_business)
        # create user_ratings_df and ensure that it has same column order as
        # ratings_df
        user_ratings_df = self.spark.createDataFrame(
            [(user_id, business_id, 5.0) for business_id in fav_business_ids]) \
            .toDF(*self.ratings_df.columns)
        return user_ratings_df, user_id, fav_business_ids

    def _regex_matching(self, fav_business):
        """
        Search for favorite business in business_df.
        If no match found, throw error, otherwise return matching ids.

        :param fav_business: [str] Favorite business of user.
        :return: [list] Indices of matching businesses.
        """
        print('\tSearching for business "{}" in database'.format(fav_business))
        sql_regex = '{}'.format(fav_business.lower())
        matches_df = self.business_df \
            .filter(lower(col('name')).like(sql_regex))
        if not len(matches_df.take(1)):
            print('Oops! No match is found')
            raise KeyError("Business name could not be found in database")
        else:
            business_ids = self._get_column_values(matches_df,
                                                   "business_id_numeric")
            business_descriptions = matches_df.drop("business_id",
                                                    "business_id_numeric") \
                .rdd.map(self._join_row_fields()).collect()
            print('\tDone. Number of matches: {}:'.format(len(business_ids)))
            for description in business_descriptions:
                print("\t\t{}".format(description))
            return business_ids

    def _join_row_fields(self):
        """ Wrapper to pass self"""

        def inner_join_row_fields(row):
            """
            Joins all fields of row.

            :param row: instance of pyspark.SQL.Row.
            :return: [string] Row fields seperated by commas.
            """
            return ", ".join([str(field_value) for field_value in row])

        return inner_join_row_fields

    def _create_user_inference_df(self, user_id, fav_buisness_ids):
        """
        Create dataframe for user with all items the model was fitted on,
        except the ones corresponding to the users favorite buisness.

        :param user_id: [int] Id of the user.
        :param fav_buisness_ids: [list[int]] Ids of matched businesses.
        :return: Dataframe for making predictions.
        """
        ## Get ids of all non-favorite buisnesses
        inference_df = self.ratings_df \
            .select("business_id_numeric") \
            .distinct() \
            .filter(~col('business_id_numeric').isin(fav_buisness_ids)) \
            .withColumn("user_id_numeric", lit(user_id)) \
            .select(self.ratings_df.drop(
            "rating").columns)  # ensure correct colum order
        return inference_df


def tune_ALS(model, train_df, val_df, maxIter, regParams, ranks):
    """
    Grid search function to select the best model based on RMSE of
    validation data.

    :param model: Sparks ALS model.
    :param train_df: Dataframe used to fit the model.
    :param val_df: Dataframe used to validate the model.
    :param maxIter: [int] maxIter parameter of Sparks ALS model.
    :param regParams: [list[float]] List of values of regParam parameter of
    Sparks ALS model that represent one dimension of the search grid.
    :param ranks: [list[int]] List of values of regParam parameter of
    Sparks ALS model that represent one dimension of the search grid.
    :return: ALS model with lowest RMSE score on validation set
    """

    min_error = float('inf')
    best_rank = -1
    best_reg = 0
    best_model = None
    for rank in ranks:
        for reg in regParams:
            als = model.copy()  # needed to make checkpointing work
            als = als.setMaxIter(maxIter).setRank(rank).setRegParam(reg)

            fitted_model = als.fit(train_df)
            predictions = fitted_model.transform(val_df)
            n_predictions = predictions.rdd.countApprox(timeout=1000)
            evaluator = RegressionEvaluator(metricName="rmse",
                                            labelCol="rating",
                                            predictionCol="prediction")
            rmse = evaluator.evaluate(predictions)
            print("\t Model with rank = {} and regParam = {} resulted in "
                "RMSE of {:<5} (based on ca. {} predicitons (CI = 0.95))"
                  .format(rank, reg, rmse, n_predictions))

            if rmse < min_error:
                min_error = rmse
                best_rank = rank
                best_reg = reg
                best_model = fitted_model
    print("\t Best model: rank = {:<5}, regParam = {:<5}, RMSE {:<5}"
          .format(best_rank, best_reg, min_error))
    return best_model


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Business Recommender",
        description="Run {}".format(os.path.basename(__file__)))
    parser.add_argument('--fav_business', nargs='?',
                        default='Sugar Bowl',
                        help='name of favorite business (5 star rating will '
                             'be assumed)')
    parser.add_argument('--dir', nargs='?',
                        default=r'D:\Python\new_yorker_challenge'
                                r'\new_yorker_challenge\data\yelp_dataset',
                        help='directory that contains input data')
    parser.add_argument('--business_filename', nargs='?',
                        default='debug_1000000_business',
                        # default='yelp_academic_dataset_business.json',
                        help='name of the file that contains information '
                             'about businesses')
    parser.add_argument('--ratings_filename', nargs='?',
                        # default='yelp_academic_dataset_review.json',
                        default='debug_1000000_review',
                        help="name of the file that contains each users "
                             "business ratings")
    parser.add_argument('--checkpoint_directory', nargs='?',
                        default=r'D:\Python\new_yorker_challenge'
                                r'\new_yorker_challenge\data\checkpoints',
                        help='checkpoint directory used by the ALS model')
    parser.add_argument('--n_rec', type=int,
                        default=10,
                        help='number of business recommendations that are '
                             'outputted')

    return parser.parse_args()


if __name__ == '__main__':
    ## get args
    args = parse_args()
    data_dir = args.dir
    business_filename = args.business_filename
    ratings_filename = args.ratings_filename
    fav_business = args.fav_business
    n_rec = args.n_rec
    checkpoint_dir = args.checkpoint_directory

    # set True to only tune model
    tune_recommender = False

    ## initial spark
    spark = SparkSession.builder \
        .appName("business recommender") \
        .getOrCreate()

    ## initial recommender system
    recommender = AlsRecommender(
        spark,
        os.path.join(data_dir, business_filename),
        os.path.join(data_dir, ratings_filename),
        checkpoint_dir=checkpoint_dir)

    if tune_recommender:
        ranks = [20, 30 ,50]
        regParams = [0.1, 0.3, 1]
        # With maxIter > 10, checkpoint_dir must be set for ALS model to
        # prevent Stackoverflow
        recommender.tune_model(maxIter=50,
                               ranks=ranks,
                               regParams=regParams)
    else:
        recommender.set_model_params(maxIter=50,
                                     rank=30,
                                     regParam=0.3)
        recommender.make_recommendations(fav_business, n_rec)

    spark.stop()
