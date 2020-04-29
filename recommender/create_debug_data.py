# full imports
import argparse
import os

# spark imports
from pyspark import StorageLevel
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# other imports
from pathlib import Path


class DebugDataCreator():
    """
    Creates small versions of specified datasets.
    """

    def __init__(self, spark_session, data_dir, business_filename,
                 ratings_filename):
        """
        :param spark_session: Instance of pyspark.sql.SparkSession.
        :param data_dir: [str] Directory from which filenames are read and to
        which debug version are written.
        :param: business_filename: [str] Name of json file that contains
        business information.
        :param: ratings_filename: [str] Name of json file that contains reviews.
        """
        self.spark = spark_session
        self.sc = spark_session.sparkContext
        self.data_dir = data_dir
        buisness_path = os.path.join(data_dir, business_filename)
        ratings_path = os.path.join(data_dir, ratings_filename)
        self.business_df = self._load_file(buisness_path)
        self.ratings_df = self._load_file(ratings_path)

    def create_and_write_debug_data(self, limit=1000, overwrite=True):
        """
        Saves first >limit< rows of both class dataframes in new
        json file.

        :param limit: [int] Limit that specifies how many rows each debug
        dataframe will have.
        :param overwrite: Toggle if existing files should be overwritten or not
        :return: -
        """
        suffix = "debug_{}".format(str(limit))
        debug_dir = os.path.join(data_dir, suffix)
        if overwrite:
            mode = 'overwrite'
        else:
            mode = None

        debug_ratings = self.ratings_df.limit(limit).select("user_id",
                                                            "business_id",
                                                            "stars")
        debug_business = self.business_df.join(debug_ratings,
                                               on="business_id",
                                               how='left_semi')
        debug_business = debug_business.select("business_id",
                                               "name",
                                               "address",
                                               "postal_code",
                                               "city")

        debug_ratings.write.json(debug_dir + '_review', mode=mode)
        debug_business.write.json(debug_dir + '_business', mode=mode)


    def _load_file(self, filepath):
        """
        Load json file as Spark dataframe.
        """
        return self.spark.read.json(filepath)

if __name__ == '__main__':
    # configuration
    limit = 10000
    data_dir = r'D:\Python\new_yorker_challenge' \
               r'\new_yorker_challenge\data\yelp_dataset'
    business_filename = 'yelp_academic_dataset_business.json'
    ratings_filename = 'yelp_academic_dataset_review.json'

    # build spark session
    spark = SparkSession.builder \
        .appName("debug data creator") \
        .getOrCreate()
    # create debug data
    debug_data_creator = DebugDataCreator(spark, data_dir,
                                          business_filename,
                                          ratings_filename)
    debug_data_creator.create_and_write_debug_data(limit)
    spark.stop()
