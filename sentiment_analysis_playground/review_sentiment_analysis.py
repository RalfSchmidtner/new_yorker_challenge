import os
import numpy as np

from pyspark import SparkConf, StorageLevel
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.ml.feature import RegexTokenizer, NGram, CountVectorizer, IDF, \
    StopWordsRemover
from pyspark.ml.pipeline import Pipeline
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import LinearSVC


def update_text_with_key_ngrams(df, n, seed=42,
                                outputCol="ngram_text",
                                pattern=r"(?!(?<='))\w+"):
    def build_text(words):
        # Wandle bag of words in sentences um und schaue in jedem der
        # sentences ob
        # eines der key_bigrams in ihm vorkommt
        # bspw. bag of words = ["hi", "i", "ralf"] und key_bigram = "i ralf" -->
        # sentence = ["hi i ralf"] und key_bigram kommt drin vor
        # Wenn bigram vorkommt, dann ersetze die zwei Wörter im Satz mit der
        # underscore version des bigrams ("i_ralf")
        sentence = ' '.join(words)
        for ngram in key_ngrams:
            if ngram in sentence:
                sentence = sentence.replace(ngram, ngram.replace(" ", "_"))
        return sentence

    outputs = {
        "tokenizer": "words",
        "ngram": "ngrams",
        "cv": "tf",
        "idf": "tf_idf",
        "build_text_udf": outputCol
    }

    # Build pipeline
    tokenizer = RegexTokenizer(inputCol="text",
                               outputCol=outputs["tokenizer"],
                               pattern=pattern,
                               gaps=False)
    ngram = NGram(n=n,
                  inputCol=tokenizer.getOutputCol(),
                  outputCol=outputs["ngram"])
    cv = CountVectorizer(inputCol=ngram.getOutputCol(),
                         outputCol=outputs["cv"])
    idf = IDF(inputCol=cv.getOutputCol(),
              outputCol=outputs["idf"])
    pipe = Pipeline(stages=[
        tokenizer,  # transform
        ngram,  # transform
        cv,  # fit_transform
        idf  # fit
    ])

    print("\t Computing tf_idf matrix for {}-grams...".format(n))
    pipe_model = pipe.fit(df)  # calls transform on tokenizer & ngram,
    # fit_transform on cv and fit on idf
    vocabulary = np.array(pipe_model.stages[2].vocabulary)
    print("\t\t vocabulary size: {}".format(len(vocabulary)))
    df = pipe_model.transform(df)

    # train test split
    train, _ = df.randomSplit([0.8, 0.2], seed=seed)
    train.persist(StorageLevel.MEMORY_AND_DISK)

    # fit linear SVM
    svc = LinearSVC(maxIter=100,
                    regParam=0.1,
                    featuresCol="tf_idf")
    print("\t Estimating key {}-grams with SVC...".format(n))
    svc_model = svc.fit(train)

    # Wähle die ngrams mit den schlechtesten/besten weights
    print("\t Update text with key {}-grams...".format(n))
    coeffs = svc_model.coefficients.toArray()
    key_ngrams = get_n_extremes_of_a_in_b(coeffs, vocabulary, 50)

    build_text_udf = F.udf(build_text)

    df = df.withColumn(outputs["build_text_udf"],
                       build_text_udf(
                           F.col(tokenizer.getOutputCol())))
    print()
    return df


def get_n_extremes_of_a_in_b(a, b, n):
    sort_idc = a.argsort()
    n_idc = np.concatenate((sort_idc[:n], sort_idc[-n:]))
    return b[n_idc]


## SETUP
conf = (SparkConf() \
        .set("spark.sql.shuffle.partitions", 100)
        .set("spark.default.parallelism", 100)
        .set("spark.driver.memory", "7g")
        # decrease to tackle out of memory erros because of heap
        .set("spark.memory.fraction", 0.6)
        .set("spark.master", "local[*]"))
spark = SparkSession.builder.config(conf=conf).getOrCreate()
data_dir = r"D:\Python\new_yorker_challenge\new_yorker_challenge\data" \
           r"\yelp_dataset"

## LOAD REVIEWS
print("Loading reviews...")
reviews = spark.read.json(
    os.path.join(data_dir, r"yelp_academic_dataset_review.json"))
reviews_mini = reviews.limit(10000)
reviews_mini.persist(StorageLevel.MEMORY_AND_DISK)

## DROP NOT NEEDED COLUMNS AND ADD LABELS
reviews_mini = reviews_mini \
    .withColumn("label",
                F.when(F.col("stars") >= 4, 1).otherwise(0)) \
    .withColumn("text", F.lower(F.col("text"))) \
    .drop("cool",
          "funny",
          "useful",
          "date",
          "stars") \
 \
## UPDATE TEXT WITH USEFUL BIGRAMS
print("Updating review text with ngrams...")
seed = 42
pattern = r"(?!(?<='))\w+"
outputCol = "ngram_text"
reviews_mini = update_text_with_key_ngrams(reviews_mini,
                                           n=2,
                                           seed=42,
                                           outputCol=outputCol,
                                           pattern=pattern)
print("\n")

## PREDICT LABEL BASED ON TF-IDF OF UPDATED TEXT
print("Computing TF-IDF matrix for updated text...")
tokenizer = RegexTokenizer(inputCol=outputCol,
                           outputCol="words_with_ngrams",
                           pattern=pattern,
                           gaps=False)
stop_words_remover = StopWordsRemover(inputCol=tokenizer.getOutputCol(),
                                      outputCol="filtered_words")
cv = CountVectorizer(inputCol=stop_words_remover.getOutputCol(),
                     outputCol="final_tf")
idf = IDF(inputCol=cv.getOutputCol(),
          outputCol="final_tf_idf")

pipe = Pipeline(stages=[
    tokenizer,
    stop_words_remover,
    cv,
    idf
])

reviews_mini = pipe.fit(reviews_mini).transform(reviews_mini)

## Train test split
train, test = reviews_mini.randomSplit([0.8, 0.2], seed=seed)
train.persist(StorageLevel.MEMORY_AND_DISK)

## Fit linear SVM
svc = LinearSVC(maxIter=100, featuresCol="final_tf_idf")
regs = [0.01, 0.1, 1]
paramGrid = ParamGridBuilder() \
    .addGrid(svc.regParam, regs) \
    .build()

tvs = TrainValidationSplit(estimator=svc,
                           estimatorParamMaps=paramGrid,
                           evaluator=BinaryClassificationEvaluator(),
                           trainRatio=0.8)
print("Search for best SVC to estimate labels based on TF-IDF matrix...")
print("\t Testing regularization parameters: {}".format(regs))
svc_model = tvs.fit(train)
print("\t Best parameter: {}".
      format(svc_model.bestModel._java_obj.getRegParam()))

print("Predict test set instance label and compute AUROC")
predictions = svc_model.transform(test).select("prediction", "rawPrediction",
                                               "label")
evaluator = BinaryClassificationEvaluator()
auroc = evaluator.evaluate(predictions)
print("AUROC: {}".format(auroc))
