FROM bde2020/spark-master:2.4.5-hadoop2.7

COPY recommender /app/recommender
COPY data /app/data
COPY spark-defaults.conf /spark/conf

RUN pip install pyspark==2.4.5

RUN apk update
RUN apk add build-base
RUN apk add python-dev
RUN pip install numpy

