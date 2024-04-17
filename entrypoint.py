import boto3
import pandas as pd
import os

import mlflow
import mlflow.sklearn
import sys

import uuid

from mlflow import MlflowClient
from minio import Minio

from io import StringIO
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def main() -> int:
    setupOsEnvironment()
    data_file = checkArguments()

    X_train, X_test, y_train, y_test = getTrainData(data_file)
    X_train_vec, X_test_vec = getVectorization(X_train, X_test)

    model = modelTrain(150000000, "lbfgs", X_train_vec, y_train)

    doPredict(model, X_test_vec, y_test)

    return 0

def checkArguments() -> str:
    if (len(sys.argv) != 2):
        print("Unacceptable args len: it must be equal to one")
        sys.exit()

    data_file = sys.argv[1]

    if (not os.path.exists(data_file)) or (not os.path.isfile(data_file)):
        print("Unacceptable data file: %s" % (data_file))
        sys.exit()

    return data_file    


def setupOsEnvironment(): 
    os.environ['MLFLOW_TRACKING_URI'] = 'http://127.0.0.1:5000'
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://127.0.0.1:9000'
    os.environ['AWS_ACCESS_KEY_ID'] = 'minio'
    os.environ['AWS_SECRET_ACCESS_KEY'] = 'minio123'
    os.environ['PYENV_ROOT'] = 'C:\\Users\\versh\\.pyenv'

    return 0

def getBotoS3Client():
    return boto3.client('s3',endpoint_url='http://127.0.0.1:9000', aws_access_key_id='minio', aws_secret_access_key='minio123')

def getTrainData(data_file = 'kinopoisk_train.csv'):
    # Настройка клиента boto3
    boto3.setup_default_session(aws_access_key_id='minio', aws_secret_access_key='minio123', region_name='us-west-1')
    
    # Инициализация клиента
    s3 = getBotoS3Client()
    
    # Считывание данных
    obj = s3.get_object(Bucket='datasets', Key=data_file)
    data = obj['Body'].read().decode('utf-8')
    df = pd.read_csv(StringIO(data))

    return train_test_split(df['review'], df['sentiment'], test_size=0.2)

def getVectorization(X_train, X_test):
    # Векторизация
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    return (X_train_vec, X_test_vec)

def modelTrain(maxIter = 150000000, modelSolver = "lbfgs", xTrainVec = [], yTrain = []): 
    # Обучение модели
    clf = LogisticRegression(max_iter=maxIter, solver=modelSolver)
    clf.fit(xTrainVec, yTrain)

    return clf

def doPredict(clf, X_test_vec, y_test):
    y_pred = clf.predict(X_test_vec)

    mlflow.autolog()

    mlflow_client = MlflowClient("http://127.0.0.1:5000")

    experiment_name = "kinopoisk"
    experiment_id = mlflow_client.create_experiment(experiment_name) if mlflow.get_experiment_by_name(experiment_name) is None else mlflow.get_experiment_by_name(experiment_name).experiment_id
    run_id = mlflow_client.create_run(str(experiment_id))

    with mlflow.start_run(experiment_id=str(experiment_id), run_id=str(run_id.info.run_id)) as run:
        model_name = "lr-model-" + str(uuid.uuid4())

        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    
        mlflow.sklearn.log_model(clf, "model", registered_model_name=model_name)

if __name__ == '__main__':
    sys.exit(main()) 
