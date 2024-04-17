import boto3
import pandas as pd
import os

import mlflow
import mlflow.sklearn
import sys

import uuid
import http.client
import json

from mlflow import MlflowClient

from minio import Minio
from io import StringIO
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

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

# Пример использования
def doPredict(clf, X_test_vec, y_test):
    # Предсказание
    y_pred = clf.predict(X_test_vec)

    params = {"solver": "lbfgs", "max_iter": 150000000, "model_type": "LogisticRegression"}

    mlflow.autolog()
    mlflow.set_tracking_uri("file:\\\\\\" + os.getcwd() + "\\mlruns") 

    # s3 = getBotoS3Client()

    with mlflow.start_run() as run:
        # Логирование параметров и метрик
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))

        model_name = "model-" + str(uuid.uuid4())
        # Логирование модели
        mlflow.sklearn.log_model(clf, artifact_path="sklearn-model", registered_model_name=model_name)

        run_id = run.info.run_id
        experiment_id = run.info.experiment_id

        root_bucket_directory = str(0) + "/" + str(run_id) + "/"

        # Подключение к MinIO серверу
        minio_client = Minio("127.0.0.1:9000", access_key="minio", secret_key="minio123", secure=False)

        uploadModelToMinIOServer(minio_client, run_id, root_bucket_directory, ["artifacts", "model" ,"metadata"])
        uploadModelToMinIOServer(minio_client, run_id, root_bucket_directory, ["artifacts", "model"])
        uploadModelToMinIOServer(minio_client, run_id, root_bucket_directory, ["metrics"])
        uploadModelToMinIOServer(minio_client, run_id, root_bucket_directory, ["params"])
        uploadModelToMinIOServer(minio_client, run_id, root_bucket_directory, ["tags"])

        mlflow_client = MlflowClient("http://127.0.0.1:5000")

        # Create source model version
        src_name = "RandomForestRegression-staging"
        client.create_registered_model(src_name)
        src_uri = f"runs:/{run.info.run_id}/sklearn-model"
        mv_src = client.create_model_version(src_name, src_uri, run.info.run_id)
        print_model_version_info(mv_src)
        print("--")

        # Copy the source model version into a new registered model
        dst_name = "RandomForestRegression-production"
        src_model_uri = f"models:/{mv_src.name}/{mv_src.version}"
        mv_copy = client.copy_model_version(src_model_uri, dst_name)
        print_model_version_info(mv_copy)


def uploadModelToMinIOServer(minio_client, run_id, root_bucket_directory, sub_folder_string_array):
    local_model_path = os.path.join(*sub_folder_string_array)
    metadata_dir = constructPathToModelFile(run_id, local_model_path)

    metadata_file_minio_path = root_bucket_directory + "/".join(sub_folder_string_array + [''])
    
    for file in list_files(metadata_dir):
        minio_client.fput_object("mlflow", metadata_file_minio_path + os.path.basename(file), file)

def constructPathToModelFile(run_id, model_directory):
    return os.path.join(os.getcwd(), "mlruns", str(0), run_id, model_directory)

def list_files(directory):
    files_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            files_list.append(file_path)
    return files_list

    # Логирование в MLflow
    # with mlflow.start_run() as run:
    #     mlflow.log_params(params)
    #     mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    
    #     model_name = "Model-" + str(uuid.uuid4())
    #     model_path = os.path.join(os.getcwd(), "mlruns", "models", model_name)
        
    #     mlflow.sklearn.save_model(clf, model_path)

    #     run_id = run.info.run_id
    #     experiment_id = run.info.experiment_id

    #     print("experiment_id: " + experiment_id)
    #     print("run_id: " + run_id)
    #     print("model_name: " + model_name)

        # s3_model_key = str(0) + "/" + str(experiment_id) + "/" + "artifacts" + "/" + model_name

        # for file in list_files(model_path):
        #     s3.upload_file(file, "mlflow", s3_model_key + "/" + os.path.basename(file))

        # mlflow.sklearn.log_model(clf, artifact_path="local_and_mlflow", registered_model_name=model_name)
        # mlflow.log_artifacts(model_path, artifact_path="data")
    
        # Логирование модели в локальную директорию и в MLflow
        # mlflow.sklearn.log_model(clf, "model", artifact_path="local_and_mlflow", registered_model_name=model_name)

        # Логирование модели в S3 и в MLflow
        # mlflow.sklearn.log_model(clf, artifact_path="s3://mlflow/" + s3_model_key, registered_model_name=model_name)

        # mlflow.sklearn.log_model(clf, "model", registered_model_name=model_name)

        # mlflow.sklearn.log_model(clf, "model", artifact_path="local_and_mlflow")

        # Укажите путь к вашей локальной модели и бакету S3, в который вы хотите ее загрузить
        # local_model_path = os.path.join(os.getcwd(), "mlruns", "models", model_name, "meta.yaml")
        # s3 = getBotoS3Client()

        # print("model_path: " + model_path) 

        # for file in list_files(model_path):
        #     print("++++ current_file: " + file)
        #     s3.upload_file(file, "mlflow", str(experiment_id) + "/" + model_name + "/" + os.path.basename(file))

        # print("++++++++++++++")
        # httpClient = http.client.HTTPConnection("127.0.0.1", port=5000)
        # headers = {"Content-type": "application/json"}

        # data = {"experiment_id": str(experiment_id),
        #         "artifact_location": "s3://mlflow/" + str(experiment_id),
        #         "parameters": {
        #             "param1": "value1",
        #             "param2": "value2"
        #             },
        #         "tags": {
        #             "tag1": "value1",
        #             "tag2": "value2"
        #             }
        #         }
        
        # response = httpClient.getresponse()

        # print("STATUS = " + str(response.status), "REASON = " + str(response.reason))
        # print("++++++++++++++")


if __name__ == '__main__':
    sys.exit(main()) 
