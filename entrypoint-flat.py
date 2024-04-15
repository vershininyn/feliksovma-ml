
import boto3
import pandas as pd
import os
import mlflow
import mlflow.sklearn

from io import StringIO
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

os.environ['MLFLOW_TRACKING_URI'] = 'http://127.0.0.1:5000'
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://127.0.0.1:9000'
os.environ['AWS_ACCESS_KEY_ID'] = 'minio'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'minio123'

# Настройка клиента boto3
boto3.setup_default_session(
    aws_access_key_id='minio',
    aws_secret_access_key='minio123',
    region_name='us-west-1'  # или другой регион, если это применимо
)


# Инициализация клиента
s3 = boto3.client('s3',
                  endpoint_url='http://127.0.0.1:9000',
                  aws_access_key_id='minio',
                  aws_secret_access_key='minio123')

# Считывание данных
obj = s3.get_object(Bucket='datasets', Key='kinopoisk_train.csv')
data = obj['Body'].read().decode('utf-8')
df = pd.read_csv(StringIO(data))

X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.2)

# Векторизация
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Обучение модели
# clf = LogisticRegression()
clf = LogisticRegression(max_iter=150000000, solver="lbfgs")
clf.fit(X_train_vec, y_train)

# Предсказание
y_pred = clf.predict(X_test_vec)

params = {"solver": "lbfgs", "max_iter": 150000000, "model_type": "LogisticRegression"}

# Логирование в MLflow
with mlflow.start_run() as run:
    # Логирование параметров и метрик

    # mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_params(params)
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    
    # Логирование модели
    mlflow.sklearn.log_model(clf, "model", registered_model_name="MyFirstModel")


# accuracy_score(y_test, y_pred)
