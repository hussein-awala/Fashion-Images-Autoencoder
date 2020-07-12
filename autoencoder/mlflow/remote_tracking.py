import os
import sys
import yaml
import mlflow
import mlflow.tensorflow


class MLflow:
    def __init__(self, config_file_path):
        with open("config.yml", "r") as config_file:
            configs = yaml.safe_load(config_file)["MLFLOW_CONFIG"]
        if "AUTHENTICATION" in configs:
            os.environ["MLFLOW_TRACKING_USERNAME"] = configs["AUTHENTICATION"]["MLFLOW_TRACKING_USERNAME"]
            os.environ["MLFLOW_TRACKING_PASSWORD"] = configs["AUTHENTICATION"]["MLFLOW_TRACKING_PASSWORD"]
        if "S3" in configs:
            os.environ["MLFLOW_S3_ENDPOINT_URL"] = configs["S3"]["MLFLOW_S3_ENDPOINT_URL"]
            os.environ["AWS_ACCESS_KEY_ID"] = configs["S3"]["AWS_ACCESS_KEY_ID"]
            os.environ["AWS_SECRET_ACCESS_KEY"] = configs["S3"]["AWS_SECRET_ACCESS_KEY"]
        if "EXPERIMENT_NAME" in configs:
            mlflow.set_experiment(configs["EXPERIMENT_NAME"])
        mlflow.set_tracking_uri(configs["TRACKING_URI"])
        print("Running {} with tracking URI {}".format(sys.argv[0], mlflow.get_tracking_uri()))

    def activate_keras(self):
        mlflow.tensorflow.autolog()
