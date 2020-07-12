import yaml
import os


class KaggleManager:
    def __init__(self, config_file):
        self.configs = None
        with open("config.yml", "r") as config_file:
            self.configs = yaml.safe_load(config_file)["KAGGLE_CONFIG"]
        os.environ["KAGGLE_USERNAME"] = self.configs["AUTHENTICATION"]["KAGGLE_USERNAME"]
        os.environ["KAGGLE_KEY"] = self.configs["AUTHENTICATION"]["KAGGLE_KEY"]

    def download_dataset(self, dataset=None):
        if dataset:
            kaggle_dataset = dataset
        elif "DATASET" in self.configs:
            kaggle_dataset = self.configs["DATASET"]
        else:
            raise Exception("You have to select the dataset you want to download")
        import kaggle
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(kaggle_dataset, path='data', unzip=True)
