import os
from src.datascience import logger
from src.datascience.entity.config_entity import DataTransformationConfig
import pandas as pd
from sklearn.model_selection import train_test_split

class DataTransformation():
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def train_test_splitting(self):
        data = pd.read_csv(self.config.data_file_path)

        train, test = train_test_split(data)

        train.to_csv(os.path.join(self.config.root_dir, "train.csv"), index = False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"), index = False)

        logger.info("Splitting data into training and test sets done.")
        logger.info(f"Shape of train set: {train.shape}")
        logger.info(f"Shape of test set: {test.shape}")