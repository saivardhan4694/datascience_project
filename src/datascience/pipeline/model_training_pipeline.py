from src.datascience.config.configuration import ConfigurationManager
from src.datascience.components.model_trainer import ModelTrainer
from src.datascience.utils.common import logger

STAGE_NAME = "Model Training Stage"

class ModelTrainingPipeline():
    def __init__(self) -> None:
        pass

    def initiate_model_training(self):
        config = ConfigurationManager()
        model_training_config = config.get_model_trainer_config()
        model_trainer = ModelTrainer(model_training_config)
        model_trainer.train()

if __name__ == "__main__":
    try:
        logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<<<")
        obj = ModelTrainingPipeline()
        obj.initiate_model_training()
        logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e
