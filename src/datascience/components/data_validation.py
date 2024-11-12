import os
import pandas as pd
from src.datascience.utils.common import logger
from src.datascience.entity.config_entity import DataValidationConfig

class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_all_columns(self) -> bool:
        try:
            validation_status = None

            data = pd.read_csv(self.config.unzip_data_dir)
            all_cols = list(data.columns)

            all_schema = self.config.all_schema
            schema_keys = all_schema.keys()
            
            for col in all_cols:
                if col not in schema_keys or str(data[col].dtype) != str(all_schema.get(col)):
                    print(col, col not in schema_keys)
                    validation_status = False
                    with open(self.config.STATUS_FILE, "w") as f:
                        f.write(f"Validatoin status: {validation_status}")
                    break
                else:
                    validation_status = True
                    with open(self.config.STATUS_FILE, "w") as f:
                        f.write(f"Validatoin status: {validation_status}")

        except Exception as e:
            logger.error(f"Error occurred during data validation: {str(e)}")
            raise e