import os
from pathlib import Path
import logging

logging.basicConfig(level = logging.INFO)
project_name = 'mlproject'

list_of_file = [
    f"src/__init__.py",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/data_ingestion.py",
    f"src/{project_name}/components/data_transformation.py",
    f"src/{project_name}/components/model_trainer.py",
    f"src/{project_name}/components/model_monitering.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/pipeline/training_pipeline.py",
    f"src/{project_name}/pipeline/training_pipeline.py",
    f"src/{project_name}/exception.py",
    f"src/{project_name}/logger.py",
    f"src/{project_name}/utils.py",
    "app.py",
    "Docker.py",
    "main.py"
]

for filepath in list_of_file:
    filepath = Path(filepath)
    filedir,file_name = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir,exist_ok=True)
        logging.info(f"creating the filedir:{filedir} of the name of {file_name}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath,'w') as f:
            pass
            logging.info(f"creating the empty file: {filepath}")

    else:
        logging.info(f"{file_name} is already exists")