import logging
import os
from datetime import datetime
Log_file = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}"
log_path = os.path.join(os.getcwd(),'logs',Log_file)
# it will create cureeent directory then create logs file  and under log file it create LOG_file
os.makedirs(log_path,exist_ok=True)
Log_file_path = os.path.join(log_path,Log_file)

logging.basicConfig(
    filename=Log_file_path,
    format = "[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
'''
%(asctime)s] = it will give current time
%(lineno)d = line number
%(name)s = Name of the logger used to log the call.like root
- %(levelname)s = level name like info
- %(message)s" = message 
'''