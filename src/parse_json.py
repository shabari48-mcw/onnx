import json
from logger import logger

def parse_json(json_path:str)->dict:
   
    logger.info("Parse Configuration JSON file\n")
    
    with open(json_path, "r") as file:
        data = json.load(file)
        return data