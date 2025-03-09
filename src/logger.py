import os
import logging
from logging.handlers import RotatingFileHandler

def setup_logging(log_filename='output.log', clear_logs=True):
    log_path = os.path.join(os.getcwd(), log_filename)

    if clear_logs:
        open(log_path, 'w').close()

    logger = logging.getLogger(log_filename)
    logger.setLevel(logging.DEBUG)


    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = RotatingFileHandler(
        log_path, maxBytes=10**6, backupCount=3
    )

    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


logger = setup_logging(log_filename='output.log')


def write_tensors_nodes(graph,org=True):
    
    tensors=graph.tensors()
    nodes=graph.nodes
    
    if org:
        
        logger.info("Write original models tensors and nodes\n")
        
        with open('./logs/org-tensor.txt',mode='w') as file:
            file.write(str(tensors))
        with open('./logs/org-nodes.txt',mode='w') as file:
            file.write(str(nodes))
            
    else:
        
        logger.info("Write isolated models tensors and nodes\n")
        
        
        with open('./logs/iso-tensor.txt',mode='w') as file:
            file.write(str(tensors))
    
        with open('./logs/iso-nodes.txt',mode='w') as file:
            file.write(str(nodes))
