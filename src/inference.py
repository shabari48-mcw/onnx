import numpy as np
import onnxruntime as ort
import onnx
import onnx_graphsurgeon as og
from logger import logger
from pathlib import Path

def infer(config):
    org_model_path = Path(config['import-path'])
    org_onnx_model = onnx.load(org_model_path)

    onnx.checker.check_model(org_onnx_model)


    session = ort.InferenceSession(org_onnx_model.SerializeToString())

    graph = og.import_onnx(org_onnx_model)

    inputs = {}
    for input_tensor in graph.inputs:
        shape =  input_tensor.shape
        dtype = np.float32 
        inputs[input_tensor.name] = np.random.rand(*shape).astype(dtype)

   
    outputs = session.run(["trajectories","scores"], inputs)
    
    logger.info("Original Model Inference Completed\n")
    
    logger.info(f"Original Model Outputs: {outputs}\n\n")

    if config['isolate']:
        isoconfig = config['isolate-config']
        
        iso_model_path = Path(config['export-path'])
        iso_onnx_model = onnx.load(iso_model_path)

        onnx.checker.check_model(iso_onnx_model)

  
        session = ort.InferenceSession(iso_onnx_model.SerializeToString())

        logger.info("Isolated Subgraph RunTime session\n")

        input_value = np.random.rand(*isoconfig['input-shape']).astype(np.float32)

        output = session.run([isoconfig['outputs']], {isoconfig['inputs']: input_value})

        logger.info(f"Isolated Model Output{output}")
        
        
