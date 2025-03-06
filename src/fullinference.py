import numpy as np
import onnxruntime as ort
import onnx
import onnx_graphsurgeon as og
from logger import logger
from pathlib import Path

  
fullorg_model_path=Path("D:/Learn DL/Emil-Net/src/simplified_emil_net.onnx")
mod_model_path=Path("D:/Learn DL/Emil-Net/src/orgmod_model.onnx")


logger.info("<--------Evaluating Full Original Model-------------->\n\n")

org_onnx_model = onnx.load(fullorg_model_path)

onnx.checker.check_model(org_onnx_model)


session = ort.InferenceSession(org_onnx_model.SerializeToString())

graph = og.import_onnx(org_onnx_model)

inputs = {}
for input_tensor in graph.inputs:
    shape =  input_tensor.shape
    dtype = np.float32 
    inputs[input_tensor.name] = np.random.rand(*shape).astype(dtype)


output1 = session.run(["trajectories","scores"], inputs)

logger.info("Original Model Inference Completed\n")

logger.info(f"Original Model Outputs: {output1}\n\n")




logger.info("<--------Evaluating Full Modified Model-------------->\n\n")

mod_onnx_model = onnx.load(mod_model_path)

onnx.checker.check_model(mod_onnx_model)


session = ort.InferenceSession(mod_onnx_model.SerializeToString())


output2 = session.run(["trajectories","scores"], inputs)

logger.info("MOdified  Model Inference Completed\n")

logger.info(f"Modified  Model Outputs: {output2}\n\n")


logger.info("\n<------------- Comparison of MSE -------->\n\n")

import torch

original_output = torch.tensor(output1[0], dtype=torch.float32)
modified_output = torch.tensor(output2[0],dtype=torch.float32)

# Compute MSE
mse = torch.nn.functional.mse_loss(original_output, modified_output)
logger.info(f"MSE between Trajectories: {mse.item()}")


original_output = torch.tensor(output1[1], dtype=torch.float32)
modified_output = torch.tensor(output2[1],dtype=torch.float32)

# Compute MSE
mse = torch.nn.functional.mse_loss(original_output, modified_output)
logger.info(f"MSE between Scores: {mse.item()}")



logger.info("Comparison Completed.")





