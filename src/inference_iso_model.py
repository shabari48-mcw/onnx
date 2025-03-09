import numpy as np
import onnxruntime as ort
import onnx
import onnx_graphsurgeon as og
from logger import logger
from pathlib import Path
from rich import print

inputs="reference_line_valid_mask"
  
# iso_model_path=Path("D:/Learn DL/Emil-Net/src/isolated_emil_net.onnx")

mod_model_path=Path("D:/Learn DL/Emil-Net/artifacts/ak-isolated.onnx")

input_value = np.random.rand(1,1,99).astype(np.float32)


logger.info("\n<-------------Isolated Subgraph RunTime session-------->\n\n")

iso_onnx_model = onnx.load(mod_model_path)


onnx.checker.check_model(iso_onnx_model)


session = ort.InferenceSession(iso_onnx_model.SerializeToString())


logger.info(f"Print Input {input_value}\n")

output1=session.run(None,{inputs:input_value})

print(f"Output: {output1}\n")




# logger.info("\n<-------------Modified Subgraph RunTime session-------->\n\n")


# mod_onnx_model = onnx.load(mod_model_path)

# onnx.checker.check_model(mod_onnx_model)

# session = ort.InferenceSession(mod_onnx_model.SerializeToString())

# logger.info(f"Print Input {input_value}\n")

# output2=session.run(None,{inputs:input_value})

# logger.info(f"Output: {output2}\n")



# logger.info("\n<------------- Comparison of MSE -------->\n\n")

# import torch

# original_output = torch.tensor(output1[0], dtype=torch.float32)
# modified_output = torch.tensor(output2[0],dtype=torch.float32)

# # Compute MSE
# mse = torch.nn.functional.mse_loss(original_output, modified_output)
# logger.info(f"MSE between Trajectories: {mse.item()}")



# logger.info("Comparison Completed.")




        
