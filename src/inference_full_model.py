import numpy as np
import onnxruntime as ort
import onnx
import onnx_graphsurgeon as og
from contextlib import redirect_stdout

from pathlib import Path
import torch

import argparse

  


def parse_args():
    parser = argparse.ArgumentParser(description='ONNX Model modification')
    parser.add_argument('-o','--org', type=str, help='Path to the original model', default='D:/Learn DL/Emil-Net/artifacts/simplified_emil_net.onnx')
    parser.add_argument('-m','--mod',type=str, help='Path to the modified model', default='D:/Learn DL/Emil-Net/artifacts/orgmod_model.onnx')
    args = parser.parse_args()
    return args
    

def main():
    args = parse_args()
    
    fullorg_model_path=Path(args.org)
    mod_model_path=Path(args.mod)
    
    print("<--------Evaluating Full Original Model-------------->\n\n")

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

    print("Original Model Inference Completed\n")

    print(f"Original Model Outputs: {output1}\n\n")


    print("<--------Evaluating Full Modified Model-------------->\n\n")

    mod_onnx_model = onnx.load(mod_model_path)

    onnx.checker.check_model(mod_onnx_model)


    session = ort.InferenceSession(mod_onnx_model.SerializeToString())


    output2 = session.run(["trajectories","scores"], inputs)

    print("MOdified  Model Inference Completed\n")

    print(f"Modified  Model Outputs: {output2}\n\n")
    
    
    print("\n<------------- Comparison of MSE -------->\n\n")

    

    original_output = torch.tensor(output1[0], dtype=torch.float32)
    modified_output = torch.tensor(output2[0],dtype=torch.float32)

 
    mse = torch.nn.functional.mse_loss(original_output, modified_output)
    print(f"MSE between Trajectories: {mse.item()}")


    original_output = torch.tensor(output1[1], dtype=torch.float32)
    modified_output = torch.tensor(output2[1],dtype=torch.float32)

    mse = torch.nn.functional.mse_loss(original_output, modified_output)
    print(f"MSE between Scores: {mse.item()}")

if __name__ == "__main__":
    with open('./full_output.log', 'w') as f:
        with redirect_stdout(f):
            main()
     




