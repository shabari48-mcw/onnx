
import argparse
import numpy as np
import onnx
import os
import onnx_graphsurgeon as og
from pathlib import Path
from parse_json import parse_json
from logger import logger,write_tensors_nodes
from inference import infer


def parse_args():
    parser=argparse.ArgumentParser(description="Inference using ONNX runtime ")
    parser.add_argument("-p","--path",type=str,help='Specify the configuration JSON Path')
    args = parser.parse_args()
    return args

def main():
    
    args = parse_args()
    json_path = os.path.join(args.path)
    config= parse_json(json_path)  
    
    logger.info(f"Configurations: {config} \n\n")
      
      
    logger.info(f"<------Loading ONNX Model {config['model-name']} from {config['import-path']}------------>\n")
    
    model_path=Path(config['import-path'])
    onnx_model=onnx.load(model_path)
    
    onnx.checker.check_model(onnx_model)
    
    graph=og.import_onnx(onnx_model)
    
    write_tensors_nodes(graph,True)
    
    tensors=graph.tensors()    
    
    if config['isolate']:
        isoconfig=config['isolate-config']
        
        graph.inputs=[tensors[isoconfig['inputs']].to_variable(dtype=np.float32,shape=isoconfig['input-shape'])]
        graph.outputs=[tensors[isoconfig['outputs']].to_variable(dtype=np.bool,shape=isoconfig['output-shape'])]
        graph.cleanup()
        
        
        logger.info("Model Subgraph Isolation Done\n")
        
        write_tensors_nodes(graph,False)
        
        onnx.save(og.export_onnx(graph),config['export-path'])
        
        logger.info(f"Saved the isolated model to {config['export-path']}\n")
        
    else :
        logger.info("Isolation not performed \n")
        
    
    logger.info("Model Isolation Execution Over")
    
    
    if config['inference']:
        infer(config)
        
    
if __name__=='__main__':
    main()
    
# isolate subgraph from start: node reference_line_valid_mask 
# to node: planner_decoder/decoder_blocks.0/cast_1_output_0 
    
    