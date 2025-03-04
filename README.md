#  ONNX Task 

## Source of the model

	Model picked up from  '/media/bmw/simplified_emil_net.onnx'


## Description of the Task

- Task 1 : To isolate subgraph 
  - start node :  reference_line_valid_mask
  - end node :planner_decoder/decoder_blocks.0/cast_1_output_0 

- Task 2 : ONNX runtime script for original as well as isolate subgrpah

## Execution command

Command to run:

    python src/main.py --path ./config/config.json
