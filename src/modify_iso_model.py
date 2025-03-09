import onnx
import onnx_graphsurgeon as gs
import numpy as np
from rich import print

# Alternative path format:
model = onnx.load("D:/Learn DL/Emil-Net/src/isolated_emil_net.onnx")

graph = gs.import_onnx(model)

#All Nodes
print("Nodes in the graph:")
for node in graph.nodes:
    print(f"{node.name} ({node.op})")

# Finding the Where node that we want to replace
where_node = None
for node in graph.nodes:
    if node.op == "Where" and "planning_decoder" in node.name and "r2r_attn" in node.name:
        print(f"Found Where node: {node.name}\n")
        where_node = node
        break

if where_node is None:
    raise ValueError("Where node not found in the graph. Check node names in the graph.")

#Inputs and Output Tensors of Where Node
print(f"\nWhere node inputs: {[inp.name for inp in where_node.inputs]}")
print(f"\nWhere node outputs: {[out.name for out in where_node.outputs]}")

# Get the inputs to the Where operator
condition = where_node.inputs[0]  # boolean condition input
neg_infinity = where_node.inputs[1]  # constant -∞
zeros = where_node.inputs[2]  # constant 0



# Get the output
original_output = where_node.outputs[0]

print(f"Outputs :{original_output}")


# Create new nodes
# 1. Cast the boolean condition to float32
cast_output = gs.Variable(name=f"{condition.name}_casted_to_float32", dtype=np.float32)

cast_node = gs.Node(
    op="Cast", 
    name=f"cast_bool_to_float_{condition.name}",
    inputs=[condition], 
    outputs=[cast_output],
    attrs={"to": 1}  # 1 is for float32
)

# 2. Create a constant for negative infinity
neg_inf_const = gs.Constant(
    name="neg_infinity_const",
    values=np.array(float('-inf'), dtype=np.float32)
)

# 3. Multiply the casted condition with negative infinity
mul_output = gs.Variable(name=f"{condition.name}_mul_neg_inf", dtype=np.float32)
mul_node = gs.Node(
    op="Mul",
    name=f"mul_cond_neg_inf_{condition.name}",
    inputs=[cast_output, neg_inf_const], 
    outputs=[mul_output]
)

# 4. Create IsNaN node to identify NaN values
isnan_output = gs.Variable(name=f"{mul_output.name}_isnan", dtype=np.bool_)
isnan_node = gs.Node(
    op="IsNaN",
    name=f"isnan_{mul_output.name}",
    inputs=[mul_output],
    outputs=[isnan_output]
)

# 5. Create a zero constant
zero_const = gs.Constant(
    name="zero_const",
    values=np.array(0.0, dtype=np.float32)
)

# 6. Create a Where node to replace NaN values with 0
new_where_output = gs.Variable(name=original_output.name, dtype=np.float32)
new_where_node = gs.Node(
    op="Where",
    name=f"where_replace_nan_{condition.name}",
    inputs=[isnan_output, zero_const, mul_output],
    outputs=[new_where_output]
)

# Find all nodes that use the original Where node's output
for node in graph.nodes:
    for i, inp in enumerate(node.inputs):
        if inp == original_output and node != where_node:
            print(f"Updating node {node.name} to use new output")
            node.inputs[i] = new_where_output

# Disconnect the original Where node from the graph
for out in where_node.outputs:
    out.inputs.clear()

for inp in where_node.inputs:
    if where_node in inp.outputs:
        inp.outputs.remove(where_node)

# Remove the original Where node from the graph
graph.nodes.remove(where_node)

# Add our new nodes to the graph
graph.nodes.extend([cast_node, mul_node, isnan_node, new_where_node])

output_model = gs.export_onnx(graph)

# Save the modified model
onnx.save(output_model, "D:/Learn DL/Emil-Net/src/modified_model.onnx")
onnx.checker.check_model(output_model)
print("Model modified ")