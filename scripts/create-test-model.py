#!/usr/bin/env python3
"""
Script to create a simple ONNX model for testing the inference server.
This creates a simple feedforward neural network with one hidden layer.
"""

import os
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, TensorProto, numpy_helper

def create_test_model(output_dir):
    """Create a simple ONNX model for testing."""
    # Create model version directory
    model_dir = os.path.join(output_dir, "test_model", "1")
    os.makedirs(model_dir, exist_ok=True)
    
    # Create a simple model (linear layer -> ReLU -> linear layer)
    input_shape = [1, 3]  # Batch size 1, 3 features
    hidden_size = 5
    output_size = 2
    
    # Initialize weights and biases with random values
    np.random.seed(42)  # For reproducibility
    weight1 = np.random.randn(3, hidden_size).astype(np.float32)
    bias1 = np.random.randn(hidden_size).astype(np.float32)
    weight2 = np.random.randn(hidden_size, output_size).astype(np.float32)
    bias2 = np.random.randn(output_size).astype(np.float32)
    
    # Create ONNX weights and biases as initializers
    weight1_tensor = numpy_helper.from_array(weight1, name="weight1")
    bias1_tensor = numpy_helper.from_array(bias1, name="bias1")
    weight2_tensor = numpy_helper.from_array(weight2, name="weight2")
    bias2_tensor = numpy_helper.from_array(bias2, name="bias2")
    
    # Create input and output tensors
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)
    hidden_tensor = helper.make_tensor_value_info("hidden", TensorProto.FLOAT, [1, hidden_size])
    relu_tensor = helper.make_tensor_value_info("relu", TensorProto.FLOAT, [1, hidden_size])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, output_size])
    
    # Create the nodes in the graph
    # First layer: input * weight1 + bias1 = hidden
    node1 = helper.make_node(
        "MatMul",
        ["input", "weight1"],
        ["matmul1"],
        name="matmul1"
    )
    
    node2 = helper.make_node(
        "Add",
        ["matmul1", "bias1"],
        ["hidden"],
        name="add1"
    )
    
    # ReLU activation
    node3 = helper.make_node(
        "Relu",
        ["hidden"],
        ["relu"],
        name="relu"
    )
    
    # Second layer: relu * weight2 + bias2 = output
    node4 = helper.make_node(
        "MatMul",
        ["relu", "weight2"],
        ["matmul2"],
        name="matmul2"
    )
    
    node5 = helper.make_node(
        "Add",
        ["matmul2", "bias2"],
        ["output"],
        name="add2"
    )
    
    # Create the graph
    graph = helper.make_graph(
        [node1, node2, node3, node4, node5],  # Nodes
        "test_model",  # Graph name
        [input_tensor],  # Inputs
        [output_tensor],  # Outputs
        [weight1_tensor, bias1_tensor, weight2_tensor, bias2_tensor]  # Initializers
    )
    
    # Create the model
    model = helper.make_model(graph, producer_name="onnx-example")
    model.opset_import[0].version = 12
    
    # Save the model
    model_path = os.path.join(model_dir, "model.onnx")
    onnx.save(model, model_path)
    print(f"Saved ONNX model to {model_path}")
    
    # Create config.json
    config = {
        "name": "test_model",
        "version": "1",
        "inputs": [
            {
                "name": "input",
                "shape": [1, 3],
                "data_type": "FLOAT32"
            }
        ],
        "outputs": [
            {
                "name": "output",
                "shape": [1, 2],
                "data_type": "FLOAT32"
            }
        ]
    }
    
    # Write config.json
    import json
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config.json to {config_path}")
    
    # Test the model with random input
    print("Testing the model with ONNX Runtime...")
    test_input = np.random.randn(1, 3).astype(np.float32)
    
    # Create a session
    session = ort.InferenceSession(model_path)
    
    # Run inference
    outputs = session.run(["output"], {"input": test_input})
    
    print(f"Input: {test_input}")
    print(f"Output: {outputs[0]}")
    print("Model test successful!")
    
    return model_dir

if __name__ == "__main__":
    # Create model in the current directory's models folder
    output_dir = os.path.join(os.getcwd(), "models")
    os.makedirs(output_dir, exist_ok=True)
    model_dir = create_test_model(output_dir)
    print(f"Test model created in {model_dir}")