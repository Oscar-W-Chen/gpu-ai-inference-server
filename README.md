# gpu-ai-inference-server
AI Inference Server that takes trained AI models, load them into memory, and executes inference requests efficiently on NVIDIA GPUs. This project utilizes C++, CUDA programming, and Golang.

# TODO LIST
- [x] Implement and test basic server by launch it through docker on local machine(CPU)
- [x] Implement cuda_util and related CMakeLists 
- [x] Implement test file for cuda_util and test cuda_util on Google Colab
- [x] Implement the model and inference_manager files
- [x] Implement the C to Go binding logic
- [ ] Understand the new files
- [ ] Update main.go to expose the underlying C++ functionalities to Go server
- [ ] Complete the notebook such that the Google Colab runs the server directly on Go without containers
- [ ] Add other Go files to set up the full functioning server
- [ ] Add to documentation under docs/


# Detailed Design
For top level design diagram, see [here](./docs/design_diagram.md)
For documentations about key components, dataflow, and component interactions, see [here](./docs/design.md).