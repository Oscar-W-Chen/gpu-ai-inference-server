# gpu-ai-inference-server
AI Inference Server that takes trained AI models, load them into memory, and executes inference requests efficiently on NVIDIA GPUs. This project utilizes C++, CUDA programming, and Golang.

# TODO LIST
- [x] Implement and test basic server by launch it through docker on local machine(CPU)
- [x] Implement cuda_util and related CMakeLists 
- [x] Implement test file for cuda_util and test cuda_util on Google Colab
- [x] Implement the model and inference_manager files
- [x] Implement the C to Go binding logic
- [x] Understand the new files
- [x] Update main.go to expose the underlying C++ functionalities to Go server
- [x] Complete the notebook such that the Google Colab runs the server directly on Go without containers
- [x] Add to documentation under docs/
- [x] Fully implement ModelRepository and integrate it with the code stack
- [x] Fully implement the model specific loading, inferencing, and unloading logic for ONNX
- [x] Put API.md on homepage
- [ ] Restructure main.go to use singleton inference manager throughout the server session 
- [ ] Remove unnecessary DEBUG messages
- [x] Put complete model files under /model directory to test loading functions
- [ ] Add other Go files to set up the full functioning server
- [ ] Run full end-to-end integration test on AI inferencing on top of GPUs


# Detailed Design
For top level design diagram, see [here](./docs/architecture-diagram.svg)
For documentations about key components, dataflow, and component interactions, see [here](./docs/design.md).

# How to run the server in Google Colab
Google Colab is the required environment to run this server since it has NVIDIA GPUs available
To run the server:
1. Open Google Colab environment. Make sure you are connected to a T4 GPU runtime
2. Open [run_server.ipynb](./docs/run_server.ipynb)
3. Substitude github username and email, github auth token, and ngrok auth token
4. Build the server
   ```
   !./scripts/build.server.py
   ```
5. Run the server
   ```
   !./scripts/run_server.sh
   ```
6. OPTIONAL: on your local terminal (Unable to do this on Google Colab), you can execute the test_client.py script run automated tests against the opened server
   ```
   python client/test_client.py --url [https://ngrok-custom-url.app]
   # For example
   python client/test_client.py --url https://9a68-34-19-48-151.ngrok-free.app/
   ```
