# DEVNOTE

## Conceptual Guide :: Part 1

### Server

#### Change Directory

```bash
cd Conceptual_Guide/Part_1-model_deployment
```

#### Convert "frozen_east_text_detection.pb" to "detection.onnx"

```bash
docker run -it --gpus all -v $(pwd):/workspace nvcr.io/nvidia/tensorflow:23.04-tf2-py3

# in container
pip install -U tf2onnx
python -m tf2onnx.convert --input frozen_east_text_detection.pb --inputs "input_images:0" --outputs "feature_fusion/Conv_7/Sigmoid:0","feature_fusion/concat_3:0" --output detection.onnx
```

#### Convert "None-ResNet-None-CTC.pth" to "str.onnx"

```bash
docker run -it --gpus all -v $(pwd):/workspace nvcr.io/nvidia/pytorch:23.04-py3

# in container
python convert_CTC_pth_to_onnx.py
```

#### Setting up the model repository

```bash
# Example repository structure
<model-repository>/
  <model-name>/
    [config.pbtxt]
    [<output-labels-file> ...]
    <version>/
      <model-definition-file>
    <version>/
      <model-definition-file>
    ...
  <model-name>/
    [config.pbtxt]
    [<output-labels-file> ...]
    <version>/
      <model-definition-file>
    <version>/
      <model-definition-file>
    ...
  ...
```

#### Run TRTIS(Triton Inference Server)

```bash
docker run --gpus=all -it --shm-size=256m --rm -p8000:8000 -p8001:8001 -p8002:8002 -v $(pwd)/model_repository:/models nvcr.io/nvidia/tritonserver:23.04-py3

# in container
tritonserver --model-repository=/models
```

### Client

#### Virtual Env (pyenv + poetry)

```bash
pyenv versions
pyenv install 3.9.15

pyenv shell 3.9.15
poetry env use $(pyenv which python)
poetry install
# message: Failed to create the collection: Prompt dismissed..
# ==> export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring

poetry show
poetry debug info
```

#### Run

```bash
poetry run python client.py
```

## Conceptual Guide :: Part 2

### Server

#### Change Directory

```bash
cd Conceptual_Guide/Part_1-model_deployment  # stay in Part 1
```

#### Convert "None-ResNet-None-CTC.pth" to "str_batch.onnx"

```bash
docker run -it --gpus all -v $(pwd):/workspace nvcr.io/nvidia/pytorch:23.04-py3

# in container
python convert_CTC_pth_to_onnx_batch.py
```

#### Change Directory

```bash
cd ../Part_2-improving_resource_utilization
```

#### On a third terminal, it is advisable to monitor the GPU Utilization to see if the deployment is saturating GPU resources.

```bash
watch -n 0.1 nvidia-smi
```

### Performance Test 1 (Use config.1.pbtxt)

`No Dynamic Batching, single model instance`

```bash
# Server
docker run --gpus=1 -it --shm-size=256m --rm -p8000:8000 -p8001:8001 -p8002:8002 -v $(pwd):/workspace/ -v $(pwd)/model_repository:/models nvcr.io/nvidia/tritonserver:23.04-py3 bash

# in container
tritonserver --model-repository=/models
```

```bash
# Client
docker run -it --net=host -v $(pwd):/workspace/ nvcr.io/nvidia/tritonserver:23.04-py3-sdk bash

# in container
perf_analyzer -m text_recognition -b 2 --shape input.1:1,32,100 --concurrency-range 2:16:2 --percentile=95
```

### Performance Test 2 (Use config.2.pbtxt)

`Just Dynamic Batching`

```bash
# Server
docker run --gpus=1 -it --shm-size=256m --rm -p8000:8000 -p8001:8001 -p8002:8002 -v $(pwd):/workspace/ -v $(pwd)/model_repository:/models nvcr.io/nvidia/tritonserver:23.04-py3 bash

# in container
tritonserver --model-repository=/models
```

```bash
# Client
docker run -it --net=host -v $(pwd):/workspace/ nvcr.io/nvidia/tritonserver:23.04-py3-sdk bash

# in container
perf_analyzer -m text_recognition -b 2 --shape input.1:1,32,100 --concurrency-range 2:16:2 --percentile=95
```

### Performance Test 3 (Use config.3.pbtxt)

`Dynamic Batching with multiple model instances`

```bash
# Server
docker run --gpus=1 -it --shm-size=256m --rm -p8000:8000 -p8001:8001 -p8002:8002 -v $(pwd):/workspace/ -v $(pwd)/model_repository:/models nvcr.io/nvidia/tritonserver:23.04-py3 bash

# in container
tritonserver --model-repository=/models
```

```bash
# Client
docker run -it --net=host -v $(pwd):/workspace/ nvcr.io/nvidia/tritonserver:23.04-py3-sdk bash

# in container
perf_analyzer -m text_recognition -b 2 --shape input.1:1,32,100 --concurrency-range 2:16:2 --percentile=95
```

## Conceptual Guide :: Part 3

## Conceptual Guide :: Part 4

## Conceptual Guide :: Part 5

## Conceptual Guide :: Part 6
