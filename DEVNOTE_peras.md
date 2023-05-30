# DEVNOTE

## 0. Preparation

### 0.1 Virtual Env (pyenv + poetry)

```bash
# install pyenv
curl https://pyenv.run | bash

# install poetry
curl -sSL https://install.python-poetry.org | python3 -
```

```bash
pyenv versions
pyenv install 3.9.15

cd to/project/root # triton-tutorials-meta

pyenv shell 3.9.15
poetry env use $(pyenv which python)
poetry install
# message: Failed to create the collection: Prompt dismissed..
# ==> export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring

poetry show
poetry debug info
```

### 0.2 On a third terminal, it is advisable to monitor the GPU Utilization to see if the deployment is saturating GPU resources.

```bash
watch -n0.1 -d nvidia-smi
```

## 1. Conceptual Guide :: Part 1

### 1.1 Server

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
docker run --gpus=1 -it --shm-size=256m --rm -p8000:8000 -p8001:8001 -p8002:8002 -v $(pwd)/model_repository:/models nvcr.io/nvidia/tritonserver:23.04-py3 # gpu count == 1 (not gpu id)
docker run --gpus='"device=2,3"' -it --shm-size=256m --rm -p8000:8000 -p8001:8001 -p8002:8002 -v $(pwd)/model_repository:/models nvcr.io/nvidia/tritonserver:23.04-py3

# in container
tritonserver --model-repository=/models
```

### 1.2 Client

#### Run

```bash
# w/o docker (recommanded)
poetry run python client.py

# w/ docker
docker run --gpus=all -it --rm --net=host --name triton-tut-part1-client -v $(pwd):/workspace python:3.9.16 bash
pip install tritonclient[http] opencv-python-headless
cd /workspace
python client.py
```

## 2. Conceptual Guide :: Part 2

### 2.1 Server

#### Change Directory

```bash
cd Conceptual_Guide/Part_1-model_deployment # stay in Part 1
```

#### Convert "None-ResNet-None-CTC.pth" to "str_batch.onnx"

```bash
docker run -it --gpus all -v $(pwd):/workspace nvcr.io/nvidia/pytorch:23.04-py3

# in container
python convert_CTC_pth_to_onnx_batch.py
```

#### Change Directory

```bash
cd Conceptual_Guide/Part_2-improving_resource_utilization # Part 2
```

#### Setting up the model repository

```bash
# triton-tutorials-meta/Conceptual_Guide/Part_2-improving_resource_utilization/model_repository/text_recognition/1

...
```

### 2.2 Performance Test

```bash
cd Conceptual_Guide/Part_2-improving_resource_utilization # Part 2
```

#### 2.2.1 No Dynamic Batching, single model instance

`Use config.1.pbtxt`

```bash
# Server
docker run --gpus=1 -it --shm-size=256m --rm -p8000:8000 -p8001:8001 -p8002:8002 -v $(pwd):/workspace/ -v $(pwd)/model_repository:/models nvcr.io/nvidia/tritonserver:23.04-py3 bash

docker run --gpus='"device=1"' -it --shm-size=256m --rm -p8000:8000 -p8001:8001 -p8002:8002 -v $(pwd):/workspace/ -v $(pwd)/model_repository:/models nvcr.io/nvidia/tritonserver:23.04-py3 bash

docker run --gpus='"device=2"' -it --shm-size=256m --rm -p8000:8000 -p8001:8001 -p8002:8002 -v $(pwd):/workspace/ -v $(pwd)/model_repository:/models nvcr.io/nvidia/tritonserver:23.04-py3 bash

# in container
tritonserver --model-repository=/models
```

```bash
# Client
docker run -it --net=host -v $(pwd):/workspace/ nvcr.io/nvidia/tritonserver:23.04-py3-sdk bash

# in container
perf_analyzer -m text_recognition -b 2 --shape input.1:1,32,100 --concurrency-range 2:16:2 --percentile=95 > perf.1.log
```

#### 2.2.2 Just Dynamic Batching

`Use config.2.pbtxt`

```bash
# Server
docker run --gpus=1 -it --shm-size=256m --rm -p8000:8000 -p8001:8001 -p8002:8002 -v $(pwd):/workspace/ -v $(pwd)/model_repository:/models nvcr.io/nvidia/tritonserver:23.04-py3 bash

docker run --gpus='"device=1"' -it --shm-size=256m --rm -p8000:8000 -p8001:8001 -p8002:8002 -v $(pwd):/workspace/ -v $(pwd)/model_repository:/models nvcr.io/nvidia/tritonserver:23.04-py3 bash

docker run --gpus='"device=2"' -it --shm-size=256m --rm -p8000:8000 -p8001:8001 -p8002:8002 -v $(pwd):/workspace/ -v $(pwd)/model_repository:/models nvcr.io/nvidia/tritonserver:23.04-py3 bash

# in container
tritonserver --model-repository=/models
```

```bash
# Client
docker run -it --net=host -v $(pwd):/workspace/ nvcr.io/nvidia/tritonserver:23.04-py3-sdk bash

# in container
perf_analyzer -m text_recognition -b 2 --shape input.1:1,32,100 --concurrency-range 2:16:2 --percentile=95 > perf.2.log
```

#### 2.2.3 Dynamic Batching with multiple model instances

`Use config.3.pbtxt`

```bash
# Server
docker run --gpus=1 -it --shm-size=256m --rm -p8000:8000 -p8001:8001 -p8002:8002 -v $(pwd):/workspace/ -v $(pwd)/model_repository:/models nvcr.io/nvidia/tritonserver:23.04-py3 bash

docker run --gpus='"device=1"' -it --shm-size=256m --rm -p8000:8000 -p8001:8001 -p8002:8002 -v $(pwd):/workspace/ -v $(pwd)/model_repository:/models nvcr.io/nvidia/tritonserver:23.04-py3 bash

docker run --gpus='"device=2"' -it --shm-size=256m --rm -p8000:8000 -p8001:8001 -p8002:8002 -v $(pwd):/workspace/ -v $(pwd)/model_repository:/models nvcr.io/nvidia/tritonserver:23.04-py3 bash

# in container
tritonserver --model-repository=/models
```

```bash
# Client
docker run -it --net=host -v $(pwd):/workspace/ nvcr.io/nvidia/tritonserver:23.04-py3-sdk bash

# in container
perf_analyzer -m text_recognition -b 2 --shape input.1:1,32,100 --concurrency-range 2:16:2 --percentile=95 > perf.3.log
```

## 3. Conceptual Guide :: Part 3

## 4. Conceptual Guide :: Part 4

## 5. Conceptual Guide :: Part 5

## 6. Conceptual Guide :: Part 6
