# DEVNOTE

## Conceptual Guide :: Part 1

### Server

#### Change Directory

```bash
cd Conceptual_Guide/Part_1-model_deployment
```

#### Convert "frozen_east_text_detection.pb" to "detection.onnx"

```bash
docker run -it --gpus all -v ${PWD}:/workspace nvcr.io/nvidia/tensorflow:23.04-tf2-py3

# in container
pip install -U tf2onnx
python -m tf2onnx.convert --input frozen_east_text_detection.pb --inputs "input_images:0" --outputs "feature_fusion/Conv_7/Sigmoid:0","feature_fusion/concat_3:0" --output detection.onnx
```

#### Convert "None-ResNet-None-CTC.pth" to "str.onnx"

```bash
docker run -it --gpus all -v ${PWD}:/workspace nvcr.io/nvidia/pytorch:23.04-py3
# or
docker run -it --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v ${PWD}:/workspace nvcr.io/nvidia/pytorch:23.04-py3

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
