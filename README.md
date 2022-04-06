# TorchScene

A scene recognition tool based on pytorch

## Installation

```bash
conda env create -f environment.yml python=3.7

conda activate scene_pytorch_tf

export PYTHONPATH=$PYTHONPATH:$(pwd)
```

## Model Zoo (Pretrained Models)

Please refer [[Model Zoo]](model_zoo.md)

## Step by step

Please download the data from [[Place2 Data]](http://places2.csail.mit.edu/download.html)


### 1. Download the data

These images are 256x256 images, in a more friendly directory structure that in train and val split the images are organized such as train/reception/00003724.jpg and val/raft/000050000.jpg

```python
sh download_data_pytorch.sh
```

### 2. Train the model with multiple GPUs

```bash
python tools/train.py
```

### 3. Remove the .module

```python
python scripts/remove_pytorch_module.py
```

### 4. Test a model

Donwload pretrained weights from [[Google Drive]](https://drive.google.com/drive/folders/1NbV3NZlgbqnLSd9zwZoz8kFpNQjUYolT?usp=sharing)

```python
python tools/test.py
```

### 5. Convert a model to TorchScript

```python
python scripts/convert_torchscript.py
```

## Acknowledge

The dataset and basic code comes from [[MIT Place365]](https://github.com/CSAILVision/places365)

Thanks for the great work!
