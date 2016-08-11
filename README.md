# 3D-R2N2: Recurrent Reconstruction Neural Network

This is the source code for the paper `3D-R2N2: A Unified Approach for Single and Multi-view 3D Object Reconstruction, ECCV 2016`. Given one or multiple views of an object, the network generate voxelized (voxel is 3D equivalent of pixel) reconstruction of the object in 3D.


## Overview

Given a set of images

- [ShapeNet rendering](http://cvgl.stanford.edu/data2/ShapeNetRendering.tgz)
- [ShapeNet voxelized models](http://cvgl.stanford.edu/data2/ShapeNetVox32.tgz)

## Installation

1. Download `ShapeNetCore` dataset
2. Set the `MODEL_ROOT_PATH` in `lib/config.py` to the extracted `ShapeNetCore`
3. Download model lists from the website. Some models do not have any faces.
4. Generate dataset `json` file by running `python tools/gen_category_list.py`
5. Voxelize all the models by running `python tools/voxelize_models.py`
6. Render all the models by runnning `python tools/render_models.py`. To run this step, you have to setup `blender`.
7. Set `cfg.DIR.MODEL_PATH`, `cfg.DIR.RENDERING_PATH` and `cfg.DIR.VOXEL_PATH` in `lib/config.py` accoringly
8. Run experiments `bash ./experiments/script/mv_lstm_vec_net.sh`

## Installation

### CUDA Setup

Follow the [instruction](http://deeplearning.net/software/theano/install.html) and set GPU + CUDA.

1. `CUDA_ROOT=/path/to/cuda/root`
2. add a cuda.root flag to THEANO_FLAGS, as in `THEANO_FLAGS='cuda.root=/path/to/cuda/root'`
3. add a [cuda] section to your .theanorc file containing the option root = /path/to/cuda/root.

Download `cuDNN3` and follow the [instruction](http://deeplearning.net/software/theano/library/sandbox/cuda/dnn.html)

A non-intrusive installation is to set

```
export LD_LIBRARY_PATH=/home/user/path_to_CUDNN_folder/lib64:$LD_LIBRARY_PATH
export CPATH=/home/user/path_to_CUDNN_folder/include:$CPATH
export LIBRARY_PATH=/home/user/path_to_CUDNN_folder/lib64:$LD_LIBRARY_PATH
```

### Theano

Install bleeding-edge Theano.

```
pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git
```

#### Theanorc Setup

```
[global]
floatX = float32
base_compiledir = /var/tmp

# Get an error message if cuDNN is not available
optimizer_including = cudnn
```

#### Issues

- Upgrading from cuDNNv2

    If you encounter

    ```
    ImportError: libcudnn.so.6.5: cannot open shared object file: No such file or directory
    ```

    Delete all binary files in the `base_compiledir` [link](http://deeplearning.net/software/theano/library/config.html#config.base_compiledir)


### Blender

Go to the blender website and download the [latest blender](https://developer.blender.org/diffusion/B/)

```
# read-only access
git clone git://git.blender.org/blender.git

# read/write access
git clone git@git.blender.org:blender.git

cd blender
git submodule update --init --recursive
git submodule foreach git checkout master
git submodule foreach git pull --rebase origin master
```

Then, follow the instruction on compiling the blender with the python module support [http://wiki.blender.org/index.php/User:Ideasman42/BlenderAsPyModule]

Make sure to set the following flags correctly.

```
WITH_PYTHON_INSTALL=OFF
WITH_PLAYER=OFF
WITH_PYTHON_MODULE=ON
...
WITH_OPENCOLLADA=ON
```

OpenCollada is optional if you wan to use collada (.dae) files. Install OpenCollada first.


### OpenCollada

This will requires gcc4.7

```
git clone https://github.com/KhronosGroup/OpenCOLLADA
mkdir OpenCOLLADA-build
cd OpenCOLLADA-build
cmake ../OpenCOLLADA -DUSE_SHARED=ON
make
make install
```

## Python 3 Requirements

On the root path, run

```
pip3 install -r requirements.txt
```


# Dataset Setting

- Download ShapeNet
- Set variables in lib/config.py
- Generate training dataset lists
    ```
    python tools/generate_category_list.py
    ```
- Generate rendering and voxelization
- Run the training script


# Erroneous Files

03624134/67ada28ebc79cc75a056f196c127ed77/model.obj
04090263/4a32519f44dc84aabafe26e2eb69ebf4/model.obj
04074963/b65b590a565fa2547e1c85c5c15da7fb/model.obj

