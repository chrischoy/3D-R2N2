# 3D-R<sup>2</sup>N<sup>2</sup>: 3D Recurrent Reconstruction Neural Network

This repository contains the source codes for the paper [Choy et al., 3D-R2N2: A Unified Approach for Single and Multi-view 3D Object Reconstruction, ECCV 2016](http://arxiv.org/abs/1604.00449). Given one or multiple views of an object, the network generates voxelized (voxel is 3D equivalent of pixel) reconstruction of the object in 3D.

## Citing this work

If you find this work useful in your research, please consider citing:

```
@article{choy20163d,
  title={3D-R2N2: A Unified Approach for Single and Multi-view 3D Object Reconstruction},
  author={Choy, Christopher B and Xu, Danfei and Gwak, JunYoung and Chen, Kevin and Savarese, Silvio},
  journal={arXiv preprint arXiv:1604.00449},
  year={2016}
}
```

## Overview

![Overview](imgs/overview.png)
*Left: images found on Ebay, Amazon, Right: overview of `3D-R2N2`*

Traditionally, single view reconstruction and multi view reconstruction are disjoint problmes that has been dealt using different approaches. In this work, we first propose a unified framework for both single and multi view reconstruction using a `3D Recurrent Reconstruction Neural Network` (3D-R2N2).

| The schematic of the `3D-Convolutional LSTM` | Inputs (red cells + feature) for each cell (purple) |
|:--------------------------------------------:|:---------------------------------------------------:|
| ![3D-LSTM](imgs/lstm.png)                    | ![3D-LSTM](imgs/lstm_time.png)                      |

We can feed in images in random order since the network is trained to be invariant to the order. The ciritical component that enables the network to be invariant to the order is the `3D-Convolutional LSTM` which we first proposed in this work. The `3D-Convolutional LSTM` selectively updates parts that are visible and keeps the parts that are self occluded (please refer to [http://cvgl.stanford.edu/3d-r2n2/](http://cvgl.stanford.edu/3d-r2n2/) for the supplementary material for analysis).

![Networks](imgs/full_network.png)
*We used two different types of networks for the experiments: a shallow network (top) and a deep residual network (bottom).*


## Datasets

We used [ShapeNet](http://shapenet.cs.stanford.edu) models to generated rendered images and voxelized models which are available below (you can follow the installation instruction below to extract it on the default directory).

- ShapeNet rendered images [ftp://cs.stanford.edu/cs/cvgl/ShapeNetRendering.tgz](ftp://cs.stanford.edu/cs/cvgl/ShapeNetRendering.tgz)
- ShapeNet voxelized models [ftp://cs.stanford.edu/cs/cvgl/ShapeNetVox32.tgz](ftp://cs.stanford.edu/cs/cvgl/ShapeNetVox32.tgz)


## Installation

The package requires python3. You can follow the direction below to install virtual environment within the repository or install anaconda for python 3.

- Download the repository

```
git clone https://github.com/chrischoy/3D-R2N2.git
```

- Setup virtual environment and install requirements and copy the theanorc file to the `$HOME` directory

```
cd 3D-R2N2
pip install virtualenv
virtualenv -p python3 py3
source py3/bin/activate
pip install -r requirements.txt
cp .theanorc ~/.theanorc
```

- Install meshlab (skip if you have another mesh viewer). If you skip this step, demo code will not visualize the final prediction.

```
sudo apt-get install meshlab
```

- Run the demo code and save the final 3D reconstruction to a meshfile named `prediction.obj`

```
python demo.py prediction.obj
```

The demo code takes 3 images of the same chair and generates the following reconstruction.

| Image 1         | Image 2         | Image 3         | Reconstruction                                                                            |
|:---------------:|:---------------:|:---------------:|:-----------------------------------------------------------------------------------------:|
| ![](imgs/0.png) | ![](imgs/1.png) | ![](imgs/2.png) | <img src="https://github.com/chrischoy/3D-R2N2/blob/master/imgs/pred.png" height="127px"> |

- Deactivate your environment when you are done

```
deactivate
```

**Note**

Activate the virtual environment before you run the experiments.

```
source py3/bin/activate
```

### Training the network

- Download datasets and place them in a folder named `ShapeNet`

```
mkdir ShapeNet/
wget ftp://cs.stanford.edu/cs/cvgl/ShapeNetRendering.tgz
wget ftp://cs.stanford.edu/cs/cvgl/ShapeNetVox32.tgz
tar -xzf ShapeNetRendering.tgz -C ShapeNet/
tar -xzf ShapeNetVox32.tgz -C ShapeNet/
```

- Train and test the network using the training shell script

```
./experiments/script/res_gru_net.sh
```

## LICENSE

MIT License
