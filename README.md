# SQN: Weakly-Supervised Semantic Segmentation of Large-Scale 3D Point Clouds (ECCV2022)

This repository contains a PyTorch implementation of [SQN](https://github.com/QingyongHu/SQN) on SemanticKitti (S3DIS have not been supported).

This repository is based on the [repository](https://github.com/qiqihaer/RandLA-Net-pytorch)

## Preparation

1. Clone this repository
2. Install some Python dependencies, such as scikit-learn. All packages can be installed with pip.
3. env : ubuntu 18.04, python 3.7.16, torch 1.12.1, numpy 1.21.5, torchvision 0.13.1, scikit-learn 0.22.2, pandas 1.3.5, tqdm 4.64.1, Cython 0.29.33
4. Install python functions. the functions and the codes are copied from the [official implementation with Tensorflow](https://github.com/QingyongHu/RandLA-Net).

```
sh compile_op.sh
```

5. Attention: please check out *./utils/nearest_neighbors/lib/python/KNN_NanoFLANN-0.0.0-py3.7-linux-x86_64.egg/* and copy the .so file to the parent folder(update in 2023.2.23: We provide a .so file of python3.7, and you don't need to copy if you are using python3.7)
6. Download the SemanticKITTI[ dataset](http://semantic-kitti.org/dataset.html#download), and preprocess the data:

```
  python utils/data_prepare_semantickitti.py
```

   Note: Please change the dataset path in the 'data_prepare_semantickitti.py' with your own path.

## Train a model

```
  python main_SemanticKITTI.py
```

or 

```
  python main_SemanticKITTI_aug.py
```
for data augmentation.


## Results

Result from [codelab](https://codalab.lisn.upsaclay.fr/competitions/6280#participate-submit_results) competition w/o data augmentation which is close to the origin paper(mIoU: 50.8)

mean IoU 50.6

mean accuracy 87.8

car 92.1

bicycle 26.0

motorcycle 27.1

truck 36.1

other-vehicle 24.2

person 38.8

bicyclist 43.4

motorcyclist 4.9

road 90.1

parking 56.2

sidewalk 72.7

other-ground 21.4

building 84.8

fence 53.2

vegetation 80.3

trunk 59.2

terrain 66.2

pole 44.1

traffic-sign 41.2


