# Adaptive Composing Augmentations on Multi-modal Graph Convolutional Network for Disease Prediction

## About
This is a Pytorch implementation of ACA-GCN. This project is highly borrowed from [Edge-variational Graph Convolutional Networks for Uncertainty-aware Disease Prediction](https://github.com/SamitHuang/EV_GCN.git) (MICCAI 2020) by Yongxiang Huang and Albert C.S. Chung.

## Prerequisites
- `Python 3.7.4+`
- `Pytorch 1.4.0`
- `torch-geometric `
- `scikit-learn`
- `NumPy 1.16.2`

Ensure Pytorch 1.4.0 is installed before installing torch-geometric. 

This code has been tested using `Pytorch` on a GTX1080TI GPU.

## Training
To train on ABIDE dataset, please run
```
python train_eval.py --dataset ABIDE --num_classes 2 --train 1
```
To train on ADNI dataset, please run
```
python train_eval.py --dataset ADNI --num_classes 2 --train 1
```
To train on ODIR dataset, please run
```
python train_eval.py --dataset ODIR --num_classes 8 --train 1
```

To get a detailed description for available arguments, please run
```
python train_eval.py --help
```
To download the used dataset, please run the following script in the `data` folder: 
```
python data/fetch_data.py 
```
If you want to train a new model on your own dataset, please change the data loader functions defined in `dataloader.py` accordingly, then run `python train_eval.py --train=1`  

## Inference and Evaluation
```
python train_eval.py --train=0
```

## Reference 
If you find this code useful in your work, please cite:


