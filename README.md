# NetMind-Face
This repo is the demo for real-time face attributes editing by using webcam

## Preparing
1 clone the github repo


2 install deps:

conda create --name demo_stgan python=3.8

conda activate demo_stgan

conda install matplotlib

pip install opencv-python

pip install pyyaml

pip install torch torchvision torchaudio

pip install torchsummary

pip install Cython

bash build.sh

3 download pretrained [weights](https://drive.google.com/file/d/1mgl5F5ze5qPls6vjCYwnshlMmTeg5RuE/view?usp=sharing) and put it into ./weights/stgan/ 


## demo
python demo_stgan.py

## showcase

![showcase](https://github.com/protagolabs/NetMind-FacialAttributeEditing/blob/main/demo_stgan.gif)

from left -> right: Original, Bangs, Eyeglasses, Mouth_Slightly_Open,  Mustache, Pale_Skin

## acknowledgement
[3DDFA_V2](https://github.com/cleardusk/3DDFA_V2)
