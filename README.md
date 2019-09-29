# Learning-to-Extract-Flawless-Slow-Motion-from-Blurry-Videos
This repository is a PyTorch implementation of the paper "Learning to Extract Flawless Slow Motion from Blurry Videos" from CVPR 2019 [[paper]](https://github.com/MeiguangJin/slow-motion/blob/master/cvpr19.pdf)[[full version]](https://github.com/MeiguangJin/slow-motion/blob/master/full_version.pdf)[[video]](https://drive.google.com/open?id=17RI3XkYs9CMlGshietzCbZSR8xMxBftj)

If you find our work useful in your research or publication, please cite our work:

@InProceedings{Jin_2019_CVPR,  
author = {Jin, Meiguang and Hu, Zhe and Favaro, Paolo},  
title = {Learning to Extract Flawless Slow Motion from Blurry Videos},  
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},  
month = {June},  
year = {2019}  
}  
## **Requirements**  
This code has been tested with Python 3.7 and Pytorch 1.1.0. 

## **Test**
Unzip a real test video and download the pretrained model (cvpr19_model.pth) from [google drive](https://drive.google.com/open?id=1gfhHKpJEYKrqx2wJ4GL9owGB4J7E7UD-). This script will generate a 10x slow motion video.
```
unzip test_video_01.zip
python test_demo.py --cuda --model cvpr19_model.pth --input test_video_01 --out result
```  
## **Dataset**
You can find the sony slow motion video dataset used in the training from the following [[link]](http://www.cvg.unibe.ch/media/data/datasets/video/jin/slow-motion.zip). 

## **Training**  
To be updated  

## **Contact**
If you have any suggestions and questions, please send an email to jinmeiguang@gmail.com
