## DeepDeblur: Text Image Recovery from Blur to Sharp

This work focuses on recovering the blurry text image. 

## Introduction ##

Based on the deep neural network, a new short connection scheme is used. Trained by the pixel regression, higher visual quality of the image can be recovered by the network from the blurry one.

 * Sequential Highway Connection (SHC) structure

    <img src="http://meijianhan.github.io/JHMei/Paper/Deblur/SHCStrc_1.png" width="600" class="center">

 * Loss curve comparing with the ResNet Structure:

    <img src="http://meijianhan.github.io/JHMei/Paper/Deblur/LossHeNew.png" width="450" class="center">

 * Visualization

    <img src="http://meijianhan.github.io/JHMei/Paper/Deblur/Testing_1.png" width="500" class="center">

    

## Citation ##

If you find our work useful in your research and relevant works, please consider citing:

```
@Article{Mei2019,
author="Mei, Jianhan and Wu, Ziming and Chen, Xiang and Qiao, Yu and Ding, Henghui and Jiang, Xudong",
title="DeepDeblur: text image recovery from blur to sharp",
journal="Multimedia Tools and Applications",
year="2019",
issn="1573-7721",
doi="10.1007/s11042-019-7251-y",
url="https://doi.org/10.1007/s11042-019-7251-y"
}
```

[comment]: # (Created by:)

[comment]: # (School of Electrical and Electronics Engineering, Nanyang Technological University, Singapore)
[comment]: # (Jianhan Mei at Nanyang Technological University)

[comment]: # (Technische Universit\"at Darmstadt, Darmstadt, Germany)
[comment]: # (Xiang Chen at Darmstadt University of Technology)

[comment]: # (The Hong Kong University of Science and Technology, Hong Kong, China)
[comment]: # (Ziming Wu at The Hong Kong University of Science and Technology)



## Required Environment ##

 - [Ubuntu](https://www.ubuntu.com/) 16.04

 - Python 2/3, in case you need the sufficient scientific computing packages, we recommend you to install [anaconda](https://www.anaconda.com/what-is-anaconda/).

 - [Tensorflow](https://www.tensorflow.org/) >= 1.5.0

 - [Keras](https://keras.io/) >= 2.2.0

 - Optional: if you need GPUs acceleration, please install [CUDA](https://developer.nvidia.com/cuda-toolkit) that the version requires >= 9.0

   

## Running The Demo ##
- ##### Text Image Dataset Generation

  Check the matlab script 'Matlab/RunProcess.m': The path of the text images should contain raw sharp text images. You can build your own dataset by convert PDF files into raw image files and save them to the text image path in "Matlab/RunProcess.m". 

  Then run the matlab script 'Matlab/RunProcess.m', which helps to build the training dataset.

- ##### Training

  Check the training data and model saving paths in "train.py", for which the training data should be consistent with the previous step. Then run the following script:

  ```python
  python train.py
  ```

- ##### Testing

  Check the testing model and data paths in "test.py". Then run the following script:

  ```python
  python test.py
  ```

  

## License

This work is released under the MIT License (refer to the LICENSE file for details).


