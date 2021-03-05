# Efficient Conditional GAN Transfer with Knowledge Propagation across Classes
Authors: [Mohamad Shahbazi](https://people.ee.ethz.ch/~mshahbazi/), [Zhiwu Huang](https://zhiwu-huang.github.io/), [Danda P.Paudel](https://people.ee.ethz.ch/~paudeld/), [Ajad Chhatkuli](https://scholar.google.ch/citations?hl=en&user=3BHMHU4AAAAJ), and [Luc Van Gool](https://scholar.google.ch/citations?hl=en&user=TwMib_QAAAAJ)
<br> Paper: https://arxiv.org/pdf/2102.06696.pdf

![alt text](images/concept.png)


## Abstract
Generative adversarial networks (GANs) have shown impressive results in both unconditional and conditional image generation. In recent literature, it is shown that pre-trained GANs, on a different dataset, can be transferred to improve the image generation from a small target data. The same, however, has not been well-studied in the case of conditional GANs (cGANs), which provides new opportunities for knowledge transfer compared to unconditional setup. In particular, the new classes may borrow knowledge from the related old classes, or share knowledge among themselves to improve the training. This motivates us to study the problem of efficient conditional GAN transfer with knowledge propagation across classes. To address this problem, we introduce a new GAN transfer method to explicitly propagate the knowledge from the old classes to the new classes. The key idea is to enforce the popularly used conditional batch normalization (BN) to learn the class-specific information of the new classes from that of the old classes, with implicit knowledge sharing among the new ones. This allows for an efficient knowledge propagation from the old classes to the new classes, with the BN parameters increasing linearly with the number of new classes. The extensive evaluation demonstrates the clear superiority of the proposed method over state-of-the-art competitors for efficient conditional GAN transfer tasks. 
The code will be available here soon.

## Overview
1. [Installation & Dependencies](#Dependencies)
2. [Prepration](#Prepration)
    1. [Directories](#Directories)
    2. [Data](#Data)
    3. [Weights](#Weights)
3. [Training](#Training)
4. [Evaluation](#Evaluation)
5. [Results](#Results)
6. [Contact](#Contact)
7. [How to cite](#How-to-cite)

## 1. Installation & Dependencies<a name="Dependencies"></a>
The code has been tested with Python 3.6 and Pytorch 1.0.1.


To install the dependencies:
```bash
pip install -r requirements.txt
```

## 2. Prepration<a name="Prepration"></a>
### 2.1. Directories<a name="Directories"></a>
The base directory will contain the following sub-directories:
<ul>
  <li><b>"data"</b>: Contains the training data</li>
  <li><b>"weights"</b>: Contains the pretraining and training weights</li>
  <li><b>"logs"</b>: Contains the training logs</li>
  <li><b>"samples"</b>: Contains the saved images during evaluation</li>
</ul> 
Before training, you should create thesub-directories <b>"data"</b> and <b>"weights"</b>. Then place the data and pretraining weights inside them accordingly. the sub-directories <b>"logs"</b> and <b>"samples"</b> will be created automatically during the training.

### 2.2. Data<a name="Data"></a>
### 2.3. Pretrained Weights<a name="Weights"></a>
Before training, BigGAN's pretrained weights should be placed in the sub-directory "weights" of the base directory.

For BigGAN on ImageNet, you can use the [pretrained weights](https://github.com/ajbrock/BigGAN-PyTorch#pretrained-models) provided by the BigGAN's authors. In this project, we have used the [main checkpoint](https://drive.google.com/file/d/1nAle7FCVFZdix2--ks0r5JBkFnKw8ctW/view).

If you want to use other datasets (e.g. Cifar10/100) as the pretraining dataset, you can first train the [BigGAN](https://github.com/ajbrock/BigGAN-PyTorch) on the desired dataset, and then, use the pretrained weights for cGANTransfer.

## 3. Training<a name="Training"></a>

## 4. Evaluation<a name="Evaluation"></a>
The main evaluation metrics used in this project are Frechet Inception Distance (FID) and Kernel Maximum Mean Discrepancy (KMMD). Inception score (IS) is also included in the code.

FID and IS are calculated during the training using the Pytorch implementation provided by [BigGAN](https://github.com/ajbrock/BigGAN-PyTorch).

For KMMD, the default implementation provided by [GAN Metrics](https://github.com/xuqiantong/GAN-Metrics) is used (Gaussian kernel with sigma=1).

KMMD calculation is deactivated by default to avoid memory errors. To activate the KMMD calculation, use the flag "--kmmd" in the script and reduce the number of generated samples used for evaluation using "----num_inception_images" (default=50000. We were able to avoid memory errors by setting it to 25000). 

Note that reducing the number of generated samples might make the FID calculation a bit less accurate.


## 5. Results<a name="Results"></a>

## 6. Contact<a name="Contact"></a>
For any questions, suggestions, or issues with the code, please contact Mohamad at <a>mshahbazi@vision.ee.ethz.ch</a>.

## 7. How to cite<a name="How-to-cite"></a>
If you find this project helpful, please consider citing us as follows:
```bash
@inproceedings{shahbazi2021cGANTransfer,
      title = {Efficient Conditional GAN Transfer with Knowledge Propagation across Classes},
      author    = {Shahbazi, Mohamad and
                   Huang, Zhiwu and
                   Paudel, ‪Danda Pani and
                   Chhatkuli, Ajad and
                   Van Gool, Luc},
      year = {2021},
      booktitle = {2021 {IEEE} Conference on Computer Vision and Pattern Recognition, {CVPR} 2021}
}
```





