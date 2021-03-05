# Efficient Conditional GAN Transfer with Knowledge Propagation across Classes
Authors: [Mohamad Shahbazi](https://people.ee.ethz.ch/~mshahbazi/), [Zhiwu Huang](https://zhiwu-huang.github.io/), [Danda P.Paudel](https://people.ee.ethz.ch/~paudeld/), [Ajad Chhatkuli](https://scholar.google.ch/citations?hl=en&user=3BHMHU4AAAAJ), and [Luc Van Gool](https://scholar.google.ch/citations?hl=en&user=TwMib_QAAAAJ)
<br> Paper: https://arxiv.org/pdf/2102.06696.pdf

![alt text](images/concept.png)


## Abstract
<p style="text-align: justify"> your-text-here 
Generative adversarial networks (GANs) have shown impressive results in both unconditional and conditional image generation. In recent literature, it is shown that pre-trained GANs, on a different dataset, can be transferred to improve the image generation from a small target data. The same, however, has not been well-studied in the case of conditional GANs (cGANs), which provides new opportunities for knowledge transfer compared to unconditional setup. In particular, the new classes may borrow knowledge from the related old classes, or share knowledge among themselves to improve the training. This motivates us to study the problem of efficient conditional GAN transfer with knowledge propagation across classes. To address this problem, we introduce a new GAN transfer method to explicitly propagate the knowledge from the old classes to the new classes. The key idea is to enforce the popularly used conditional batch normalization (BN) to learn the class-specific information of the new classes from that of the old classes, with implicit knowledge sharing among the new ones. This allows for an efficient knowledge propagation from the old classes to the new classes, with the BN parameters increasing linearly with the number of new classes. The extensive evaluation demonstrates the clear superiority of the proposed method over state-of-the-art competitors for efficient conditional GAN transfer tasks. 
The code will be available here soon.
</p>

## Overview
<ol>
  <li><a href="https://github.com/mshahbazi72/cGANTransfer/blob/main/README.md#1dependencies">Dependencies</a></li>
  <li><a href="https://github.com/mshahbazi72/cGANTransfer/blob/main/README.md#2prepration">Prepration</a>
    <ol>
      <li><a href="https://github.com/mshahbazi72/cGANTransfer/blob/main/README.md#21directories">Directories</a></li>
      <li><a href="https://github.com/mshahbazi72/cGANTransfer/blob/main/README.md#22data">Data</a></li>
      <li><a href="https://github.com/mshahbazi72/cGANTransfer/blob/main/README.md#23weights">Weights</a></li>
    </ol>
  </li>
  <li><a href="https://github.com/mshahbazi72/cGANTransfer/blob/main/README.md#3training">Training</a></li>
  <li><a href="https://github.com/mshahbazi72/cGANTransfer/blob/main/README.md#4results">Results</a></li>
  <li><a href="https://github.com/mshahbazi72/cGANTransfer/blob/main/README.md#5contact">Contact</a></li>
  <li><a href="https://github.com/mshahbazi72/cGANTransfer/blob/main/README.md#6how-to-cite">How to cite</a></li>
</ol>

## 1. Dependencies


## 2. Prepration
### 2.1. Directories
### 2.2. Data
### 2.3. Weights

## 3. Training


## 4. Results

## 5. Contact
For any questions, suggestions, or issues with the code, please contact Mohamad at <a>mshahbazi@vision.ee.ethz.ch</a>.

## 6. How to cite:





