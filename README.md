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
  <li><a href="#dep">Dependencies</a></li>
  <li><a href="#prep">Prepration</a>
    <ol>
      <li><a href="#dir">Directories</a></li>
      <li><a href="#data">Data</a></li>
      <li><a href="#weight">Weights</a></li>
    </ol>
  </li>
  <li><a href="#train">Training</a></li>
  <li><a href="#res">Results</a></li>
  <li><a href="#contact">Contact</a></li>
  <li><a href="#cite">How to cite</a></li>

</ol>

<div id="dep">
</div>

<div id="prep">
</div>

<div id="train">
</div>

<div id="res">
</div>

<div id="contact">
  For any questions, suggestions, or issues with the code, please contact Mohamad at [mshahbazi@vision.ee.ethz.ch](mshahbazi@vision.ee.ethz.ch)
</div>

<div id="cite">
</div>






