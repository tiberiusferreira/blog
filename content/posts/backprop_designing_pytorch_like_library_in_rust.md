---
title: "Backprop: a Pytorch Like Library in Rust"
date: 2020-02-10T15:21:03-03:00
draft: true
---

# Dipping my toes into the existing Deep Learning Ecosystem

After experimenting with [Yolo](https://pjreddie.com/darknet/yolo/) for an 
[university project](/posts/machine_learning_rust_simd_i) I got really 
curious and excited about the whole computer vision field, and since most of it now
based on deep convolutional networks I started reading up on them. 

This quickly lead me to two fields: Tensorflow and Pytorch, both expect to be used
using the Python API. I followed some tutorials from Pytorch's official website and
took some classes from [Fast.ai course](https://www.fast.ai). 
 
The classes were great and it is unbelievable how much power is put in the hands of 
Python programmers without them needing to know much about how it all works. 
Need to make the code run on the GPU? Just call a ```.cuda()``` or ```.to(device=cuda)```
is it almost as magical as the recent achievements of deep learning themselves. 

The trade-off, of course, is having to write Python 
(or Swift, if TensorFlow for Swift takes off) and losing the ability to control-click
a function to see what it does since eventually you hit the Python/C++ interface, 
the famous [Two language problem](https://www.quora.com/What-is-the-2-language-problem-in-data-science)
 (or three if you count CUDA).
 
 
Crossing this barrier is a challenge since the C++ part is normally distributed as a binary, but
even if I had the source code crossing this barrier is a challenge, both for my IDE and me (shifting the
mental model from one language to the other).

I have also grown very accustomed to the my IDE and compiler working together providing both
great autocomplete and catching most of my mistakes before or during compilation. 

I can see why most data science people are in Python and why it makes sense to build the
deep learning ecosystem around them. Once you have chosen Python it is really hard to avoid
the two language problem.  


