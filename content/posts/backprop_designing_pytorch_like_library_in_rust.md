---
title: "Current Deep Learning Ecosystem from a Rust Developer perspective"
date: 2020-02-10T15:21:03-03:00
draft: false
---

## How I got involved
After experimenting with [Yolo](https://pjreddie.com/darknet/yolo/) for an 
[university project](/posts/machine_learning_rust_simd_i) I got really 
curious and excited about the whole computer vision field, and since most of it now
based on deep convolutional networks I started reading up on them. 

This quickly lead me to two fields: Tensorflow and Pytorch, both built to be used
through the Python API. I followed some tutorials from Pytorch's official website and
took some classes from [Fast.ai course](https://www.fast.ai). 
 
## The good

The classes were great and it is unbelievable how much power is put in the hands of 
Python programmers without them needing to know much about how it all works. 

Using Tensors as "regular" variables and populating gradients with a simple ```.backward()```
is fantastic. 
Need to make the code run on the GPU? Just call a ```.cuda()``` or ```.to(device=cuda)```
is it almost as magical as the recent achievements of deep learning themselves. 

## The bad

The trade-off, of course, is having to write Python 
(or Swift, if TensorFlow for Swift takes off) and losing the ability to control-click
a function to see what it does since eventually you hit the Python/C++ interface, 
the famous [Two language problem](https://www.quora.com/What-is-the-2-language-problem-in-data-science)
 (or three if you count CUDA).
 
 
Crossing this barrier is a challenge since the C++ part is normally distributed as a binary, but
even if one downloaded the source code, crossing this barrier is a challenge, both for my IDE and me (shifting the
mental model from one language to the other).

I have also grown very accustomed to the my IDE and compiler working together providing both
great autocomplete and catching most of my mistakes before or during compilation. 

I can see why most data science people are in Python and why it makes sense to build the
deep learning ecosystem around them, but it is frustrating coming from Rust.

## The future

These problems are known and used as motivation for creating Swift for Tensorflow during
the last two classes of [Fast.ai course](https://www.fast.ai). There, Swift is pitched
as a language as fast as C  *(that sounds familiar...)* with automatic differentiation 
baked in the language, and most importantly,
["infinitely hackable"](https://forums.swift.org/t/what-makes-swift-infinitely-hackable/28531)
 which boils down to being able to leverage accelerators directly through Swift, using
[MLIR's](https://mlir.llvm.org) infrastructure and some meta-programing 
(think CUDA-like DSL in Swift, but not NVIDIA specific).




This got me curious, what would would a Rust Deep Learning Library look like? 


## What would it take to build an "usable" DL library in Rust?

> Warning: This is my take on the subject, which could be wildly wrong and naive.


I know that is not *impossible*, since [Yolo](https://github.com/pjreddie/darknet) uses its own
custom library written in C + CUDA. 

### A good Tensor library

Well, a Tensor library which can run on the CPU and GPU is a must. On the CPU front there is
[NDarray](https://github.com/rust-ndarray/ndarray) which [doesn't plan](https://github.com/rust-ndarray/ndarray/issues/89#issuecomment-284195600) on supporting GPU
anytime soon. 

Here I think the long term plan is going the same route as Swift, using MLIR (and ultimately the compiler) 
to figure out the best way to execute a given computation on the hardware at hand. But while that
is not yet feasible using [Arrayfire](https://github.com/arrayfire/arrayfire-rust) or [RustaCUDA](https://github.com/bheisler/RustaCUDA)
should allow for an usable library. 

For reference, Pytorch implements CUDA kernels directly, but I think it should be avoided as much
as possible since it is a lot of work, requires quite a lot of knowledge to do properly and leads to the *Two language problem*.

Since a great deal of DL computations are reduced to matrix multiplication and accumulation, we could
implement only these essential pieces using custom kernels hopefully getting 80% of the results for 20% of the work.

 
### Automatic Differentiation
 
Something like Pytorch's [Autograd](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html) system.

This might be a challenge given Rust ownership rules. There are some libraries in this area, such as
[Wyrm](https://github.com/maciejkula/wyrm) and [Rust-Autograd](https://github.com/raskr/rust-autograd).

The former uses lazily evaluated graphs which helps with the ownership problems and the later
uses the Rc<<RefCell<<T>>>> pattern.

I've been [exploring alternatives](https://github.com/tiberiusferreira/Autograd-Experiments), but I don't have anything
substantial to offer yet. 


### What if I want something *now*?

I've used [tch-rs](https://github.com/LaurentMazare/tch-rs) which a fantastic crate providing bindings from Pytorch (C++)
 to Rust, which pretty much means all the power and most of the ergonomics of Pytorch, but from Rust!
 
 
 However, it does not come without it's own problems, such as possible 
 [Undefined Behavior](https://github.com/LaurentMazare/tch-rs/issues/33)
  and a [Dangling Pointer](https://github.com/LaurentMazare/tch-rs/issues/152#issuecomment-583101903)
  every once in a while. The maintainer is very active, so those problems are not so bad in practice.
