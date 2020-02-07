---
title: "Machine Learning, Rust and SIMD - I"
date: 2019-10-28T03:00:04-03:00
draft: false
---

In the [last post](/posts/botejao_an_excuse_to_write_rust) about 
[Botejão](https://github.com/tiberiusferreira/botejao), my Telegram Menu Bot,
 I discussed a feature I would like to add to it:
allow the user to see how many people are in the restaurants line.

I also mentioned that the university itself would only install and allow the 
use of the camera if the processing was done on-device and the images never left it. 
The only output would be how many people are in the line, through an API, for privacy reasons. 

I would really like to get access to an image as output, anonymized of course, to let the users 
check by themselves if the line is too long. It would also help debug any weird API results. 

## Simple Goals

- From an image output how many people are in it
- Output an anonymized image (Could be useful for debugging and to inspect visually how many people are in the actual line)
- Run on a Raspberry Pi 3B (this is the device provided by [Smart Campus](http://smartcampus.prefeitura.unicamp.br))

So I went ahead and took some sample pictures:

<img src="/post_images/p2/p1_day.png" width="1000">

Its 2019, we have (almost) self driving cars, so surely there are many good solutions to person
 recognition in images.
 
Googling "State of the art pedestrian recognition" quickly lead me to [Yolo](https://pjreddie.com/darknet/yolo/)

After compiling, I tested on the sample image and was quite pleased with the results.

<img src="/post_images/p2/p1_det.png" width="1000">

It even worked when it was dark:

<img src="/post_images/p2/p2.png" width="1000">

<img src="/post_images/p2/p2_det.png" width="1000">

Sadly running the lastest Yolo-V3 took almost half a minute 
on my 2017 Macbook Pro CPU and upwards of 1.6GB of RAM. 
While the CPU time would be much worse on the Raspberry, it would not 
run at all because the model 3B only has 1GB of RAM.

Normally these models are meant to be run on a Nvidia GPU (CUDA) or 
specialized hardware such as [Googles TPU](https://coral.withgoogle.com/products/accelerator/). 

Google's USB TPU accelerator and a Tensor Flow version of Yolo, [DarkFlow](https://github.com/thtrieu/darkflow.git), would be 
one solution. However, there is no budget for it.  Also there is no need for real time detection, one update every few 
 minutes is fine, so I might be able to get away with a CPU implementation.  
 
 The original Yolo implementation is done in pure C code which seems like 
 a good opportunity to learn about Rusts FFI and see how far I can take it with 
 Rusts SIMD and multi-threading support.
 
 There is a lighter version of Yolo: Yolo-Tiny which requires much less RAM and is much faster, 
 however it failed to detect some people in the images which would make anonymizing them much harder, so it was a non-starter.
 
 
## Anonymization 

After getting the results shown above, I was pretty confident I could trust Yolo to put bounding boxes
around each person. So it was just a matter of digging into the codebase and changing it to fill the
whole box instead of just drawing a contour. By the tenth _Segmentation Fault_ most of my C memories
had come back and I had some results to show.

 
 <img src="/post_images/p2/filled_boxes.jpg" width="1000">
 

## Getting it to work on a Raspberry

After compiling Yolo and running it for the first time on the Raspberry I was met with a the famous _Segmentation Fault_.

I expected it to run out of memory, but not crash immediately. Switching the configuration and neural net weights to Yolo-tiny
gave the same result. Eventually I was lead to a github issue:  [YOLO: segmentation fault on RaspberryPi](https://github.com/pjreddie/darknet/issues/823)
and switched to another a Yolo fork: https://github.com/AlexeyAB/darknet and ported my changes.

By changing YoloV3's height and width configuration to 384 instead of the original 608 I managed to run it
within the 1GB of RAM of the Raspberry while still detecting every person. It took around *5 minutes* to process one image.

```
real	5m36.687s
user	5m12.121s
sys	0m1.680s
```

Enabling OpenMP dropped it to less than 2m which is quite a speedup.

```
C ARM OpenMP
real	1m44.369s
user	5m26.866s
sys	0m1.541s
```

On part II I will explore my attempts at identifying and optimizing the hot parts of the C code. 





