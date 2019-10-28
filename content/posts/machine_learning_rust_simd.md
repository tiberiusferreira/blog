---
title: "Machine Learning, Rust and Simd"
date: 2019-10-28T03:00:04-03:00
draft: true
---

In the [last post](/posts/botejao_a_motivation_to_write_rust) about 
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
- Output an anonymized image (Could be useful for debugging and if done)
- Run on a Raspberry Pi 3B 

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
specialized hardware such as [Googles TPU](https://coral.withgoogle.com/products/accelerator/) .   






