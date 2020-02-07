---
title: "Botejão - An excuse to write Rust"
date: 2019-10-27T17:23:34-03:00
draft: false
---

When learning Rust I started many toy projects. One of them was [Botejão](https://github.com/tiberiusferreira/botejao).

It was never meant to be anything more than a Telegram Bot to check my universities (UNICAMP) [daily menu](https://www.prefeitura.unicamp.br/servicos/divisao-de-alimentacao/cardapio-dos-restaurantes). In the beginning users asked the bot for the menu and then got a reply.

<img src="/post_images/p1/botejao_v1.jpeg" width="300">
 

Later it got a few upgrades and now it sends the day's menu every night to a [channel](https://t.me/botejao_unicamp).

<img src="/post_images/p1/botejao_v2.jpeg" width="300">

However, I always felt like this bot could do more. There are already many good Telegram bots from where one can check 
today's menu such as @BandeconatoBot and @BandejaoBot. So showing the menu is a good start, but not a differentiating factor.

## A differentiating feature

At my university the line to get inside the restaurant can vary drastically depending on the menu, hour of the day and period of the year.

For example, students join the university at the beginning of the year and for the first few months most of them go to the restaurants. 
By the second half of the year, the novelty factor wears off and some of them start cooking at home or eating elsewhere. 

All in all, it is really hard to predict if there will be a line or not given a date and time of the day.

So I went for the simplest solution: the bot should send an image of the current restaurant line so the user can decide to go now or later.

## Getting permission

As one can image, in order to install a camera and start taking photos from the campus I need permission from the university's itself.

So I sent an email to [Smart Campus](http://smartcampus.prefeitura.unicamp.br) the organization inside UNICAMP which seemed the most appropriate. 
We had a meeting and they revealed that other students had the same idea before and that they could provide and install the camera.
 
However, they would only do it if the image never left the device and the only output as how many people were in the line,
for privacy reasons.
 This was the reason previous attempts were not successful. 
 
All I heard was the excuse I needed to dip my toes into Machine Learning.

 
   
