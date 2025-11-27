# SP4CD
SP4CD: A Hierarchical Stripe Patch-Based Method for Change Detection in Remote Sensing Images 

## Overview
![image](https://github.com/ZehuaChenLab/SP4CD/blob/main/images/images/overview.png)

## Motivation
![image](https://github.com/ZehuaChenLab/SP4CD/blob/main/images/images/motivation.png)
Motivation for our method. We split the images in the training set into striped panes, count the number of objects in the original image, and count the number of objects that remain in only one pane (fully retained) after the split, in order to calculate the fraction of fully retained objects. There are two slice ways: in the height and width direction. (a) shows the the fraction of fully retained objects when the four datasets involved in the experiment are split in two directions. (b) provides an example of slice in the height direction on the WHU-CD dataset.

## Pseudocode
![image](https://github.com/ZehuaChenLab/SP4CD/blob/main/images/images/code1.png)
![image](https://github.com/ZehuaChenLab/SP4CD/blob/main/images/images/code2.png)

## PICI
![image](https://github.com/ZehuaChenLab/SP4CD/blob/main/images/images/PICI.png)

## ISPS
![image](https://github.com/ZehuaChenLab/SP4CD/blob/main/images/images/ISPS.png)

## Results
![image](https://github.com/ZehuaChenLab/SP4CD/blob/main/images/images/result1.png)
![image](https://github.com/ZehuaChenLab/SP4CD/blob/main/images/images/result2.png)
![image](https://github.com/ZehuaChenLab/SP4CD/blob/main/images/images/com1.png)
![image](https://github.com/ZehuaChenLab/SP4CD/blob/main/images/images/com2.png)
![image](https://github.com/ZehuaChenLab/SP4CD/blob/main/images/images/com3.png)
![image](https://github.com/ZehuaChenLab/SP4CD/blob/main/images/images/com4.png)

## Heatmap
![image](https://github.com/ZehuaChenLab/SP4CD/blob/main/images/images/heatmap.png)

