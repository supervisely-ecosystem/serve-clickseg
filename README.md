<div align="center" markdown>
<img src="xxx"/>  

# Interactive Segmentation with ClickSEG
  
<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#Pretrained-models">Pretrained models</a> •
  <a href="#How-to-run">How to run</a> •
  <a href="#Controls">Controls</a> •
  <a href="#Acknowledgment">Acknowledgment</a> 
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/serve-clickseg)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervise.ly/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/serve-clickseg)
[![views](https://app.supervise.ly/img/badges/views/supervisely-ecosystem/serve-clickseg.png)](https://supervise.ly)
[![runs](https://app.supervise.ly/img/badges/runs/supervisely-ecosystem/serve-clickseg.png)](https://supervise.ly)
 
</div>

## Overview

Application key points:  
- Manually selected ROI
- Deploy on GPU(faster) or CPU(slower)
- Accurate predictions in most cases
- Correct prediction interactively with `red` and `green` clicks
- Select from [14 pretrained models](../README.md#Pretrained-models)
- Models are class agnostic, you can segment any object from any domain

ClickSeg Interactive segmentation algorithms allow users to explicitly control the predictions using interactive input at several iterations, in contrast to common semantic and instance segmentation algorithms that can only input an image and output a segmentation mask in one pass. Such interaction makes it possible to select an object of interest and correct prediction errors.

<img src="gif"/>

Besides segmenting new objects, proposed method allows to correct external masks, e.g. produced by other
instance or semantic segmentation models. A user can fix false negative and false positive regions with positive (green)
and negative (red) clicks, respectively.

## Pretrained models

### CDNet: Conditional Diffusion for Interative Segmentation (ICCV2021)
**config:**
```
Input Size: 384 x 384
Previous Mask: No
Iterative Training: No
```
<table>
    <thead align="center">
        <tr>
            <th rowspan="2"><span style="font-weight:bold">Train</span><br><span style="font-weight:bold">Dataset</span></th>
            <th rowspan="2">Model</th>
            <th>GrabCut</th>
            <th>Berkeley</th>
            <th>Pascal<br>VOC</th>
            <th>COCO<br>MVal</th>
            <th>SBD</th>    
            <th>DAVIS</th>
            <th>DAVIS585<br>from zero</th>
            <th>DAVIS585<br>from init</th>
        </tr>
        <tr>
            <td>NoC<br>85/90%</td>
            <td>NoC<br>85/90%</td>
            <td>NoC<br>85/90%</td>
            <td>NoC<br>85/90%</td>
            <td>NoC<br>85/90%</td>
            <td>NoC<br>85/90%</td>
            <td>NoC<br>85/90%</td>
            <td>NoC<br>85/90%</td>
        </tr>
    </thead>
    <tbody align="center">
        <tr>
            <td rowspan="1">SBD</td>
            <td align="center"><a href="https://drive.google.com/drive/folders/1XzUlpPqbzAyMt009HVpEeW31ln1FeXfX?usp=sharing">ResNet34<br>(89.72 MB)</a></td>
            <td>1.86/2.18</td>
            <td>1.95/3.27</td>
            <td>3.61/4.51</td>
            <td>4.13/5.88</td>
            <td>5.18/7.89</td>
            <td>5.00/6.89</td>
            <td>6.68/9.59</td>
            <td>5.04/7.06</td>
        </tr>
    </tbody>
    <tbody align="center">
        <tr>
            <td rowspan="1">COCO+<br>LVIS</td>
            <td align="center"><a href="https://drive.google.com/drive/folders/1XzUlpPqbzAyMt009HVpEeW31ln1FeXfX?usp=sharing">ResNet34<br>(89.72 MB)</a></td>
            <td>1.40/1.52</td>
            <td>1.47/2.06</td>
            <td>2.74/3.30</td>
            <td>2.51/3.88</td>
            <td>4.30/7.04</td>
            <td>4.27/5.56</td>
            <td>4.86/7.37</td>
            <td>4.21/5.92</td>
        </tr>
    </tbody>
</table>


### FocalClick: Towards Practical Interactive Image Segmentation (CVPR2022)
**config:**
```
S1 version: coarse segmentator input size 128x128; refiner input size 256x256.  
S2 version: coarse segmentator input size 256x256; refiner input size 256x256.  
Previous Mask: Yes
Iterative Training: Yes
```
<table>
    <thead align="center">
        <tr>
            <th rowspan="2"><span style="font-weight:bold">Train</span><br><span style="font-weight:bold">Dataset</span></th>
            <th rowspan="2">Model</th>
            <th>GrabCut</th>
            <th>Berkeley</th>
            <th>Pascal<br>VOC</th>
            <th>COCO<br>MVal</th>
            <th>SBD</th>    
            <th>DAVIS</th>
            <th>DAVIS585<br>from zero</th>
            <th>DAVIS585<br>from init</th>
        </tr>
        <tr>
            <td>NoC<br>85/90%</td>
            <td>NoC<br>85/90%</td>
            <td>NoC<br>85/90%</td>
            <td>NoC<br>85/90%</td>
            <td>NoC<br>85/90%</td>
            <td>NoC<br>85/90%</td>
            <td>NoC<br>85/90%</td>
            <td>NoC<br>85/90%</td>
        </tr>
    </thead>
        <tbody align="center">
        <tr>
            <td rowspan="1">COCO+<br>LVIS</td>
            <td align="center"><a href="https://drive.google.com/drive/folders/1XzUlpPqbzAyMt009HVpEeW31ln1FeXfX?usp=sharing">HRNet18s-S1<br>(16.58 MB)</a></td>
            <td>1.64/1.88</td>
            <td>1.84/2.89</td>
            <td>3.24/3.91</td>
            <td>2.89/4.00</td>
            <td>4.74/7.29</td>
            <td>4.77/6.56</td>
            <td>5.62/8.08</td>
            <td>2.72/3.82</td>
        </tr>
    </tbody>
     <tbody align="center">
        <tr>
            <td rowspan="1">COCO+<br>LVIS</td>
            <td align="center"><a href="https://drive.google.com/drive/folders/1XzUlpPqbzAyMt009HVpEeW31ln1FeXfX?usp=sharing">HRNet18s-S2<br>(16.58 MB)</a></td>
            <td>1.48/1.62</td>
            <td>1.60/2.23</td>
            <td>2.93/3.46</td>
            <td>2.61/3.59</td>
            <td>4.43/6.79</td>
            <td>3.90/5.23</td>
            <td>4.87/6.87</td>
            <td>2.47/3.30</td>
        </tr>
    </tbody>
    <tbody align="center">
        <tr>
            <td rowspan="1">COCO+<br>LVIS</td>
            <td align="center"><a href="https://drive.google.com/drive/folders/1XzUlpPqbzAyMt009HVpEeW31ln1FeXfX?usp=sharing">HRNet32-S2<br>(119.11 MB)</a></td>
            <td>1.64/1.80</td>
            <td>1.70/2.36</td>
            <td>2.80/3.35</td>
            <td>2.62/3.65</td>
            <td>4.24/6.61</td>
            <td>4.01/5.39</td>
            <td>4.77/6.84</td>
            <td>2.32/3.09</td>
        </tr>
    </tbody>
         <tbody align="center">
        <tr>
            <td rowspan="1">Combined+<br>Dataset</td>
            <td align="center"><a href="https://drive.google.com/drive/folders/1XzUlpPqbzAyMt009HVpEeW31ln1FeXfX?usp=sharing">HRNet32-S2<br>(119.11 MB)</a></td>
            <td>1.30/1.34</td>
            <td>1.49/1.85</td>
            <td>2.84/3.38</td>
            <td>2.80/3.85</td>
            <td>4.35/6.61</td>
            <td>3.19/4.81</td>
            <td>4.80/6.63</td>
            <td>2.37/3.26</td>
        </tr>
    </tbody>
    <tbody align="center">
        <tr>
            <td rowspan="1">COCO+<br>LVIS</td>
            <td align="center"><a href="https://drive.google.com/drive/folders/1XzUlpPqbzAyMt009HVpEeW31ln1FeXfX?usp=sharing">SegFormerB0-S1<br>(14.38 MB)</a></td>
            <td>1.60/1.86</td>
            <td>2.05/3.29</td>
            <td>3.54/4.22</td>
            <td>3.08/4.21</td>
            <td>4.98/7.60</td>
            <td>5.13/7.42</td>
            <td>6.21/9.06</td>
            <td>2.63/3.69</td>
        </tr>
    </tbody>
    <tbody align="center">
        <tr>
            <td rowspan="1">COCO+<br>LVIS</td>
            <td align="center"><a href="https://drive.google.com/drive/folders/1XzUlpPqbzAyMt009HVpEeW31ln1FeXfX?usp=sharing">SegFormerB0-S2<br>(14.38 MB)</a></td>
            <td>1.40/1.66</td>
            <td>1.59/2.27</td>
            <td>2.97/3.52</td>
            <td>2.65/3.59</td>
            <td>4.56/6.86</td>
            <td>4.04/5.49</td>
            <td>5.01/7.22</td>
            <td>2.21/3.08</td>
        </tr>
    </tbody>
    <tbody align="center">
        <tr>
            <td rowspan="1">COCO+<br>LVIS</td>
            <td align="center"><a href="https://drive.google.com/drive/folders/1XzUlpPqbzAyMt009HVpEeW31ln1FeXfX?usp=sharing">SegFormerB3-S2<br>(174.56 MB)</a></td>
            <td>1.44/1.50</td>
            <td>1.55/1.92</td>
            <td><b>2.46/2.88</b></td>
            <td><b>2.32/3.12</b></td>
            <td><b>3.53/5.59</b></td>
            <td>3.61/4.90</td>
            <td>4.06/5.89</td>
            <td>2.00/2.76</td>
        </tr>
    </tbody>
    <tbody align="center">
        <tr>
            <td rowspan="1"><b>Combined<br>Datasets</b></td>
            <td align="center"><b><a href="https://drive.google.com/drive/folders/1XzUlpPqbzAyMt009HVpEeW31ln1FeXfX?usp=sharing">SegFormerB3-S2<br>(174.56 MB)</a></b></td>
            <td><b>1.22/1.26</b></td>
            <td><b>1.35/1.48</b></td>
            <td>2.54/2.96</td>
            <td>2.51/3.33</td>
            <td>3.70/5.84</td>
            <td><b>2.92/4.52</b></td>
            <td><b>3.98/5.75</b></td>
            <td><b>1.98/2.72</b></td>
        </tr>
    </tbody>
</table>


### Efficient Baselines using MobileNets and PPLCNets
**config:**
```
Input Size: 384x384.
Previous Mask: Yes
Iterative Training: Yes
```
<table>
    <thead align="center">
        <tr>
            <th rowspan="2"><span style="font-weight:bold">Train</span><br><span style="font-weight:bold">Dataset</span></th>
            <th rowspan="2">Model</th>
            <th>GrabCut</th>
            <th>Berkeley</th>
            <th>Pascal<br>VOC</th>
            <th>COCO<br>MVal</th>
            <th>SBD</th>    
            <th>DAVIS</th>
            <th>DAVIS585<br>from zero</th>
            <th>DAVIS585<br>from init</th>
        </tr>
        <tr>
           <td>NoC<br>85/90%</td>
            <td>NoC<br>85/90%</td>
            <td>NoC<br>85/90%</td>
            <td>NoC<br>85/90%</td>
            <td>NoC<br>85/90%</td>
            <td>NoC<br>85/90%</td>
            <td>NoC<br>85/90%</td>
            <td>NoC<br>85/90%</td>
        </tr>
    </thead>
    <tbody align="center">
        <tr>
            <td rowspan="1">COCO+<br>LVIS</td>
            <td align="center"><a href="https://drive.google.com/drive/folders/1XzUlpPqbzAyMt009HVpEeW31ln1FeXfX?usp=sharing">MobileNetV2<br>(7.5 MB)</a></td>
            <td>1.82/2.02</td>
            <td>1.95/2.69</td>
            <td>2.97/3.61</td>
            <td>2.74/3.73</td>
            <td>4.44/6.75</td>
            <td>3.65/5.81</td>
            <td>5.25/7.28</td>
            <td>2.15/3.04</td>
        </tr>
    </tbody>
        <tbody align="center">
        <tr>
            <td rowspan="1">COCO+<br>LVIS</td>
            <td align="center"><a href="https://drive.google.com/drive/folders/1XzUlpPqbzAyMt009HVpEeW31ln1FeXfX?usp=sharing">PPLCNet<br>(11.92 MB)</a></td>
            <td>1.74/1.92</td>
            <td>1.96/2.66</td>
            <td>2.95/3.51</td>
            <td>2.72/3.75</td>
            <td>4.41/6.66</td>
            <td>4.40/5.78</td>
            <td>5.11/7.28</td>
            <td>2.03/2.90</td>
        </tr>
    </tbody>
    <tbody align="center">
        <tr>
            <td rowspan="1">Combined<br>Datasets</td>
            <td align="center"><a href="https://drive.google.com/drive/folders/1XzUlpPqbzAyMt009HVpEeW31ln1FeXfX?usp=sharing">MobileNetV2<br>(7.5 MB)</a></td>
            <td>1.50/1.62</td>
            <td>1.62/2.25</td>
            <td>3.00/3.61</td>
            <td>2.80/3.96</td>
            <td>4.66/7.05</td>
            <td>3.59/5.24</td>
            <td>5.05/7.12</td>
            <td>2.06/2.97</td>
        </tr>
    </tbody>
        <tbody align="center">
        <tr>
            <td rowspan="1">Combined<br>Datasets</td>
            <td align="center"><a href="https://drive.google.com/drive/folders/1XzUlpPqbzAyMt009HVpEeW31ln1FeXfX?usp=sharing">PPLCNet<br>(11.92 MB)</a></td>
            <td>1.46/1.66</td>
            <td>1.63/1.99</td>
            <td>2.88/3.44</td>
            <td>2.75/3.89</td>
            <td>4.44/6.74</td>
            <td>3.65/5.34</td>
            <td>5.02/6.98</td>
            <td>1.96/2.81</td>
        </tr>
    </tbody>
</table>

## Prediction preview (SegformerB3-S2):

<div align="center" markdown>
 <img src="https://raw.githubusercontent.com/supervisely-ecosystem/serve-clickseg/master/demo_data/prediction.jpg" width="40%"/>
</div>

## How to run

1. Start the application from Ecosystem

<img src="xxx" />

2. Select the pretrained model and deploy it on your device by clicking `Serve` button

<img src="xxx" />

3. You'll see `Model has been successfully loaded` message indicating that the application has been successfully started and you can work with it from now on.
 
<img src="xxx" />

## Controls

| Key                                                           | Description                               |
| ------------------------------------------------------------- | ------------------------------------------|
| <kbd>Left Mouse Button</kbd>                                  | Place a positive click                    |
| <kbd>Shift + Left Mouse Button</kbd>                          | Place a negative click                    |
| <kbd>Scroll Wheel</kbd>                                       | Zoom an image in and out                  |
| <kbd>Right Mouse Button</kbd> + <br> <kbd>Move Mouse</kbd>    | Move an image                             |
| <kbd>Space</kbd>                                              | Finish the current object mask            |
| <kbd>Shift + H</kbd>                                          | Higlight instances with random colors     |
| <kbd>Ctrl + H</kbd>                                           | Hide all labels                           |


<p align="left"> <img align="center" src="https://i.imgur.com/jxySekj.png" width="50"> <b>—</b> Auto add positivie point to rectangle button (<b>ON</b> by default for SmartTool apps) </p>

<div align="center" markdown>
<img src="https://i.imgur.com/dlaLrsi.png" width="90%"/>
</div>

<p align="left"> <img align="center" src="https://i.imgur.com/kiwbBkj.png" width="200"> <b>—</b> SmartTool selector button, switch between SmartTool apps and models</p>

<div align="center" markdown>
<img src="https://i.imgur.com/FATcNZU.png" width="90%"/>
</div>

## Acknowledgment

This app is based on the great work `ClickSEG: A Codebase for Click-Based Interactive Segmentation` [github](https://github.com/XavierCHEN34/ClickSEG). ![GitHub Org's stars](https://img.shields.io/github/stars/XavierCHEN34/ClickSEG?style=social)


