# Expanded Morley User Guide

Table of contents
=================

<!--ts-->
   * [Before running morley](#Before-running-morley)
   * [GUI usage](#GUI-usage)
      * [Launch](#Launch)
      * [Parameters setting](#Parameters-setting)
        * [Rotation](#Rotation)
        * [Contours recognition](#Contours-recognition)
        * [Color ranges](#Color-ranges)
   * [CLI usage](#CLI-usage)
   * [Output](#Output)

<!--te-->



## Before running morley
Prepare a directory with raw photos. Folders with photos must have a certain structure: head folder contains subfolders named after the plant groups to be 
compared with raw photos. Head folder must be selected as the 
“Image directory”. Head folder will be used as the beggining of the report files name (for example HeadFolderName_MeasuringParameter_Date.csv --> 4567days_seed_germ_2022-09-21.csv). Names of subfolders will be used as group labels in visualisation part. A seed photo contains the seed only and is created as a fragment of plant image.

<p align="center">
<img src="folder_tree_wo_template.png" width=50% height=50%>
</p>


 <p>
   <img src="template.JPG" width=10% align="left" />
 </p>
 
 
<!-- 
<p>
  <figure><img src="template.jpg" align="left"/><figcaption>caption</figcaption></figure>
<!--   <figure><img src="img2.jpg" /><figcaption>caption2</figcaption></figure> -->
<!-- </p> -->

<!-- <figure class="class1" align="left"><img src="template.JPG" title="title" align="left"/><figcaption align="left">caption</figcaption></figure> -->


Ensure the seed image has the same resolution,
not higher than the original photo, do not use programs that can increase the resolution (for example, Photoshop ), just cut a seed from any image using 
a simple image redactor (for example, Paint).  We recommend you to choose the seed that will not be covered by the roots of the plant or turned in a strange way (see an example on the left).
Cut the image as close to the edges of the seed as possible.


## GUI usage:

### Launch
 Run ``` morley``` command on the command line (or in Anaconda Prompt). 
<p align="center">
<img src="load button.PNG" width=50% height=50%>
</p>

*For quick start, download the example photos folder from https://github.com/dashabezik/Morley/tree/main/ or select your own photos. To test the program you can run it using bigger photo sets placed here: https://github.com/dashabezik/plants*

### Parameters setting
&emsp; Select directory with folders containing raw photos. Remember rules from the [notion](#Before-running-morley) above.

&emsp; Select file with seed template.

&emsp; Select output directory.

&emsp; Set paper sticker size in $mm^2$, value =  width (mm) x length (mm), and germination threshold in mm (seedlings with both sprout and root lengths below that threshold will be counted together with non-germinated seeds). For example dataset, use 6241 for paper size in $mm^2$ and germination threshold you prefer. 
germination threshold is a parameter for evaluating germination rate. Plants with sprout and roots lengths below the threshold value (simultaneously) will be considered as non-germinated seeds.

#### Rotation
#
&emsp; Rotate images by clicking the “Rotate image” button. Select the angle so that the location of the objects and the sprout-root orientation correspond to these characteristics in the schematic image on the left.

<p align="center">
<img src="rotation.PNG" width=50% height=50%>
</p>



  >**NOTION:** For correct processing, the paper sticker should be the most left contour, the seeds must compile a vertical line in the center, and the leaves and the roots must be on the left and right from that vertical line, respectively. All the original photos should have the same orientation of sprouts and roots.

After setting the rotation angle, all the photos will be properly rotated, including the seed template image.


#### Contours recognition
#
&emsp; Push "Recognition settings" button to set parameters for plant, root, sprout and seed recognition. Initial parameters, that on average should be suitable for any dataset, are set by default. 

 On this step your goal is to find the values of the parameters to reach covering plants with contours and avoid their merging. Initial values of the parameters are setted, you should just fix them a little bit if it will be needed. See the picture below to understand possible problems*. 

<p align="center">
<img src="bluring_modified.png" width=70% height=70%>
</p>

\* *The parameter values for these pictures are chosen to be extreme. When choosing options, the appearance of the contours will change less contrast. You  can see similar patterns if the contour detection parameters are not suitable for your data. The default parameters that are now in the program are approximate parameters that approximately fit all the photosets we used.*


  >**NOTION:**
  >What are the blurring parameters?
  >
  >‘morph’ is a size of structuring element for morphological transformation, 
  >
  >‘gauss’ is the parameter of gaussian blurring, 
  >
  >‘canny_top’ is the threshold for contours’ identification: any edges with intensity gradient more than ‘canny_top’ are detected as edges* 


Move the trackers to achieve the best recognition of whole plant contour:

<p align="center">
<img src="2.PNG" width=50% height=50%>
</p>

#### Color ranges
#

&emsp; Color range parameters. In the search we use the HSV color coding. The window displays 6 trackers: lower and upper bounds for each of the 3 encoding components (h, s, v). The result of the selection will be the color range of pixels that correspond to the object that we want to highlight in the picture. The window also shows the binary mask of the photo: white pixels are shown that fall into the selected range, black - pixels that do not fall into the range. Your task at this stage is to choose 2 ranges (for sprouts and for roots) that will successfully display the desired objects. 
  
 <p>
   <img src="gl1_p1.jpg" width=30% align="left" />
 </p>
  
For a clearer separation of roots from sprouts, during the search we color the image with a block type: green block for sprouts and pink for roots, so the hue(h - hue ) for roots and sprouts will lie in opposite separated ranges (roots - (125, 165) or wider and seedlings - (20, 55) or wider).

At this stage, the saturation parameter (s - saturation) does not affect anything (so far we have not met such plants or photographs in the course of work), therefore its limits cover the entire range (0.255).

The brightness parameter (v - value, or brightness) selects only light areas to exclude the dark background, so its approximate values range from 100 to 255.

  

 <p> 
  
   <img src="h.png" width=40% />
 </p>
 <p>
   
  <img src="s.png" width=40% />
 </p>
<p>
    <img src="v.png" width=40% />
 </p>


<br clear="left"/>

 <p>
   <img src="hsv.PNG" width=20% align="left" />
 </p>
In the first step, as soon as you get to this tab, the default values for the color components of the roots are displayed. Customize them or leave them as they are and click the "Set roots" button on the right. Next, you need to choose a color range for the sprouts. To do this, move the hue sliders to a range of yellow-green hues (for example, from 0 to 60). At this point, the exact numbers are not so important, because the shades are spaced in a range of hue in non-overlapping areas, so you can easily take a wider range, focusing only on the picture you see.



<br clear="left"/>

<p align="center">
<img src="2tab.png" width=50% height=50%>
</p>

&emsp; The next step of seed segmentation is quite similar to the previous one. Here your goal is to find the color range for the seeds. The window displays the same trackers and a binary mask for an uncolored photo (without any filters). The difference is that you should choose the range for natural seed color. The default parameters are selected for yellow seed (seeds of wheat and peas, that were used are yellow). 
>Hue has only a yellow range (0, 20). The top value is 20 to exclude green pixels of sprouts.
>Saturation scale has a saturated range (100, 255) to exclude white-close unsaturated pixels of roots and sprouts.
>Brightness scale has a light range (100, 255) to exclude a dark background and in some cases you can increase the bottom board to exclude some roots and sprouts areas.

<p align="center">
<img src="3tab.png" width=50% height=50%>
</p>

_____________________ 

&emsp; Press the ‘RUN’ button to start processing. Program has accomplished evaluation when progress bar shows 100% and logging window will notify you when the search is over.

## CLI usage

&emsp; You can run Morley as a command-line interface. To run Morley CLI run ```morley``` commnd with arguments. Use config file obtained from GUI. 

```
morley C:\Users\dasha\plants\set1.json
```

&emsp; You can add some parameters, for example to use one configuration file for different datasets use additional ```-i, -t ```and ```-o ```parametersб so you can change the parameters specified in the file. For example, use the same color and blur settings from the file, but write different paths in the command line.

```
morley path\to\configuration\file\set1.json -i path\to\the\first\dataset

morley path\to\configuration\file\set1.json -i path\to\the\second\dataset
```


    morley [-h] [-i INPUT] [-t TEMPLATE] [-o OUTPUT_DIR] [-r {0,90,180,270}] [-a PAPER_AREA] [-g THRESHOLD] config

    Morley CLI: run a Morley analysis in headless mode.

    positional arguments:
      config                A JSON file with settings. You can obtain one by saving settings from GUI mode. Other
                            arguments override the values in config.

    optional arguments:
      -h, --help            show this help message and exit
      -i INPUT, --input INPUT
                            Input directory.
      -t TEMPLATE, --template TEMPLATE
                            Template file.
      -o OUTPUT_DIR, --output-dir OUTPUT_DIR
                            Output directory.
      -r {0,90,180,270}, --rotation {0,90,180,270}
                            Input photo rotation.
      -a PAPER_AREA, --paper-area PAPER_AREA
                            Paper area, mm^2.
      -g THRESHOLD, --threshold THRESHOLD
                            Germination threshold, mm.






## Output
 &emsp; The output files can be found  in the output directory.  Program generates the following files:

  - the .csv tables with p-values corresponding to all pairwise comparisons between sample groups, the calculated germination efficiency, sprout and root lengths, total plant areas and the summary table with all digital measurements.

 - the figures characterizing distributions of measured plant sizes, bar plots with mean values and standard deviations, and heatmaps visualizing the conclusions on statistical significance of the morphometric changes.
