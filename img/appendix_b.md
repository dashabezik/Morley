# Expanded Morley User Guide

Table of contents
=================

<!--ts-->
   * [Before running morley](#Before-running-morley)
   * [USAGE](#USAGE)
      * [Launch](#stdin)
      * [Local files](#local-files)
      * [Remote files](#remote-files)
      * [Multiple files](#multiple-files)
      * [Combo](#combo)
      * [Auto insert and update TOC](#auto-insert-and-update-toc)
      * [GitHub token](#github-token)
      * [TOC generation with Github Actions](#toc-generation-with-github-actions)
   * [Tests](#tests)
   * [Dependency](#dependency)
   * [Docker](#docker)
     * [Local](#local)
     * [Public](#public)
<!--te-->



### Before running morley
Prepare a directory with raw photos. Folders with photos must have a certain structure: head folder contains subfolders named after the plant groups to be 
compared with raw photos. Head folder must be selected as the 
“Image directory”. Head folder will be used as the beggining of the report files name (for example HeadFolderName_MeasuringParameter_Date.csv --> 4567days_seed_germ_2022-09-21.csv). Names of subfolders will be used as group labels in visualisation part. A seed photo contains the seed only and is created as a fragment of plant image.

<p align="center">
<img src="https://github.com/dashabezik/Morley/blob/main/img/folder_tree_wo_template.png" width=50% height=50%>
</p>

Ensure the seed image has the same resolution,
not higher than the original photo, do not use programs that can increase the resolution (for example, Photoshop ), just cut a seed from any image using 
a simple image redactor (for example, Paint).  We recommend you to choose the seed that will not be covered by the roots of the plant or turned in a strange way.
Cut the image as close to the edges of the seed as possible.


## USAGE:
### 
0. Run ``` morley``` command on the command line (or in Anaconda Promp). 
<p align="center">
<img src="https://github.com/dashabezik/Morley/blob/main/img/load button.PNG" width=50% height=50%>
</p>

1. For quick start, download the example photos folder from https://github.com/dashabezik/Morley/tree/main/ or select your own photos. To test the program you can run it using bigger photo sets placed here: https://github.com/dashabezik/plants


2. Select directory with folders containing raw photos. Remember rules from the [NOTION](#Before-running-morley) above.

3. Select file with seed template.

4. Select output directory.

5. Rotate images by clicking the “Rotate image” button. Select the angle so that the location of the objects and the sprout-root orientation correspond to these characteristics in the schematic image on the left.

NOTION: For correct processing, the paper sticker should be the most left contour, the seeds must compile a vertical line in the center, and the leaves and the roots must be on the left and right from that vertical line, respectively. All the original photos should have the same orientation of sprouts and roots. 
After setting, all the photos will be properly rotated, including the seed template image.

Tweak image button is for setting parameters for plant, root, sprout and seed recognition. Initial parameters are set by default (see the table with examples for different datasets). 

 On this step your goal is to find the values of the parameters to reach covering plants with contours and avoid their merging. Initial values of the parameters are setted, you should just fix them a little bit if it will be needed. See the picture below to understand possible problems*. 


* The parameter values for these pictures are chosen to be extreme. When choosing options, the appearance of the contours will change less contrast. You  can see similar patterns if the contour detection parameters are not suitable for your data. The default parameters that are now in the program are approximate parameters that approximately fit all the photosets we used
NOTION:
What are the blurring parameters?
‘morph’ is a size of structuring element for morphological transformation, 
‘gauss’ is the parameter of gaussian blurring, 
‘canny_top’ is the threshold for contours’ identification: any edges with intensity gradient more than ‘canny_top’ are detected as edges 
Move the trackers to achieve the best recognition of whole plant contour:


Color range parameters. In the search we use the HSV color coding. The window displays 6 trackers: lower and upper bounds for each of the 3 encoding components (h, s, v). The result of the selection will be the color range of pixels that correspond to the object that we want to highlight in the picture. The window also shows the binary mask of the photo: white pixels are shown that fall into the selected range, black - pixels that do not fall into the range. Your task at this stage is to choose 2 ranges (for sprouts and for roots) that will successfully display the desired objects.
For a clearer separation of roots from sprouts, during the search we color the image with a block type: green block for sprouts and pink for roots, so the hue(h - hue ) for roots and sprouts will lie in opposite separated ranges (roots - (125, 165) or wider and seedlings - (20, 55) or wider).
At this stage, the saturation parameter (s - saturation) does not affect anything (so far we have not met such plants or photographs in the course of work), therefore its limits cover the entire range (0.255).
The brightness parameter (v - value, or brightness) selects only light areas to exclude the dark background, so its approximate values range from 100 to 255.
H 
S  
V 


In the first step, as soon as you get to this tab, the default values for the color components of the roots are displayed. Customize them or leave them as they are and click the "Set roots" button on the right. Next, you need to choose a color range for the sprouts. To do this, move the hue sliders to a range of yellow-green hues (for example, from 0 to 60). At this point, the exact numbers are not so important, because the shades are spaced in a range of hue in non-overlapping areas, so you can easily take a wider range, focusing only on the picture you see.

The next step of seed segmentation is quite similar to the previous one. Here your goal is to find the color range for the seeds. The window displays the same trackers and a binary mask for an uncolored photo (without any filters). The difference is that you should choose the range for natural seed color. The default parameters are selected for yellow seed. 
Hue has only a yellow range (0, 20). The top value is 20 to exclude green pixels of sprouts.
Saturation scale has a saturated range (100, 255) to exclude white-close unsaturated pixels of roots and sprouts.
Brightness scale has a light range (100, 255) to exclude a dark background and in some cases you can increase the bottom board to exclude some roots and sprouts areas.

Set paper sticker size in mm2, value =  width (mm) x length (mm), and germination threshold in mm (seedlings with both sprout and root lengths below that threshold will be counted together with non-germinated seeds). For example dataset, use 6241 for paper size in mm2 and germination threshold you prefer. Press the ‘RUN’ button to start processing.
 
Program has accomplished evaluation when progress bar shows 100% and logging window will notify you when the search is over:

 
The output files can be found  in the output directory.  Program generates the following files:

the .csv tables with p-values corresponding to all pairwise comparisons between sample groups, the calculated germination efficiency, sprout and root lengths, total plant areas and the summary table with all digital measurements.
the figures characterizing distributions of measured plant sizes, bar plots with mean values and standard deviations, and heatmaps visualizing the conclusions on statistical significance of the morphometric changes.