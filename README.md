# Morley

Morphometry of germinated plants in Python

Morley is an open-sourse software for plants morphometry: measuring sprouts, roots length, plants area.

# Overview

Morley is a software for measuring morphological parameters of plants using plants photo, which is noninvasive for plants. Morley calculates sprouths length, length of the longest root, summ of the length all the roots and plant area, represents statistical analysis with plots: bar plots of the parameters of different groups, their comparison on heatmap and histogramms with distribution of the parameters in different groups.

You and use GUI, that is plased here, or learn the algorithm deeper using Jupyter Notebook placed [here](https://github.com/dashabezik/plants)

# Installation

Morley requires Python 3.9 or newer. Read how to install Python [here](https://github.com/dashabezik/Morley/blob/main/doc/installation.md#Python-installation). You can install Morley GUI from PyPI:
```
pip install morley
```
\* *use this command in a Python compiller or in Bash command line (for example, in [Anaconda Prompt](https://github.com/dashabezik/Morley/blob/main/doc/installation.md#Morley-installation))*

Alternatively, you can install morley directly from GitHub:

```
pip install git+https://github.com/dashabezik/Morley.git
```

# Example

Run ``` morley ``` command on the command line (or in [Anaconda Prompt](https://github.com/dashabezik/Morley/blob/main/doc/installation.md#Launch)), follow the instructions in the User Guide (more detailed instruction is in [Appendix B](https://github.com/dashabezik/Morley/blob/main/doc/appendix_b.md)). For quickstart you can use [example file](https://github.com/dashabezik/Morley/blob/main/photos.rar). Bigger raw photos dataset, which you can use an advanced example, is placed [here](https://github.com/dashabezik/plants).

![terminal animation](doc/morley_launch.gif)

Check the results in the chosen output folder. To compare the recieved results you can look through the results recieved by myself: for quickstart [here](https://github.com/dashabezik/Morley/blob/main/report.rar), for bigger datasets [here](https://github.com/dashabezik/plants).

# User Guide
### Input files

As input, the program takes photographs of plants placed in a row on a plain contrasting background and an example of a seed photo.
Morley supports the most popular image files formats (JPEG files, Portable Network Graphics, et al., full list is [here](https://docs.opencv.org/3.4/d4/da8/group__imgcodecs.html#ga288b8b3da0892bd651fce07b3bbd3a56) ).


All the photos should be equally oriented (sprouts->roots line should have the same direction).
To convert pixels into millimeters you should locate a size standard near the plants on the spoughts side. Read more about how to prepare photos in the [Appendix A](https://github.com/dashabezik/Morley/blob/main/doc/appendix_a.md).

Folder with photos must have certain structure: head folder contains subfolders with raw photos divided by the groups. Read more [here](https://github.com/dashabezik/Morley/blob/main/doc/appendix_b.md#before-running-morley)

<p align="center">
<img src="https://github.com/dashabezik/Morley/blob/main/doc/folder_tree_wo_template.png" width=50% height=50% title = "Folders structure." >

  **Figure 1** Folders structure.

</p>




### Search parameters

Before the measurement start choose:
* input directory
* seed template file
* output directory
* angle of photo rotation<sup>1</sup>;
* bluring parameters<sup>2</sup>;
* sprouts and roots color ranges<sup>3</sup>;
* seeds color range<sup>3</sup>;
* area of the size standerd, $mm^2$;
* germination threshold, mm.


>1) To choose the needed angle rotate the photo up to the example condition: sptoughts are on the left and roots are on the right. See the gif above or read the 5th point of [user guide](https://github.com/dashabezik/Morley/blob/main/doc/appendix_b.md#Rotation)

>2) Bluring parameters are 'morph', 'gauss', 'canny_top'. Examples of these parameters you can look in the [**Table 1**](https://github.com/dashabezik/plants), how to set them read in the [user guide](https://github.com/dashabezik/Morley/blob/main/doc/appendix_b.md#Contours-recognition). On this step your goal is to find the values of the parameters to reach covering plants with contours and avoid their merging. Initial values of the parameters are setted, you should just fix them a little bit if it will be needed.



>3) Color ranges for sprouts and roots should be chosen as light-green and light-violet colors in HSV system. Color range for seeds should be chosen as their natural color (light-yellow for wheat and peas in HSV system). How to set them read in the [user guide](https://github.com/dashabezik/Morley/blob/main/doc/appendix_b.md#Color-ranges). Initial values of the parameters are setted, you should just fix them a little bit if it will be needed.













