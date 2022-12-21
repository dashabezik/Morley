# Morley

Morley is open-sourse software for plants morphometry: measuring sprouts, roots length, plants area.

# Overview

Morley is a software tool for measuring morphological parameters of plants using plants photo,
which is noninvasive for plants. Morley calculates sprout length, length of the longest root, total length of the roots
and plant area, performs statistical analysis and produces plots: bar plots of the parameters of different groups,
their comparison in a heatmap and histogramms with distribution of the parameters in different groups.

You can use Morley through the GUI and CLI interfaces by installing the package, or learn the algorithm deeper using the
Jupyter Notebook placed [here](https://github.com/dashabezik/plants).

# Useful links

Read the following information before the **first usage**.
* [Python and Morley installation.](https://github.com/dashabezik/Morley/blob/main/doc/installation.md)
* [Protocol for image acquisition. Rules to make plant photos suitable for Morley.](https://github.com/dashabezik/Morley/blob/main/doc/appendix_a.md)
* [Extended user guide with explanation of Morley parameters: rotation angle, blurring settings, color ranges.](https://github.com/dashabezik/Morley/blob/main/doc/appendix_b.md)


# Installation

Morley requires Python 3.9 or newer. Read how to install Python
[here](https://github.com/dashabezik/Morley/blob/main/doc/installation.md#Python-installation).
You can install Morley GUI from PyPI:
```
pip install morley
```
\* *use this command in the command line (for example, in [Anaconda Prompt](https://github.com/dashabezik/Morley/blob/main/doc/installation.md#Morley-installation))*

Alternatively, you can install Morley directly from GitHub:

```
pip install git+https://github.com/dashabezik/Morley.git
```

# Example

Run `morley` command on the command line (or in
[Anaconda Prompt](https://github.com/dashabezik/Morley/blob/main/doc/installation.md#Launch)),
follow the instructions in the User Guide (more detailed instructions are in
[the extended user guide](https://github.com/dashabezik/Morley/blob/main/doc/appendix_b.md)).
For quickstart you can use [example input files](photos.zip).
Bigger raw photos datasets, which you can use an advanced example, are placed [here](https://github.com/dashabezik/plants).

![terminal animation](doc/morley_launch.gif)

Check the results in the chosen output folder. An example of the output is available [here](report.zip),
results for bigger datasets are published [here](https://github.com/dashabezik/plants).

# User Guide
### Input files

As input, the program takes photographs of plants placed in a row on a plain contrasting background,
and an example of a seed photo.
Morley supports the most popular image files formats (JPEG, PNG, etc., listed
[here](https://docs.opencv.org/3.4/d4/da8/group__imgcodecs.html#ga288b8b3da0892bd651fce07b3bbd3a56)).


All the photos should be equally oriented (all sprouts on one side, all roots on the other).
To convert pixels into millimeters you should locate a size standard near the plants **on the spouts side**.
Read more about how to prepare photos in the
[protocol for image acquisition](doc/appendix_a.md).

The folder with photos must have certain structure: head folder contains subfolders with raw photos by groups.
Read more [here](doc/appendix_b.md#before-running-morley)

<p align="center">
<img src="https://github.com/dashabezik/Morley/blob/main/doc/folder_tree_wo_template.png" width=50% height=50% title = "Folders structure." >

  Folder structure.

</p>


### Search parameters

Before starting the analysis, you must set:
* input directory
* seed template file
* output directory
* angle of photo rotation<sup>1</sup>;
* bluring parameters<sup>2</sup>;
* sprouts and roots color ranges<sup>3</sup>;
* seeds color range<sup>3</sup>;
* area of the size standerd, $mm^2$;
* germination threshold, mm.


>1) To choose the needed angle rotate the photo to match the example: sprouts on the left and roots on the right.
See the GIF above or check the [extended user guide](https://github.com/dashabezik/Morley/blob/main/doc/appendix_b.md#Rotation).

>2) Bluring parameters are 'morph', 'gauss', 'canny_top'. Example values of these parameters used on real data
are listed in [**Table 1**](https://github.com/dashabezik/plants).
Check the [extended user guide](doc/appendix_b.md#Contours-recognition) on how to choose the settings for your data.
Your goal is to find the values of the parameters such that plant contours are detected accurately.
Initial values of the parameters are set, you should just fix them a little bit if needed.

>3) Color ranges for sprouts and roots should be chosen as light-green and light-violet colors in HSV system.
Color range for seeds should be chosen as their natural color (light-yellow for wheat and peas in HSV system).
How to set them read in the [user guide](doc/appendix_b.md#Color-ranges).
Initial values of the parameters are set, you should just fix them a little bit if needed.
