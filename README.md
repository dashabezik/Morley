# Morley

Morphometry of germinated plants in Python

Morley is an open-sourse software for plants morphometry: measuring sprouts, roots length, plants area.

# Overview

Morley is a software for measuring morphology parameters of plants using plants photo, which is noninvasive for plants. Morley calculates sprouths length, length of the longest root, summ of the length all the roots and plant area, represents statistical analysis with plots: bar plots of the parameters of different groups, their comparison on heatmap and histogramms with distribution of the parameters in different groups.

You and use GUI, that is plased here, or learn the algorithm deeper using Jupyter Notebook placed [here] (https://github.com/dashabezik/plants)

# Installation

Morley requires Python 3.7 or newer. You can install Morley GUI from PyPI:
```
pip install morley
```
\* *use this command in a Python compiller or in Bash command line*

Alternatively, you can install directly from GitHub:

```
pip install git+https://github.com/dashabezik/Morley.git
```

# Example

Run ``` morley ``` command on the command line (or in the compiller), follow the instructions in the User Guide (more detailed instruction is in [supplementary material]()). Check the results in the chosen output folder. 

Raw photos, which you can use an example are placed [here](https://github.com/dashabezik/plants)

![terminal animation](img/morley_launch.gif)

To compare the recieved results you can look through the results recieved by me [here](https://github.com/dashabezik/plants)

# User Guide
### Input files

Morley supports the most popular image files formats (JPEG files, Portable Network Graphics, et al., full list [here](https://docs.opencv.org/3.4/d4/da8/group__imgcodecs.html#ga288b8b3da0892bd651fce07b3bbd3a56) ).
All the photos should be equally oriented (sproughts->roots line should have the same direction).
To convert pixels into millimeters you should locate a size standard near the plants on the sptoughts side. More information in [article](some_link).

Folder with photos must have certain structure: head folder contains subfolders with raw photos divided by the groups and template folder, subfolders contain raw photos and template photo contains seed photo.
<img src="https://github.com/dashabezik/Morley/blob/main/img/folder_tree.png" width=50% height=50%>
 **Figure 1.** Folders structure.

Head folder will be used as the beggining of the report files name (for example HeadFolderName_MeasuringParameter_Date.csv  -->  4567days_seed_germ_2022-09-21.csv). Names of subfolders will be used as group labels in visualisation part. 

### Search parameters

Before the measurement start choose:
* angle of photo rotation;
* bluring parameters;
* sproughts and roots color ranges;
* seeds color range;
* area of the size standerd, $mm^2$;
* germination threshold, mm (plants with sprought and roots lengths below that threshold will be counted together with non-germinated seeds).

1) To choose the angle rotate pic up to the example condition: sptoughts are on the left and roots are on the right. See the gif above or read [suplementary materials](article link)
2) Bluring parameters are 'morph', 'gauss', 'canny_top'* . Examples of these parameters you can look in the **Table 1** above. On this step your goal is to find the values of the parameters to reach covering plants with contours and aviod their merging. Initial values of the parameters are setted, you should just fix them a little bit if it will be needed.

\* *‘morph’ is a size of structuring element for morphological transformation, 
‘gauss’ is the parameter of gaussian blurring, 
‘canny_top’ is the threshold for contours’ identification: any edges with intensity gradient more than ‘canny_top’ are detected as edges*

3) Color ranges for sproughts and roots should be chosen as light-green and light-violet colors in HSV system (picture)


**Table 1**

||Wheat 4-7 days|Peas treated with ferrum salt|Wheat treated with Fe nanoparticles|
| ---------|-------------------|-----------------------|------------------------------------|
|Blurring |Morph = 9, gauss = 7, canny_top = 179|Morph = 7, gauss = 3, canny_top = 118|Morph = 5, gauss = 3, canny_top = 130|
|Leaves color|h:(0, 43); s:(0,255); v:(102,255)|h:(0, 50); s:(0,255); v:(94,255)|h:(0, 71); s:(0,255); v:(120,255)|
|Roots color|h:(54, 255); s:(0,255); v:(118,255)|h:(61, 255); s:(0,255); v:(94,255)|h:(51, 255); s:(0,255); v:(134,255)|
|Seed color|h:(0, 21); s:(88,255); v:(179,255)|h:(0, 28); s:(111,255); v:(132,255)|h:(0, 31); s:(35,255); v:(156,255)|










