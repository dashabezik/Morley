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

![terminal animation](morley/morley_launch.gif)

To compare the recieved results you can look through the results recieved by me [here](https://github.com/dashabezik/plants)

# User Guid
### Input files

Morley supports the most popular image files formats (JPEG files, Portable Network Graphics, et al., full list [here](https://docs.opencv.org/3.4/d4/da8/group__imgcodecs.html#ga288b8b3da0892bd651fce07b3bbd3a56) ).
Folder with photos must have certain structure: head folder contains subfolders with raw photos divided by the groups and template folder, subfolders contain raw photos and template photo contains seed photo.


head folder will be used as the beggining of the report files name

