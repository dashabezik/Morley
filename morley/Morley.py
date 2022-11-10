#!/usr/bin/env python
# coding: utf-8

# # Import

# In[ ]:


import imutils
from imutils import contours
import scipy
from scipy.spatial import distance as dist
import cv2 as cv
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
from os import path, listdir
import os, random
import time
import seaborn  as sns
import gc
import sys
import datetime
import csv
import math
from sklearn.linear_model import LinearRegression
import matplotlib.patches as mpatches
from collections import defaultdict
import time
from . import gui

# # Functions

# If you need files which include russian letters in their names, read images with 
# 
# cv2.imdecode(np.fromfile('изображение.png', dtype=np.uint8), cv2.IMREAD_UNCHANGED)
# 
# Look https://www.cyberforum.ru/python/thread2513567.html

# In[1]:


def nothing(*arg):
    pass


# ### Rotate img

# Function for picture rotation (clockwise). Arguments:
# 
# \- img: picture after cv.imread.
# 
# \- rotate: int or str, 90,180 or 270. Default None 

# In[ ]:


def rotate_pic(img,rotate = None):
    rotate_dict = {90:cv.ROTATE_90_CLOCKWISE,
              180:cv.ROTATE_180,
              270:cv.ROTATE_90_COUNTERCLOCKWISE}
    if rotate!=None:
        if int(rotate)!=0:
            img = cv.rotate(img, rotate_dict[int(rotate)])
    return img


# ### Function for histogram plot.
# Arguments:
# 
# \- l_or_r: str, leaves or roots(or plants), is used for plot title
# 
# \- color_deff: dict, int or seaborn color palette, is used for coloring plot. If the type is int, meaning the quantity of groups on histogram, then seaborn Set2 will be used
# 
# \- df: pandas DataFrame, data for histogram plot
# 
# \- columns: dict or list, enumeration of needed columns in dataset. If it is necessary to merge different columns of df, you can use dict, where keys are test groups and values are columns names of df of test groups subsections. Isf you need the same task you also can use 2D list, but int numbers will be as group numbers. For example: columns = {'Monday': \['M_col1', 'M_col2'\], 'Friday':\['F_col1', 'F_col2','F_col3'\]}
# 
# \- is_save: bool, True or False, means the necessity of saving the plot
# 
# \- figname: str, ends with the needed format (jpg, png). Means the name of the file, if you want it to be saved. It also may contain the path to the needed folder.
# 
# \- top_border: int or float. Means the top border of the data. If the data has unreal outliers you can drop it using this threshold. The default value is 300
# 
# \- xlabel: str. The default is 'length, mm'
# 
# \- param: str. Means the measured parameter (for ex. length, square, width).The default is 'length'

# In[ ]:


def pic_filename(plot_type, plant_param, path):
    report_filename = (str(path[:-1])+'_'+str(plant_param)+'_'+plot_type+
                             '_'+str(datetime.datetime.now().date())+'.jpg')
    return report_filename

def hist(tmp_l,tmp_r_max, tmp_r_sum,tmp_p, whiskers_dict, is_save = False, figname=None):
    fig, axes = plt.subplots(len(tmp_l.columns), 4, figsize=(35, 8*len(tmp_l.columns)))

    matplotlib.rcParams.update({'font.size': 20})
    param_type = 0
    fig.suptitle('X axis: Length, mm (root max, root sum and leaves); Area, mm2 (plant area);'
                 +'\n Y axis: Frequency, rel. units')
    param = ['leaves','roots_max','roots_sum', 'plant_area']
    for tmp in [tmp_l,tmp_r_max, tmp_r_sum, tmp_p]:
        iterator = 0
        
        for g in tmp_l.columns:
            tmp[g] = tmp[tmp[g]>0][g]
            plt.subplot(len(tmp.columns), 4, param_type+4*iterator+1)
            mean = round(pd.Series(tmp[g].values.reshape(-1)).dropna().mean())
            ci = round(whiskers_dict[param[param_type]][g])
            label = (str(param[param_type])+' '+str(g) +'\n'+
                     f'shapiro p-value = {scipy.stats.shapiro(pd.Series(tmp[g].values.reshape(-1)).dropna())[1]:.2e}'+
                    '\n'+'mean = '+str(mean)+'$\pm$'+str(ci))

            sns.distplot(pd.Series(tmp[g].values.reshape(-1)).dropna(),
                     color=sns.color_palette("Set2")[iterator],
                                 label=label)
            plt.xlim(left = 0)
            light = mpatches.Patch(color=sns.color_palette("Set2")[param_type], label=r'{param}'.format(param =  param[param_type]))
            plt.legend(loc = 'upper right')

            iterator +=1
        param_type+=1
#     plt.show()
    for ax in axes.flat:
        ax.set(xlabel='x-label', ylabel='y-label')
        
    if is_save:
        if figname is None:
            figname = pic_filename('hist','l_rm_rs',path_to_file_folder_fixed)
            print(path.join(path_to_output_dir,figname))
        plt.savefig(path.join(path_to_output_dir,figname),bbox_inches = 'tight')
    plt.show()


# ### Shapiro-Wilk test function
# The function is used for checking the normality for the data. Perform the Shapiro-Wilk test for normality.
# 
# The Shapiro-Wilk test tests the null hypothesis that the data was drawn from a normal distribution.
# 
# The function builds the table with the test results for two type or data: for leaves and for roots. As a result the function returns 1D Pandas table with the results of the test.
# 
# Arguments:
# 
# \- df: pandas DataFrame. The data for the test
# 
# \- columns_l: dict, enumeration of needed columns in dataset with the data for leaves. If it is necessary to merge different columns of df, you can use dict, where keys are test groups and values are columns names of df of test groups subsections. Isf you need the same task you also can use 2D list, but int numbers will be as group numbers. For example: columns = {'Monday': \['M_col1_leaves', 'M_col2_leaves'\], 'Friday':\['F_col1_leaves', 'F_col2_leaves','F_col3_leaves'\]} 
# 
# \- columns_r: dict, enumeration of needed columns in dataset with the data for roots. If it is necessary to merge different columns of df, you can use dict, where keys are test groups and values are columns names of df of test groups subsections. Isf you need the same task you also can use 2D list, but int numbers will be as group numbers. For example: columns = {'Monday': \['M_col1_roots', 'M_col2_roots'\], 'Friday':\['F_col1_roots', 'F_col2_roots','F_col3_roots'\]} 
# 
# \-is_save: bool, True or False, means the necessity of saving the table
# 
# \- figname: str, ends with the needed format (csv). Means the name of the file, if you want it to be saved. It also may contain the path to the needed folder.

# In[ ]:


def shapiro_test (df, columns_dict):
    import scipy.stats as sps
    shapiro_table = pd.DataFrame(index =list(str(x) for x in list(columns_dict[list(columns_dict.keys())[0]].keys())),
                                columns = columns_dict.keys())

    for i in list(columns_dict.keys()): # columns
        for j in shapiro_table.index: #strings
            sh_result = scipy.stats.shapiro(df[columns_dict[i][str(j)]].dropna().values.reshape(-1))[1]
            shapiro_table.loc[j][i]  =sh_result

    return shapiro_table


# ### P-value function
# The function is used for checking the independence of two groups using T-test building the table for two type of the data: for leaves and for roots. As a result the function returns symetric 2D Pandas table with the results of the test containing 2 blocks (leaves and roots). Each cell is the result of the T-test by comparing groups of the column and of the row.
# 
# Arguments:
# 
# \- df: pandas DataFrame. The data for the test
# 
# \- columns_l: dict, enumeration of needed columns in dataset with the data for leaves. If it is necessary to merge different columns of df, you can use dict, where keys are test groups and values are columns names of df of test groups subsections. Isf you need the same task you also can use 2D list, but int numbers will be as group numbers. For example: columns = {'Monday': \['M_col1_leaves', 'M_col2_leaves'\], 'Friday':\['F_col1_leaves', 'F_col2_leaves','F_col3_leaves'\]} 
# 
# \- columns_r: dict, enumeration of needed columns in dataset with the data for roots. If it is necessary to merge different columns of df, you can use dict, where keys are test groups and values are columns names of df of test groups subsections. Isf you need the same task you also can use 2D list, but int numbers will be as group numbers. For example: columns = {'Monday': \['M_col1_roots', 'M_col2_roots'\], 'Friday':\['F_col1_roots', 'F_col2_roots','F_col3_roots'\]} 
# 
# \-is_save: bool, True or False, means the necessity of saving the table
# 
# \- figname: str, ends with the needed format (csv). Means the name of the file, if you want it to be saved. It also may contain the path to the needed folder.

# In[ ]:


def pvalue_calc(df1,df2,is_norm):
    ret = 0
    is_not_norm = not is_norm
    method = is_norm*'Unpaired T-test'+is_not_norm*'Mann Whitney U-test'
    if method=='Unpaired T-test':
        ret = scipy.stats.ttest_ind(df1,df2)
    if method=='Mann Whitney U-test':
        ret = scipy.stats.mannwhitneyu(df1,df2, use_continuity = False ,alternative = 'two-sided')
    return ret        

def p_value_function (df, columns, is_norm):
    import scipy.stats as sps
    pvalue_table = pd.DataFrame(index = list(str(x) for x in columns.keys()),
                                columns=list(str(x) for x in columns.keys()))
    for i in (list(columns.keys())):
        for j in (list(columns.keys())):
            pvalue_table[str(i)].loc[str(j)] = pvalue_calc(pd.Series(df[columns[i]].values.reshape(-1)).dropna(),
                                                            pd.Series(df[columns[j]].values.reshape(-1)).dropna(),is_norm)[1]
    pvalue_table.fillna(value='.', inplace = True)

    return pvalue_table


# Информация о пакете Annotator взята из ресурса:
# 
# https://levelup.gitconnected.com/statistics-on-seaborn-plots-with-statannotations-2bfce0394c00

# ### Bar-plot function
# Arguments:
# 
# \- l_or_r: str, leaves or roots(or plants), is used for plot title
# 
# \- color_deff: dict, int or seaborn color palette, is used for coloring plot. If the type is int, meaning the quantity of groups on histogram, then seaborn Set2 will be used
# 
# \- df: pandas DataFrame, data for histogram plot
# 
# \- columns: dict or 1d list, enumeration of needed columns in dataset. If it is necessary to merge different columns of df, you can use dict, where keys are test groups and values are columns names of df of test groups subsections. Isf you need the same task you also can use 2D list, but int numbers will be as group numbers. For example: columns = {'Monday': \['M_col1', 'M_col2'\], 'Friday':\['F_col1', 'F_col2','F_col3'\]}. // If you don't need to merge df, you can just use 1d list with group names, for example \['Monday', 'Friday'\]
# 
# \- pv_table: pandas DataFrame. P-value data to print them on the plot.
# 
# \- control_label: int or str. Name of the control group. This value also should be in columns.keys() (or in 1d list of columns). The type is dependent on the columns.keys()s elements type
# 
# \- comparison_points: list of str or int. Names of groups for comparison with the control group. The elements of the list must be in columns.keys() (or in 1d list of columns), and types also should coincide. This parametr is optional, the default is all the groups exept control_label. If you don't need all the groups to be compared, define this parametr. 
# 
# \- is_save: bool, True or False, means the necessity of saving the plot.The default is False.
# 
# \- figname: str, ends with the needed format (jpg, png). Means the name of the file, if you want it to be saved. It also may contain the path to the needed folder.
# 
# \- union_DF_length: int. Means the length of tmp DataFrame. This tmp DataFrame is needed to merge different collumns of the same group (If you have several photos as a part of the same group, you will have several columns in resulted df). The default value is 130
# 
# \- xlabel: str. The default is 'group number'
# 
# \- ylabel: str. The default is 'length, mm'
# 
# \- param: str. Means the measured parameter (for ex. length, square, width).The default is 'length'
# 

# In[ ]:


def pic_filename(plot_type, plant_param, path):
    report_filename = (str(path[:-1])+'_'+str(plant_param)+'_'+plot_type+
                             '_'+str(datetime.datetime.now().date())+'.jpg')
    return report_filename

def bar_plot_function(l_or_r, color_deff, df, columns, pv_table,
                      is_save = False, figname=None,  union_DF_length = 500, xlabel = 'group label', ylabel = 'length, mm',
                      param = 'length', auto_or_man = 'automatic',  is_drop_outliers = False):
    
    if type(columns)==dict:
        tmp = pd.DataFrame(columns=list(columns.keys()),index=np.arange(union_DF_length))
        for i in columns.keys():
            tmp[i] = pd.DataFrame(pd.Series(df[columns[i]].values.reshape(-1)).dropna()) 
        group_number_names = list(columns.keys())
    elif type(columns)==list: #### still 1D
        tmp = df
        group_number_names = columns ## если 2D массив, то range где кажд индекс +1 range(1, len(columns)+1)
    if is_drop_outliers:
        tmp = drop_outliers(tmp, tmp.columns)         
    c = sns.color_palette("Set2")
        

    matplotlib.rcParams.update({'font.size': 20})
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(r'{0}'.format(l_or_r))
    
          
    a = sns.barplot(ax = axes[0], data=tmp[group_number_names], palette=c)
    plt.xlabel(xlabel, fontsize = 20)
    plt.ylabel(ylabel, fontsize = 20)
    
    for ax in axes.flat:
        ax.set( ylabel=ylabel)
    
    whiskers = defaultdict(type(group_number_names[0]))
    i=0
    for j in group_number_names:
        whiskers[j] = ((a.get_lines()[i].get_data()[1][1]-a.get_lines()[i].get_data()[1][0])/2)
        i+=1
    
    matplotlib.rcParams.update({'font.size': 20}) 

    for i in range(0,pv_table.shape[1]):
        for j in range(0,pv_table.shape[1]):
            if i>j:
                pv_table[pv_table.columns[i]].loc[pv_table.columns[j]] = np.nan
            if pv_table[pv_table.columns[i]].loc[pv_table.columns[j]]>0.05:
                pv_table[pv_table.columns[i]].loc[pv_table.columns[j]] = 1
            if pv_table[pv_table.columns[i]].loc[pv_table.columns[j]]<0.05:
                pv_table[pv_table.columns[i]].loc[pv_table.columns[j]] = 0.00003
                
    f=np.array(pv_table, dtype='float64')
    a = [[0.247, 0.41176, 0.349],[0.624, 0.8967, 0.81] ]
    sns.heatmap(f, xticklabels=pv_table.columns, yticklabels=pv_table.columns, cbar=False, cmap = a, ax = axes[1])
    dark = mpatches.Patch(color=a[0], label='p_value<0.05')
    light = mpatches.Patch(color=a[1], label='p_value>0.05')
    plt.legend(handles=[dark, light])

    if is_save:
        if figname is None:
            figname = pic_filename('bar',l_or_r.replace(' ', ''),path_to_file_folder_fixed)
            print(path.join(path_to_output_dir,figname))
        plt.savefig(path.join(path_to_output_dir,figname),bbox_inches = 'tight')
    plt.show()
    
    return tmp, whiskers


# ### Seed germination counter
# 
# Function for counting the rate of non germinated seeds in all groups. The default value to consider the seed as nongerminated is 10 mm. If any root of leave has appropriate length, the seed is considered germinated (look at the table).
# 
# 
# | l  r || l  r || l  r || l  r || l  r |
# |   --- || --- ||  --- ||   --- ||   --- |
# | 16   0 || 9  9 || 9  50 || 16  50 ||  0 5 |
# |   V || X ||  V ||   V ||   X |
# 
# 

# In[ ]:


def seed_germination(df,group_names,threshold = 10, is_save = False, figname = None):
    non_germinated_table = pd.DataFrame(columns=group_names, index=np.arange(1))
    for i in group_names:
        l_columns = 'leaves_length_'+path_to_file_folder_fixed+i
        r_columns = 'roots_max_length_'+path_to_file_folder_fixed+i
        l = df[[i for i in measure_full2.columns if i.startswith(l_columns)]]
        r = df[[i for i in measure_full2.columns if i.startswith(r_columns)]]
        full_number = (np.array((r>=0))*np.array((l>=0))).sum()
        non_germinated_table[i].loc[0] = 1-(np.array((r<threshold))*np.array((l<threshold))).sum()/full_number
    
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111)    
    plt.xlabel('group label', fontsize = 20)
    plt.ylabel('distribution density', fontsize = 20)    
    sns.barplot(x=non_germinated_table.columns, y = non_germinated_table.values[0],
                palette=sns.color_palette("Set2"))
    plt.title('Germination efficiency', fontsize=20)

    if is_save:
        if figname is None:
            figname = pic_filename('bar','seed_germ',path_to_file_folder_fixed)
        plt.savefig(path.join(path_to_output_dir,figname))
    plt.show()
    return non_germinated_table


# ### Length calculating
# 
# The function calculates the plant part length by its width, square and pixel_per_metric coefficient. If the width is zero, functon returns length value as 0.

# In[ ]:


def length(width, square, pixel):
    if (width!=0):
        length = square/(width*pixel)
    else:
        length = 0
    return length


# ### Folders_list_functions

# Если папок для перечисления много, то можно вызвать функцию, которая скомпанует всё что лежит в папке в один лист

# In[ ]:


def folders_list_function(path_to_file_folder):
    folders_list=[]
    for filename_in_folder in listdir(path_to_file_folder):
        folders_list.append(filename_in_folder)

    if '.ipynb_checkpoints' in folders_list:
        folders_list.remove('.ipynb_checkpoints')
    if 'template' in folders_list:
        folders_list.remove('template')
    return folders_list


# ### files_dicts
# 
# The function builds dicts with names of the columns in df, based on photos filenames. The main feachure is that all the columns names are merged by test groups names. Keys are the test groups names. Returns leaves_dict, roots_dict, roots_area_dict, plant_area_dict -- dicts with columns related to a specific test group.

# In[ ]:


def files_dicts(path_to_file_folder_fixed):
    plant_parameters = ['roots_sum','roots_max','plant_area','leaves']
    
    folders_list = folders_list_function(path_to_file_folder_fixed)

    leaves_dict = dict()
    for i in folders_list:
        leaves_dict[i] = []

    for g in folders_list:
    #     pic_num=0
        path_to_file_folder = path_to_file_folder_fixed
        path_to_file_folder = path.join(path_to_file_folder, str(g)+'/')
        for filename_in_folder in listdir(path_to_file_folder):
    #         pic_num +=1
    #         if pic_num>3:
    #             continue
            if filename_in_folder!='.ipynb_checkpoints':
                file_name = path.join(path_to_file_folder, filename_in_folder)
                leaves_dict[g].append('leaves_length_'+file_name)

    roots_dict = dict()
    for i in folders_list:
        roots_dict[i] = []   

    for g in folders_list:
    #     pic_num=0
        path_to_file_folder = path_to_file_folder_fixed
        path_to_file_folder = path.join(path_to_file_folder, str(g)+'/')
        for filename_in_folder in listdir(path_to_file_folder):
    #         pic_num +=1
    #         if pic_num>3:
    #             continue
            if filename_in_folder!='.ipynb_checkpoints':
                file_name = path.join(path_to_file_folder, filename_in_folder)
                roots_dict[g].append('roots_length_'+file_name)

    roots_max_dict = dict()
    for i in folders_list:
        roots_max_dict[i] = [] 

    for g in folders_list:
        path_to_file_folder = path_to_file_folder_fixed
        path_to_file_folder = path.join(path_to_file_folder, str(g)+'/')
        for filename_in_folder in listdir(path_to_file_folder):
            if filename_in_folder!='.ipynb_checkpoints':
                file_name = path.join(path_to_file_folder, filename_in_folder)
                roots_max_dict[g].append('roots_max_length_'+file_name)

    plant_area_dict = dict()
    for i in folders_list:
        plant_area_dict[i] = [] 

    for g in folders_list:
        path_to_file_folder = path_to_file_folder_fixed
        path_to_file_folder = path.join(path_to_file_folder, str(g)+'/')
        for filename_in_folder in listdir(path_to_file_folder):
            if filename_in_folder!='.ipynb_checkpoints':
                file_name = path.join(path_to_file_folder, filename_in_folder)
                plant_area_dict[g].append('plant_area_'+file_name)
                
    dicts = {'roots_sum':roots_dict,
             'roots_max':roots_max_dict,
             'plant_area': plant_area_dict,
             'leaves':leaves_dict}
    
    return dicts


# ### Drop outliers
# 
# Function drops values lying lower or upper than 25 or 75 quartille.

# In[ ]:


def drop_outliers(df, columns):
    for x in list(columns):
#         print(x)
#         print(df[x].values.reshape(-1))
        q75,q25 = np.percentile(pd.Series(df[x].values.reshape(-1)).dropna(),[75,25])
#         print(q75,q25)
        intr_qr = q75-q25

        max = q75+(1.5*intr_qr)
        min = q25-(1.5*intr_qr)
        
        df.loc[df[x] < min,x] = np.nan
        df.loc[df[x] > max,x] = np.nan
    return df


# ### Drop seeds

# Можно попробовать посчитать сколько ненулевых пикселей в маске и это будет быстрее чем попиксельный подсчет. Т к тип маски это массив нулевых-ненулевых пиксилей

# In[ ]:


def pixel_color_conditions(pixel ,h1=0, h2=255, s1=0, s2=255, v1=0, v2=255):
    h, s, v = pixel # pixel = img_hsv[x, y]
    h_condition = (h>=h1)&(h<=h2)
    s_condition = (s>=s1)&(s<=s2)
    v_condition = (v>=v1)&(v<=v2)
    full_condition = h_condition*s_condition*v_condition
    return full_condition

def drop_seeds_slow(src, contours, h1=0, h2=255, s1=0, s2=255, v1=0, v2=255):
    src_hsv  = cv.cvtColor(src, cv.COLOR_BGR2HSV)
    seeds_square_list = np.zeros(len(contours))
    h_min = np.array((h1, s1, v1), np.uint8)
    h_max = np.array((h2, s2, v2), np.uint8)
    tresh = cv.inRange(src_hsv, h_min, h_max)
    for i in range(len(contours)):
        c = contours[i]
        cimg1 = np.zeros_like(src)
        cv.drawContours(cimg1, contours, i, color=255, thickness=-1)
        pts = np.where(cimg1 == 255)
        for x, y in zip(pts[0], pts[1]):                   
            if pixel_color_conditions(src_hsv[x, y], h1, h2, s1, s2, v1, v2):
                src_hsv[x, y] = [0,0,0]
                seeds_square_list[i]+=1
    src  = cv.cvtColor(src_hsv, cv.COLOR_HSV2BGR)
    plt.figure(figsize=(14,14))
    plt.imshow(tresh)
    plt.show()
    return src, seeds_square_list

def drop_seeds(src, h1=0, h2=255, s1=0, s2=255, v1=0, v2=255):
    src_hsv  = cv.cvtColor(src, cv.COLOR_BGR2HSV)
    h_min = np.array((h1, s1, v1), np.uint8)
    h_max = np.array((h2, s2, v2), np.uint8)
    tresh = cv.inRange(src_hsv, h_min, h_max)
    tresh=cv.bitwise_not(tresh)
    mask = cv.bitwise_or(src, src, mask=tresh)
    return mask


# ### Linear approximation
# After seed position search it is needed to find the line dividing leaves and roots.

# In[ ]:


def linear_approx(x,y):
    x=np.array(x).reshape(len(x),1)# Построй м модель
    m=LinearRegression()
    z1=m.fit(x,y)
    z1.score(x,y)# Показывать примерный эффект - 1.0
    z1.predict([[10]])#//Прогноз, результат31
    p=z1.coef_# Отображение коэффициента
    q=z1.intercept_
    return p, q


# ### Color range counter

# Функция считает, сколько пикселей на картинке лежит в данном цветовом диапазоне внутри каждого контура. На выходе дает массивразмером совпадающим с 

# In[ ]:


def color_range_counter(src, contours, h1=0, h2=255, s1=0, s2=255, v1=0, v2=255):
    src_hsv  = cv.cvtColor(src, cv.COLOR_BGR2HSV)
    h_min = np.array((h1, s1, v1), np.uint8)
    h_max = np.array((h2, s2, v2), np.uint8)
    thresh = cv.inRange(src_hsv, h_min, h_max)
    thresh = np.clip(thresh, 0,1)
    counter = np.zeros(len(contours))
    for i in range(len(contours)):
        c = contours[i]
        cimg1 = np.zeros_like(thresh)
        cv.drawContours(cimg1, contours, i, color=255, thickness=-1)
#         plt.figure(figsize=(14,14))
#         plt.imshow(cimg1)
#         plt.show()
#         plt.figure(figsize=(14,14))
#         plt.imshow(thresh)
#         plt.show()

#         print('plant area = ', cv.contourArea(c) )
        cimg1 = np.clip(cimg1, 0 ,1)
        counter[i] = (cimg1*thresh).sum()
#     print(counter)
    return counter


# ### Find_paper

# In[ ]:


def find_paper (src, template_size, square_threshold, position_x_axes, canny_top = 100, canny_bottom = 10, morf = 7, gauss = 3):
    gr = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    bl=cv.GaussianBlur(src,(gauss,gauss),0)
    canny = cv.Canny(bl, canny_bottom, canny_top)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (int(morph), int(morph)))
    closed = cv.morphologyEx(canny, cv.MORPH_CLOSE, kernel)
    contours0 = cv.findContours(closed.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
    # contours0 = contours0[0] if imutils.is_cv2() else contours0[1]
    (contours0, _) = contours.sort_contours(contours0)
    

    
    for cont in contours0:
#         cv.drawContours(src,[cont],0,(0,255,0),-2)
        center, radius = cv.minEnclosingCircle(cont)
        if (cv.contourArea(cont)>square_threshold)&(center[0]>position_x_axes): #position = src.shape[1]//8
            sm = cv.arcLength(cont, True)
            apd = cv.approxPolyDP(cont, 0.025*sm, True)
            center, radius = cv.minEnclosingCircle(cont)
            cv.drawContours(src, [cont], -1, (0,255,0), -2)
            if len(apd) == 4:
                is_paper_founded = True
                paper = cont
    #                 print('paper')
                cv.drawContours(src, [cont], -1, (0,255,0), -2)
                pixelsPerMetric = 7.6
                box = cv.minAreaRect(cont)
                box = cv.boxPoints(box)
                box = np.array(box, dtype="int")
                (tl, tr, br, bl) = box
                (tltrX, tltrY) = midpoint(tl, tr)
                (blbrX, blbrY) = midpoint(bl, br)
                (tlblX, tlblY) = midpoint(tl, bl)
                (trbrX, trbrY) = midpoint(tr, br)
                dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
                dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
    #             if (dB/template_size > (square_threshold/)):                
                pixelsPerMetric = math.sqrt(cv.contourArea(cont)/(template_size))
                ppm.append(pixelsPerMetric)
            else:
                pixelsPerMetric = ppm[-1]

            
            

            rect = cv.minAreaRect(apd)
            box = cv.boxPoints(rect) # поиск четырех вершин прямоугольника
            box = np.int0(box) # округление координат
            print(pixelsPerMetric)
            cv.drawContours(src,[cont],0,(0,255,0),-2)
            break
    plt.figure(figsize=(14,14))
    plt.imshow(src)
    plt.show()
    return pixelsPerMetric


# ### Random file

# In[ ]:


def random_file(path_to_file_folder):
    a=random.choice(os.listdir(path_to_file_folder))
    while (a=='template')|(a=='.ipynb_checkpoints'):
        a=random.choice(os.listdir(path_to_file_folder))
    path_to_file = path.join(path_to_file_folder, a+'/')
    b = random.choice(os.listdir(path_to_file))
    while (b=='.ipynb_checkpoints'):
        b=random.choice(os.listdir(path_to_file))
    path_to_file = path.join(path_to_file, b)
    return path_to_file


# ### Class of parameters

# In[ ]:


class picture_params:   
    count = 0  
    def __init__(self):  
        picture_params.count += 1 
    def contour_params(self, morph, gauss, canny_bottom, canny_top): 
        self.morph = morph 
        self.gauss = gauss 
        self.canny_bottom = canny_bottom 
        self.canny_top = canny_top
  
    def color(self, h1,h2,s1,s2,v1,v2):
        self.h1 = h1
        self.h2 = h2
        self.s1 = s1
        self.s2 = s2
        self.v1 = v1
        self.v2 = v2
    
    def display_count(self):  
        print('Groups total number: %d' % picture_params.count)
        
    def return_bl_params(self):  
        return self.morph, self.gauss, self.canny_bottom, self.canny_top 
    def display_element(self):
        attrs = vars(self)
        print(', '.join("%s: %s" % item for item in attrs.items()))
        
    def return_colors(self):
        return self.h1,self.h2,self.s1,self.s2,self.v1,self.v2


# # Main.AUTOMATIC

# ## Parameters

# In[ ]:


def position_x_axes(src,divider):
    return src.shape[1]//divider


def add_annotation(name, text):
    with open(name, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(text)
        f.write(content)
        writer = csv.writer(f)
    


def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def get_state_values(param):
    l=[]
    for i in gui.state[param]:
        l.append(gui.state[param][i])
    return l

def search(paper_size):
    gui.state['paper_area'] = paper_size.get()
    ppm = [7.45]
    rotate = gui.state['rotation']
    path_to_file_folder_fixed = gui.state['paths']['input']
    path_to_output_dir = gui.state['paths']['out_dir']
    paper_area = gui.state['paper_area']
    paper_area_thresold = gui.state['paper_area_thresold']
    x_pos_divider = 10
    contour_area_threshold = gui.CONTOUR_AREA_THRESHOLD # look at your img size and evaluate the threshold, 1000 is recomended
    template_filename = gui.state['paths']['template_file']
    hlb,hlt,slb,slt,vlb,vlt = get_state_values('roots')
    hrb,hrt,srb,srt,vrb,vrt = get_state_values('leaves')
    hsb,hst,ssb,sst,vsb,vst = get_state_values('seed')
    morph, gs, c_top,_ = get_state_values('settings')
    c_bottom=0
    
    ###SEARCH###
    
    measure_full2 = pd.DataFrame(columns=[],index=np.arange(30))
    # ppm - pixel per metric, массив с коэфам пересчета пикселя в мм, на случай плохого поиска стикера на фото

    folders_list = folders_list_function(path_to_file_folder_fixed)
    # pic_num=0
    for g in folders_list:
        path_to_file_folder = path.join(path_to_file_folder_fixed, str(g)+'/')
        for filename_in_folder in listdir(path_to_file_folder):

            if filename_in_folder=='.ipynb_checkpoints':
                continue
    #         pic_num +=1
    #         if pic_num>3:
    #             continue
            ### CONTOURS ###

            print('...LOOKING FOR CONTOURS...')

            file_name = path.join(path_to_file_folder, filename_in_folder)

            ### Plant contour ####       

            src = cv.imread(file_name)
            src = rotate_pic(src, rotate)
            gr = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
            bl=cv.GaussianBlur(src,(gs,gs),0)
            canny = cv.Canny(bl, c_bottom, c_top)
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (morph,morph))
            closed = cv.morphologyEx(canny, cv.MORPH_CLOSE, kernel)
            contours0 = cv.findContours(closed.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
            # contours0 = contours0[0] if imutils.is_cv2() else contours0[1]
            (contours0, _) = contours.sort_contours(contours0)
            pixelsPerMetric = None
            quantity_of_plants = 0
            real_conts = []

            pixelsPerMetric = find_paper(src,paper_area,paper_area_thresold, position_x_axes(src,x_pos_divider),
                                         canny_top=c_top, canny_bottom=c_bottom,morf=morph)

            for cont in contours0:
                if (cv.contourArea(cont)>contour_area_threshold):
                    center, radius = cv.minEnclosingCircle(cont) #recomended range of plants position is between 1/3 and 2/3
                    if ((cv.contourArea(cont) > contour_area_threshold)&
                        (center[0] > src.shape[1]//3)&(center[0] < src.shape[1]*2//3)):
                        real_conts.append(cont)
    #                     cv.drawContours(src,[cont],0,(255,255,5),2)

            quantity_of_plants = len(real_conts)
            print(quantity_of_plants)
            print(pixelsPerMetric)

            ### SEEDS ###

            print('...LOOKING FOR SEEDS POSITION...')

            img2 = cv.imread(file_name,0)
            img2 = rotate_pic(img2, rotate)
            template = cv.imread(template_filename,0)
            template = rotate_pic(template, rotate)
        #         template = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
            w, h = template.shape[::-1]

            # All the 6 methods for comparison in a list
            # methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
            #             'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']
            methods = ['cv.TM_CCOEFF_NORMED']
            for meth in methods:
                img = img2.copy()
                method = eval(meth)
                # Apply template Matching
                res = cv.matchTemplate(img,template,method)
                threshold = 0.55
                loc = np.where( res > threshold)
                x=np.array([])
                y=np.array([])
                for pt in zip(*loc[::-1]):
                    if (pt[0] > src.shape[1]/3)&((pt[0] < 2*src.shape[1]/3)):
                        cv.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0,0,255), 5)
                        x=np.append(x,pt[0])
                        y=np.append(y,pt[1])
                slope, intercept = linear_approx(y,x)
                p1 = [int(intercept),0]
                p2 = [int(slope*img.shape[0]+intercept),img.shape[0]]
                pts_leaves = np.array([[0,0],p1,p2,[0,img.shape[0]]])
                pts_roots = np.array([[p1[0]+3*w//4,p1[1]],[p2[0]+3*w//4,p2[1]],[img.shape[1],img.shape[0]],[img.shape[1],0]])
                plt.figure(figsize = (14,14))
                plt.subplot(121),plt.imshow(res,cmap = 'gray')
                plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
                plt.subplot(122),plt.imshow(img,cmap = 'gray')
                plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
                plt.suptitle(meth)
                plt.show()

            src = drop_seeds(src,hsb,hst,ssb,sst,vsb,vst)
            src_black_seeds = src.copy()
            src_black_seeds = cv.cvtColor(src_black_seeds, cv.COLOR_BGR2HSV)
            ### COLOR ###

            print('...MAKING COLOR...')

            overlay = src.copy()
            cv.drawContours(overlay, [pts_leaves], -1,(0,224,79), -1)
            opacity = 0.25
            cv.drawContours(overlay, [pts_roots], -1, (240,0,255), -5)
            cv.addWeighted(overlay, opacity, src, 1 - opacity, 0, src)
            bl = cv.medianBlur(src, 7)
            bl=cv.GaussianBlur(bl,(5,   5),0)
            img_hsv = cv.cvtColor(bl, cv.COLOR_BGR2HSV)
            cv.imwrite('colored/{0}'.format(filename_in_folder), src)
            print(src.shape)
            print('mean_left_x = ', mean_left_x, 'mean_right_x  =', mean_right_x)

            ## WIDTH ###

            print('...WIDTH CALCULATION...')
            measure = pd.DataFrame(columns=['roots_area_{0}'.format(file_name), 'leaves_area_{0}'.format(file_name),
                                            'roots_length_{0}'.format(file_name), 'leaves_length_{0}'.format(file_name),
                                            'roots_width_{0}'.format(file_name), 'leaves_width_{0}'.format(file_name),
                                           'plant_area_{0}'.format(file_name),'seed_area_{0}'.format(file_name),
                                           'roots_max_length_{0}'.format(file_name),'roots_max_width_{0}'.format(file_name)],
                                   index=np.arange(len(real_conts)))


            for i in range(quantity_of_plants):
                is_first = True
                is_first_r = True
                is_first_r_max = True
                roots = 0
                leaves = 0
                lamount = 0
                ramount = 0
                r_max_amount = 0
                c = real_conts[i]
                left = tuple(c[c[:, :, 0].argmin()][0])
                right = tuple(c[c[:, :, 0].argmax()][0])
                top = tuple(c[c[:, :, 1].argmin()][0])
                bottom = tuple(c[c[:, :, 1].argmax()][0])
                cv.line(img_hsv, left, right, (255, 255, 255), thickness=2)
                step = (right[0]-mean_right_x)//3
                if (mean_right_x-left[0])//3!=0:                
                    for y in range(left[0],mean_left_x,(mean_right_x-left[0])//3):
                        is_first = True
                        for x in range(top[1],bottom[1]):#иттерация по вертикали, т к img.shape => (height, width), но компонента контура (х,у)
                            h, s, v = img_hsv[x, y]
                            if (cv.pointPolygonTest(real_conts[i],(x,y), False)):
                                if (v>vlb)&(h>hlb)&(h<hlt):
                                    lamount = lamount + 1*is_first
                                    is_first = False
                                    leaves += 1
                                else:
                                    is_first = True
                            else:
                                is_first = True
                else:
                    leaves_width = 0 

    # r_max_amaunt - счетчик для максимальной длины корня, ramount - для суммарной длины                

                if step!=0:
                    for y in range(mean_right_x, right[0],step):#идем по ввертикальным линиям   
                        is_first_r = True
                        is_first_r_max = True
                        for x in range(top[1],bottom[1]):#иттерация по вертикали, т к img.shape => (height, width), но компонента контура (х,у)
                            h, s, v = img_hsv[x, y]
                            if (cv.pointPolygonTest(real_conts[i],(x,y), False)):# если точка внутри контура
                                if (v>vrb)&((h>hrb)&(h<hrt)):# если эта точка = корень, а не фон 
                                    ramount = ramount + 1*is_first_r#если это первое вхождение корня, то число корней+=1
                                    r_max_amount=r_max_amount+1*is_first_r_max
                                    is_first_r_max = False
                                    is_first_r = False#далее вхождение уже не первое
                                    roots += 1#число пикселей +=1
                                else:
                                    is_first_r = True #если это не корень а фон, то вхождения нет
                            else:
                                is_first_r = True#если это не в контуре, то вхождения точно нет
                if (lamount == 0.0)|(lamount == 0):
                    leaves_width = 0
                else: 
                    leaves_width = leaves/lamount

                if (ramount == 0.0)|(ramount == 0)|(step == 0):
                    roots_width = 0
                    roots_max_width = 0
                else:
                    roots_width = roots/ramount
                    roots_max_width = roots/r_max_amount

                measure['roots_width_{0}'.format(file_name)].iloc[i] = roots_width
                measure['roots_max_width_{0}'.format(file_name)].iloc[i] = roots_max_width
                measure['leaves_width_{0}'.format(file_name)].iloc[i]= leaves_width

            ### PIXEL COUNTING ###

            print('...PIXEL COUNTING...')

            for i in range(len(real_conts)):
                c = real_conts[i]
                measure.iloc[i]['plant_area_{0}'.format(file_name)] = cv.contourArea(c)
    #             measure.iloc[i]['seed_area_{0}'.format(file_name)] = seed_area
            measure['leaves_area_{0}'.format(file_name)] =color_range_counter(src, real_conts, hlb,hlt,slb,slt,vlb,vlt)
            measure['roots_area_{0}'.format(file_name)] =color_range_counter(src, real_conts, hrb,hrt,srb,srt,vrb,vrt)
            measure['seed_area_{0}'.format(file_name)] =color_range_counter(src_black_seeds, real_conts, 0,1,0,1,0,1)
            measure['plant_area_{0}'.format(file_name)] = measure['plant_area_{0}'.format(file_name)]-measure['seed_area_{0}'.format(file_name)]
            measure['plant_area_{0}'.format(file_name)] = measure.apply(lambda x: x['plant_area_{0}'.format(file_name)]/(pixelsPerMetric*pixelsPerMetric), axis = 1 )
            measure['roots_length_{0}'.format(file_name)] = measure['roots_area_{0}'.format(file_name)] 
            measure['leaves_length_{0}'.format(file_name)] = measure['leaves_area_{0}'.format(file_name)] 

            measure['roots_length_{0}'.format(file_name)] = measure.apply(lambda x: length(x['roots_width_{0}'.format(file_name)],x['roots_area_{0}'.format(file_name)],pixelsPerMetric), axis = 1 )
            measure['roots_max_length_{0}'.format(file_name)] = measure.apply(lambda x: length(x['roots_max_width_{0}'.format(file_name)],x['roots_area_{0}'.format(file_name)],pixelsPerMetric), axis = 1 )
            measure['leaves_length_{0}'.format(file_name)] = measure.apply(lambda x: length(x['leaves_width_{0}'.format(file_name)],x['leaves_area_{0}'.format(file_name)],pixelsPerMetric), axis = 1 )
            measure_full2 = measure_full2.join(measure, how = 'outer')

            plt.figure(figsize = (14,14))
            plt.imshow(src)
            plt.show()
    print ("{:g} s".format(time.clock() - start_time))

    del res,bl,overlay, img, img2, img_hsv, gr, canny, src, closed, src_black_seeds
    
    dicts = files_dicts(path_to_file_folder_fixed)
    roots_sum_dict, roots_max_dict, plant_area_dict, leaves_dict = dicts.values()
    seed_germ = seed_germination(measure_full2, roots_max_dict.keys(), threshold=7, is_save=True)
    shap =shapiro_test(measure_full2, dicts)
    plant_parameters = ['roots_sum','roots_max','plant_area','leaves']
    v= [0,0,0,0]
    p_value_dict = dict(zip(plant_parameters, v))
    for i in plant_parameters:
        is_not_norm = any(shap[i]<0.05)
        is_norm = not is_not_norm
        test_type = 'test_type = '+str(is_norm*'Unpaired T-test'+is_not_norm*'Mann Whitney U-test')+'\n' +'\n'

        p_value_dict[i] = (p_value_function(measure_full2, dicts[i],is_norm),test_type)
        
    whiskers_dict = {'roots_sum': {},
                   'roots_max': {},
                   'plant_area': {},
                   'leaves': {}}
    result_dict = {'roots_sum': '',
                   'roots_max': '',
                   'plant_area': '',
                   'leaves': '',
                  'full_file_photo_separated': measure_full2}

    for i in whiskers_dict.keys():
        ylabel = 'length, mm'*(i!='plant_area')+'area, mm2'*(i=='plant_area')
        result_dict[i], whiskers_dict[i] = bar_plot_function(i, 5, measure_full2, dicts[i], p_value_dict[i][0], ylabel=ylabel,
                                  is_save= True, union_DF_length=250)
    
    hist(result_dict['leaves'],result_dict['roots_max'], result_dict['roots_sum'],result_dict['plant_area'],
     whiskers_dict,True)
    
    report_information = ('Date and time: ' + str(datetime.datetime.now())+'\n'+
                      'Program settings and initial information: \n'+ 
                      'path_to_file_folder_fixed = '+ str(path_to_file_folder_fixed)+'\n'+
                      'paper_area = '+str(paper_area)+'mm2; paper_area_thresold = '+str(paper_area_thresold)+'pixels \n'+
                      'paper threshold position = photo width/x_pos_divider = img.shape[0]/'+str(x_pos_divider)+'\n'+
                      'contour_area_threshold = '+str(contour_area_threshold)+' pixels \n'+
                      'template_filename = '+str(template_filename)+'\n'+
                      'leaves parameters'+ str(group_param_leaves.return_colors()) +'\n'+
                      'roots parameters'+str(group_param_roots.return_colors())+'\n'+
                      'seeds parameters'+str(group_param_seeds.return_colors())+'\n'+
                      'blur parameters'+str(group_param.return_bl_params())+'\n' +'\n' )

    for i in result_dict.keys():
        report_table_filename = str(path_to_file_folder_fixed[:-1])+'_'+str(i)+'_'+str(datetime.datetime.now().date())+'.csv'
        result_dict[i].to_csv(path.join(path_to_output_dir,report_table_filename))

        with open(path.join(path_to_output_dir,report_table_filename), 'r+') as f:
            content = f.read()
            f.seek(0, 0)
            f.write(report_information)
            f.write(content)
            writer = csv.writer(f)


    shap_filename = str(path_to_file_folder_fixed[:-1])+'_shapiro_'+str(datetime.datetime.now().date())+'.csv'
    shap.to_csv(path.join(path_to_output_dir,shap_filename))
    add_annotation(path.join(path_to_output_dir,shap_filename), report_information)

    for i in whiskers_dict.keys():
        pval_filename = str(path_to_file_folder_fixed[:-1])+'_pvalue_'+str(i)+str(datetime.datetime.now().date())+'.csv'
        p_value_dict[i][0].to_csv(path.join(path_to_output_dir,pval_filename))
        test_type = p_value_dict[i][1]
        add_annotation(path.join(path_to_output_dir,pval_filename), report_information+test_type)

    seed_germ_filename = str(path_to_file_folder_fixed[:-1])+'_seed_germ_'+str(datetime.datetime.now().date())+'.csv'
    seed_germ.to_csv(path.join(path_to_output_dir,seed_germ_filename))
    add_annotation(path.join(path_to_output_dir,seed_germ_filename), report_information)

    
    
    

    
    
    
    


