import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import askopenfilename, askdirectory
from PIL import ImageTk, Image
import cv2 as cv
import imutils
from imutils import contours
import numpy as np
import pandas as pd

from functools import partial
import os


import Morley

state = {
    'settings': {
        'morph': 0.0,
        'gauss':0.0,
        'canny_top': 255.0
    },
    'color':{
        'h_bottom':0.0,
        'h_top':255.0,
        's_bottom':0.0,
        's_top':255.0,
        'v_bottom':0.0,
        'v_top':255.0,
    },
    'seed':{
        'h_bottom':0.0,
        'h_top':255.0,
        's_bottom':0.0,
        's_top':255.0,
        'v_bottom':0.0,
        'v_top':255.0,
    },
    'paths': {
        'out_dir': os.getcwd()
    },
    'rotation': 1
}


contour_area_threshold = 1000

def set_state_variables(d):
    for k, v in d.items():
        if isinstance(v, dict):
            set_state_variables(v)
        if isinstance(v, float):
            d[k] = tk.IntVar(value=v)
        # if isinstance(v, str):
        #     d[k] = tk.StringVar(value=v)


def get_image_dirname(label):
    fname = askdirectory(title='Raw image directory')

    if fname:
        state['paths']['input'] = fname
        label['text'] = f'Selected image directory: {os.path.basename(fname)}.'


def get_template_file(label):
    fname = askopenfilename(title='Seed template file')

    if fname:
        state['paths']['template_file'] = fname
        label['text'] = f'Selected seed template: {os.path.basename(fname)}.'
        
def get_file(label):
    fname = askopenfilename(title='Img file')

    if fname:
        state['fname'] = fname
        label['text'] = f'Selected img file: {os.path.basename(fname)}.'


def get_out_dirname(label):
    fname = askdirectory(title='Output directory')

    if fname:
        state['paths']['out_dir'] = fname
        label['text'] = f'Output directory: {os.path.basename(fname)}.'


def slider_changed(label, img_widget, event):
    val = slider.get()
    val2 = slider_2.get()
    label['text'] = '{:.2f}'.format(val)
    print('val 1 =',val, ',val 2 = ',val2)
    arr = state['img_arr']
    arr = (arr.astype(float) * val).astype('uint8')

    obj = ImageTk.PhotoImage(Image.fromarray(arr))
    img_widget.image = obj
    img_widget['image'] = obj
    
def blur(param,img_widget, label, morph, gauss,canny_top,event):
    morph = state['settings']['morph'].get()
    morph = 2*morph+1
    gauss = state['settings']['gauss'].get()
    gauss = 2*gauss+1
    canny_bottom = 0
    canny_top = state['settings']['canny_top'].get()
    label['text'] = morph*(param==0)+gauss*(param==1)+canny_top*(param==2)
    
#     img_arr = cv.imread(state['fname'])
#     state['img_arr'] = img_arr
    src = state['img_arr']
    gr = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    bl=cv.GaussianBlur(src,(gauss,gauss),0)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (morph, morph))
    closed = cv.morphologyEx(bl, cv.MORPH_CLOSE, kernel)
    canny = cv.Canny(closed, canny_bottom, canny_top)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (morph, morph))
    closed = cv.morphologyEx(canny, cv.MORPH_CLOSE, kernel)
    contours0 = cv.findContours(closed.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
    (contours0, _) = contours.sort_contours(contours0)
    real_conts = []
    
    for cont in contours0:
        center, radius = cv.minEnclosingCircle(cont)
        if ((cv.contourArea(cont)>contour_area_threshold)&
            (center[0] >src.shape[1]//4)&(center[0] < 2*src.shape[1]//3)):
            sm = cv.arcLength(cont, True)
            apd = cv.approxPolyDP(cont, 0.02*sm, True)
            cv.drawContours(src, [cont], -1, (255,0,0), -2)
            real_conts.append(cont)
#     print('contours',len(real_conts))
    state['img_arr'] = src
    src = src.astype('uint8')
    src = imutils.resize(src, height=500)
    obj = ImageTk.PhotoImage(Image.fromarray(src))
    img_widget.image = obj
    img_widget['image'] = obj
    group_param = Morley.picture_params()
    Morley.picture_params.contour_params(group_param, morph,gauss,canny_bottom,canny_top)
    print(group_param.return_bl_params())
    
def color(img_widget, event):
    src = state['img_color_to_analys']
    hsv = cv.cvtColor(src, cv.COLOR_BGR2HSV )
    template = cv.imread(state['paths']['template_file'],0)
#     template = Morley.rotate_pic(template, rotate)  ### Надо сделать импорт ротейт значения и сделать его вперед ползунков
    w, h = template.shape[::-1]
    
    methods = ['cv.TM_CCOEFF_NORMED']
    for meth in methods:
        img_color_tmp = cv.imread(state['fname'],0)
#         img_color_tmp = rotate_pic(img_color_tmp, rotate)
        method = eval(meth)
        # Apply template Matching
        res = cv.matchTemplate(img_color_tmp,template,method)
        threshold = 0.55
        loc = np.where( res > threshold)
        numbers0=[]
        for pt in zip(*loc[::-1]):
            if (pt[0] > src.shape[1]/3)&((pt[0] < 2*src.shape[1]/3)):
                numbers0.append(pt[0])
    #             cv.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0,0,255), 5)

    numbers = pd.Series(numbers0)
    Mean = numbers.mean()
    Median = numbers.median()
    Mode = numbers.mode()[0]   
    mean_left_x = int(Mode)-w//4
    mean_right_x = int(Mode) + 3*w//4
    mean_left_x = round(mean_left_x)
    mean_right_x = round(mean_right_x)

    overlay = src.copy()
    cv.rectangle(overlay, (0,src.shape[0]), (mean_left_x, 0), (0,224,79), -1)
    opacity = 0.25
    cv.rectangle(overlay, (mean_right_x, src.shape[0]), (src.shape[1],0), (240,0,255), -5)
    cv.addWeighted(overlay, opacity, src, 1 - opacity, 0, src)
    
    h1 = state['color']['h_bottom'].get()
    s1 = state['color']['s_bottom'].get()
    v1 = state['color']['v_bottom'].get()
    h2 = state['color']['h_top'].get()
    s2 = state['color']['s_top'].get()
    v2 = state['color']['v_top'].get()
    
    # формируем начальный и конечный цвет фильтра
    h_min = np.array((h1, s1, v1), np.uint8)
    h_max = np.array((h2, s2, v2), np.uint8)

    # накладываем фильтр на кадр в модели HSV
    thresh = cv.inRange(hsv, h_min, h_max)
    
#     state['img_color_to_analys'] = src
    thresh = thresh.astype('uint8')
    thresh = imutils.resize(thresh, height=500)
    obj_color = ImageTk.PhotoImage(Image.fromarray(thresh))
    state['img_color_to_analys'] = src
    state['img_color_to_show'] = thresh
    img_widget.image = obj_color
    img_widget['image'] = obj_color

def seed(img_widget, event):
    h1 = state['seed']['h_bottom'].get()
    s1 = state['seed']['s_bottom'].get()
    v1 = state['seed']['v_bottom'].get()
    h2 = state['seed']['h_top'].get()
    s2 = state['seed']['s_top'].get()
    v2 = state['seed']['v_top'].get()
    
    h_min = np.array((h1, s1, v1), np.uint8)
    h_max = np.array((h2, s2, v2), np.uint8)
    
    src = state['img_seed_to_analys']
    hsv = cv.cvtColor(src, cv.COLOR_BGR2HSV )
    thresh = cv.inRange(hsv, h_min, h_max)
    thresh = thresh.astype('uint8')
    thresh = imutils.resize(thresh, height=500)
    obj_color = ImageTk.PhotoImage(Image.fromarray(thresh))
#     state['img_seed_to_analys'] = src
    state['img_seed_to_show'] = thresh
    img_widget.image = obj_color
    img_widget['image'] = obj_color
    
    
def rotation(w):
    window = tk.Toplevel(w)
    window.title('Rotate image')
    window.geometry('300x200')
    var=tk.IntVar()
    var.set(1)
    rad0 = tk.Radiobutton(window, text="90", variable=var, value=0)
    rad1 = tk.Radiobutton(window, text="180", variable=var, value=1)
    rad2 = tk.Radiobutton(window, text="270", variable=var, value=2)
    
    rad0.pack()
    rad1.pack()
    rad2.pack()
    state['rotation'] = var
    print(state['rotation'])




def clear(w): # clear all the wigets
    list = w.grid_slaves()
    for l in list:
        l.destroy()

def close_window (root): 
    root.destroy()
    
def tweak_image(w):
    
    window = tk.Toplevel(w)
    window.title('Tweak image')
    window.geometry('900x700')
    img_arr = cv.imread(state['fname'])
    img_arr_2 = imutils.resize(cv.imread(state['fname']), height=500)
    state['img_arr'] = img_arr
    state['img_color_to_analys'] = img_arr
    state['img_color_to_show'] = np.zeros_like(img_arr)
    state['img_seed_to_analys'] = img_arr
    state['img_seed_to_show'] = np.zeros_like(img_arr)
    
    def contours_tab():
        img_arr = state['img_arr']
        img_arr_2 = imutils.resize((img_arr), height=500)
        img_frame = tk.Frame(master=window)
        img_obj = ImageTk.PhotoImage(Image.fromarray(img_arr_2))
        img = tk.Label(master=img_frame, image=img_obj)
        img.image = img_obj  # https://stackoverflow.com/questions/23901168/how-do-i-insert-a-jpeg-image-into-a-python-tkinter-window#comment118214713_23905585
        img.pack(fill=tk.BOTH, expand=True)
        img_frame.pack()
        tweak_frame = tk.Frame(master=window)

        def colors_tab():
            clear(tweak_frame)
            img_frame.destroy()
            img_arr_color = state['img_color_to_show']
            img_arr_2_color = imutils.resize((img_arr_color), height=500)
            img_frame_color = tk.Frame(master=window)
            img_obj_color = ImageTk.PhotoImage(Image.fromarray(img_arr_2_color))
            img_color = tk.Label(master=img_frame_color, image=img_obj_color)
            img_color.image = img_obj_color
            img_color.pack(fill=tk.BOTH, expand=True)
            img_frame_color.pack()
            color_label = tk.Label(master=tweak_frame, text="Choosing color for pixel counting")
            color_label.grid(column= 0 , row=0)
            sliders_list = list(state['color'].keys())
            col = 0
            row = 0
            for i in sliders_list:
                l = tk.Label(master=tweak_frame, text=i)
                s = ttk.Scale(master=tweak_frame, from_ = 0, to = 255, value = state['color'][i].get(),
                                     variable=state['color'][i], command=partial(color,img_color))
                s.grid(column=col+1, row=(row//2)*2)
                l.grid(column=col+1, row=(row//2)*2+1)
                row = row+1
                col = (col+1)%2
                
            def back(img_frame_to_destroy, def_to_call):
                img_frame_to_destroy.destroy()
                clear(tweak_frame)
                def_to_call()

            def seeds_tab():
                button_b1.destroy()
                button_n2.destroy()
                img_frame_color.destroy()
                img_arr_seed = state['img_seed_to_show']
                img_arr_2_seed = imutils.resize((img_arr_seed), height=500)
                img_frame_seed = tk.Frame(master=window)
                img_obj_seed = ImageTk.PhotoImage(Image.fromarray(img_arr_2_seed))
                img_seed = tk.Label(master=img_frame_seed, image=img_obj_seed)
                img_seed.image = img_obj_seed
                img_seed.pack(fill=tk.BOTH, expand=True)
                img_frame_seed.pack()
                col = 0
                row = 0
                for i in sliders_list:
                    l = tk.Label(master=tweak_frame, text=i)
                    s = ttk.Scale(master=tweak_frame, from_ = 0, to = 255, value = state['seed'][i].get(),
                                         variable=state['seed'][i], command=partial(seed,img_seed))
                    s.grid(column=col+1, row=(row//2)*2)
                    l.grid(column=col+1, row=(row//2)*2+1)
                    row = row+1
                    col = (col+1)%2
                button_b2 = tk.Button(tweak_frame, text = 'Back', command = partial(back,img_frame_seed,colors_tab))
                button_end = tk.Button(tweak_frame, text = 'Exit') #, command = close_window(window)
                button_b2.grid(column=0, row=7)
                button_end.grid(column=6, row=7)
            button_b1 = tk.Button(tweak_frame, text = 'Back', command = partial(back,img_frame_color, contours_tab))
            button_n2 = tk.Button(tweak_frame, text = 'Next', command = seeds_tab)
            button_b1.grid(column=0, row=7)
            button_n2.grid(column=6, row=7)


        morph_label = tk.Label(master=tweak_frame, text="Choosing parameters for contour recognition")
        morph_label.grid(column= 0 , row=0)
        morph_slider_lbl = tk.Label(master=tweak_frame, text="morph")
        morph_slider = ttk.Scale(master=tweak_frame, from_ = 1, to = 5, value = state['settings']['morph'].get(),
                                 command=partial(blur,0,img ,morph_slider_lbl,state['settings']['morph'],
                                                 state['settings']['gauss'],state['settings']['canny_top']),
                                 variable=state['settings']['morph'])
        morph_slider.grid(column=1, row=4)
        morph_slider_lbl.grid(column=1, row=5)

        gauss_slider_lbl = tk.Label(master=tweak_frame, text="gauss")
        gauss_slider = ttk.Scale(master=tweak_frame, from_ = 1, to = 5, value = state['settings']['gauss'].get(),
                                 command=partial(blur,1,img ,gauss_slider_lbl,state['settings']['morph'],
                                                 state['settings']['gauss'],state['settings']['canny_top']),
                                 variable=state['settings']['gauss'])
        gauss_slider.grid(column=1, row=6)
        gauss_slider_lbl.grid(column=1, row=7)

        canny_top_slider_lbl = tk.Label(master=tweak_frame, text="Canny")
        canny_top_slider = ttk.Scale(master=tweak_frame, from_ = 0, to = 255, value = state['settings']['canny_top'].get(),
                                     command=partial(blur,2,img ,canny_top_slider_lbl, state['settings']['morph'],
                                                     state['settings']['gauss'],state['settings']['canny_top']),
                                     variable=state['settings']['canny_top'])
        canny_top_slider.grid(column=2, row=6)

        canny_top_slider_lbl.grid(column=2, row=7)

        button_n1 = tk.Button(tweak_frame, text = 'Next', command = colors_tab)
        button_n1.grid(column=6, row=7)
        
        
        
        

        tweak_frame.pack()
    contours_tab()
