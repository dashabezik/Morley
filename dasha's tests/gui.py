import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import askopenfilename, askdirectory
from PIL import ImageTk, Image
import cv2 as cv
import imutils
from imutils import contours

from functools import partial
import os

state = {
    'settings': {
        'morph': 0.0,
        'gauss':0.0,
        'canny_top': 255.0
    },
    'paths': {
        'out_dir': os.getcwd()
    }
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
    
def blur(param, label, morph, gauss,canny_top,event):
    morph = state['settings']['morph'].get()
    morph = 2*morph+1
    gauss = state['settings']['gauss'].get()
    gauss = 2*gauss+1
    canny_bottom = 0
    canny_top = state['settings']['canny_top'].get()
    label['text'] = morph*(param==0)+gauss*(param==1)+canny_top*(param==2)
    
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
    print('contours',len(real_conts))
    src = src.astype('uint8')
    src = imutils.resize(src, height=500)
    obj = ImageTk.PhotoImage(Image.fromarray(src))

def tweak_image(w):
    window = tk.Toplevel(w)
    window.title('Tweak image')
    window.geometry('900x600')


    img_frame = tk.Frame(master=window)
    img_arr = cv.imread(state['fname'])
    img_arr_2 = imutils.resize(cv.imread(state['fname']), height=500)
    state['img_arr'] = img_arr
    img_obj = ImageTk.PhotoImage(Image.fromarray(img_arr_2))
    img = tk.Label(master=img_frame, image=img_obj)
    img.image = img_obj  # https://stackoverflow.com/questions/23901168/how-do-i-insert-a-jpeg-image-into-a-python-tkinter-window#comment118214713_23905585

    img.pack(fill=tk.BOTH, expand=True)
    img_frame.pack()

    tweak_frame = tk.Frame(master=window)
    
    morph_slider_lbl = tk.Label(master=tweak_frame, text="morph")
    morph_slider = ttk.Scale(master=tweak_frame, from_ = 1, to = 5, command=partial(blur,0,morph_slider_lbl,
                                                                               state['settings']['morph'],state['settings']['gauss'],
                                                                                   state['settings']['canny_top']),
                       variable=state['settings']['morph'])
    morph_slider.grid(column=1, row=4)
    morph_slider_lbl.grid(column=1, row=5)
    
    gauss_slider_lbl = tk.Label(master=tweak_frame, text="gauss")
    gauss_slider = ttk.Scale(master=tweak_frame, from_ = 1, to = 5, command=partial(blur,1,gauss_slider_lbl,
                                                                               state['settings']['morph'],state['settings']['gauss'],
                                                                                   state['settings']['canny_top']),
                       variable=state['settings']['gauss'])
    gauss_slider.grid(column=1, row=6)
    gauss_slider_lbl.grid(column=1, row=7)

    canny_top_slider_lbl = tk.Label(master=tweak_frame, text="Canny")
    canny_top_slider = ttk.Scale(master=tweak_frame, from_ = 0, to = 255, value = 255,
                                 command=partial(blur,2,canny_top_slider_lbl, state['settings']['morph'],state['settings']['gauss'],
                                                 state['settings']['canny_top']), variable=state['settings']['canny_top'])
    canny_top_slider.grid(column=2, row=6)

    canny_top_slider_lbl.grid(column=2, row=7)
        
    
    
    
    tweak_frame.pack()
