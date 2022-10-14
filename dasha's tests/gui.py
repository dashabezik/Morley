import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import askopenfilename, askdirectory
from PIL import ImageTk, Image
import cv2
import imutils

from functools import partial
import os

state = {
    'settings': {
        'slider': 1.0
    },
    'paths': {
        'out_dir': os.getcwd()
    }
}

morph_v = 1
gauss = 1

def set_state_variables(d):
    for k, v in d.items():
        if isinstance(v, dict):
            set_state_variables(v)
        if isinstance(v, float):
            d[k] = tk.DoubleVar(value=v)
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
    
def morph(label,var):
    morph_v = var
    label['text'] = var
    print('morph =',var)


def tweak_image(w):
    window = tk.Toplevel(w)
    window.title('Tweak image')
    window.geometry('900x600')


    img_frame = tk.Frame(master=window)
    img_arr = imutils.resize(cv2.imread(state['fname']), height=500)
    state['img_arr'] = img_arr
    img_obj = ImageTk.PhotoImage(Image.fromarray(img_arr))
    img = tk.Label(master=img_frame, image=img_obj)
    img.image = img_obj  # https://stackoverflow.com/questions/23901168/how-do-i-insert-a-jpeg-image-into-a-python-tkinter-window#comment118214713_23905585

    img.pack(fill=tk.BOTH, expand=True)
    img_frame.pack()

    tweak_frame = tk.Frame(master=window)
    slider_lbl = tk.Label(master=tweak_frame, text="Tweaking 1")
    slider = ttk.Scale(master=tweak_frame, from_ = 1, to = 11, command=partial(morph,slider_lbl))
    slider.grid(column=1, row=4)

    slider_lbl.grid(column=1, row=5)
    
    slider_lbl_2 = tk.Label(master=tweak_frame, text="Tweaking 2")
    slider_2 = ttk.Scale(master=tweak_frame, command=partial(slider_changed, slider_lbl_2, img))
    slider_2.grid(column=1, row=6)

    slider_lbl_2.grid(column=1, row=7)
    
    tweak_frame.pack()
