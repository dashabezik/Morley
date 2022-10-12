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


def get_out_dirname(label):
    fname = askdirectory(title='Output directory')

    if fname:
        state['paths']['out_dir'] = fname
        label['text'] = f'Output directory: {os.path.basename(fname)}.'


def slider_changed(label, img_widget, event):
    val = state['settings']['slider'].get()
    label['text'] = '{:.2f}'.format(val)

    arr = state['img_arr']
    arr = (arr.astype(float) * val).astype('uint8')

    obj = ImageTk.PhotoImage(Image.fromarray(arr))
    img_widget.image = obj
    img_widget['image'] = obj


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
    slider_lbl = tk.Label(master=tweak_frame)
    slider = ttk.Scale(master=tweak_frame, command=partial(slider_changed, slider_lbl, img), variable=state['settings']['slider'])
    slider.pack()

    slider_lbl.pack()
    tweak_frame.pack()
    window.bind('<Motion>', partial(track_mouse, img))


def track_mouse(widget, event):
    arr = state['img_arr'].copy()
    if event.widget == widget:
        arr[:, event.x, :] = 255
        obj = ImageTk.PhotoImage(Image.fromarray(arr))
        widget.image = obj
        widget['image'] = obj
