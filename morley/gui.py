import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import askopenfilename
from PIL import ImageTk, Image
import cv2
import imutils

from functools import partial
import os

state = {
    'settings': {}
}

def get_image_filename(label):
    fname = askopenfilename(title='Image file')

    if fname:
        state['fname'] = fname
        label['text'] = f'Selected image: {os.path.basename(fname)}.'


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
