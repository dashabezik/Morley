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
    'rotation': 1,
}

CONTOUR_AREA_THRESHOLD = 1000
FORMAT='{:.0f}'


class FormatLabel(tk.Label):

    def __init__(self, master=None, cnf={}, **kw):

        # default values
        self._format = FORMAT
        self._textvariable = None

        # get new format and remove it from `kw` so later `super().__init__` doesn't use them (it would get error message)
        if 'format' in kw:
            self._format = kw['format']
            del kw['format']

        # get `textvariable` to assign own function which set formatted text in Label when variable change value
        if 'textvariable' in kw:
            self._textvariable = kw['textvariable']
            self._textvariable.trace('w', self._update_text)
            del kw['textvariable']

        # run `Label.__init__` without `format` and `textvariable`
        super().__init__(master, cnf={}, **kw)

        # update text after running `Label.__init__`
        if self._textvariable:
            #self._update_text(None, None, None)
            self._update_text(self._textvariable, '', 'w')

    def _update_text(self, a, b, c):
        """update text in label when variable change value"""
        self["text"] = self._format.format(self._textvariable.get())


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
        state['template'] = cv.imread(fname, 0)
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


def blur(img_widget, event):
    morph = state['settings']['morph'].get()
    morph = 2 * morph + 1
    gauss = state['settings']['gauss'].get()
    gauss = 2 * gauss + 1
    canny_bottom = 0
    canny_top = state['settings']['canny_top'].get()

    src = state['img_arr'].copy()
    # print('sum:', src.sum())
    # gr = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    bl = cv.GaussianBlur(src, (gauss, gauss), 0)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (morph, morph))
    closed = cv.morphologyEx(bl, cv.MORPH_CLOSE, kernel)
    canny = cv.Canny(closed, canny_bottom, canny_top)
    # kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (morph, morph))
    closed = cv.morphologyEx(canny, cv.MORPH_CLOSE, kernel)
    contours0 = cv.findContours(closed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
    (contours0, _) = contours.sort_contours(contours0)
    real_conts = []

    for cont in contours0:
        center, radius = cv.minEnclosingCircle(cont)
        if ((cv.contourArea(cont) > CONTOUR_AREA_THRESHOLD) and
                (center[0] > src.shape[1] // 4) & (center[0] < 2 * src.shape[1] // 3)):
            # sm = cv.arcLength(cont, True)
            # apd = cv.approxPolyDP(cont, 0.02 * sm, True)
            cv.drawContours(src, [cont], -1, (255, 0, 0), -2)
            real_conts.append(cont)

    src = src.astype('uint8')
    src = imutils.resize(src, height=500)
    obj = ImageTk.PhotoImage(Image.fromarray(src))
    img_widget.image = obj
    img_widget['image'] = obj


def color(img_widget, event):
    src = state['img_arr'].copy()
    # hsv = cv.cvtColor(src, cv.COLOR_BGR2HSV)
    template = state['template']  # TODO: Morley.rotate_pic(template, rotate)
    w, h = template.shape[::-1]

    method = cv.TM_CCOEFF_NORMED
    res = cv.matchTemplate(state['img_arr_0'], template, method)
    threshold = 0.55
    loc = np.where(res > threshold)
    numbers0 = []
    for pt in zip(*loc[::-1]):
        if (pt[0] > src.shape[1] / 3) and (pt[0] < 2 * src.shape[1] / 3):
            numbers0.append(pt[0])

    numbers = pd.Series(numbers0)
    mode = numbers.mode()[0]
    mean_left_x = int(mode) - w // 4
    mean_right_x = int(mode) + 3 * w // 4
    mean_left_x = round(mean_left_x)
    mean_right_x = round(mean_right_x)

    overlay = src.copy()
    cv.rectangle(overlay, (0, src.shape[0]), (mean_left_x, 0), (0,224,79), -1)
    opacity = 0.25
    cv.rectangle(overlay, (mean_right_x, src.shape[0]), (src.shape[1], 0), (240, 0, 255), -5)
    cv.addWeighted(overlay, opacity, src, 1 - opacity, 0, src)
    hsv = cv.cvtColor(src, cv.COLOR_BGR2HSV)

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
    #     state['img_color_to_analys'] = src
    # state['img_color_to_show'] = thresh
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

    src = state['img_arr'].copy()
    hsv = cv.cvtColor(src, cv.COLOR_BGR2HSV)
    thresh = cv.inRange(hsv, h_min, h_max)
    thresh = thresh.astype('uint8')
    thresh = imutils.resize(thresh, height=500)
    obj_color = ImageTk.PhotoImage(Image.fromarray(thresh))
    img_widget.image = obj_color
    img_widget['image'] = obj_color


def rotation(w):
    window = tk.Toplevel(w)
    window.title('Rotate image')
    window.geometry('300x200')
    var=tk.IntVar()
    var.set(1)
    rad0 = tk.Radiobutton(window, text="90", variable=var, value=90)
    rad1 = tk.Radiobutton(window, text="180", variable=var, value=180)
    rad2 = tk.Radiobutton(window, text="270", variable=var, value=270)
    def get_val(label):
        label['text'] = var.get()
    l = tk.Label(window, text = 'initial')
    b = tk.Button(window, text ='Set Value', command = partial(get_val,l))
    l.pack()
    b.pack()
    rad0.pack()
    rad1.pack()
    rad2.pack()
    state['rotation'] = var.get()
    print(state['rotation'])


def clear(w): # clear all the wigets
    for c in w.grid_slaves():
        c.destroy()


def add_color_sliders(d, frame, command, startrow=1):
    for i, param in enumerate('hsv'):
        row = startrow + 2 * i
        name = tk.Label(master=frame, text=param)
        name.grid(column=0, row=row, rowspan=2)
        for j, suffix in enumerate(['bottom', 'top']):
            var = d[f'{param}_{suffix}']
            label = FormatLabel(master=frame, textvariable=var)
            slider = ttk.Scale(master=frame, from_=0, to=255, variable=var, command=command)
            slider.grid(column=j + 1, row=row)
            label.grid(column=j + 1, row=row + 1)
    return row + 1


def seeds_tab(img, tweak_frame):
    clear(tweak_frame)
    seed(img, None)

    row = 1 + add_color_sliders(state['seed'], tweak_frame, partial(seed, img))
    button_b2 = tk.Button(tweak_frame, text='Back', command=partial(colors_tab, img, tweak_frame))
    button_end = tk.Button(tweak_frame, text='Done', command=lambda: tweak_frame.winfo_toplevel().destroy())
    button_b2.grid(column=0, row=row)
    button_end.grid(column=2, row=row)


def colors_tab(img, tweak_frame):
    clear(tweak_frame)
    color(img, None)

    color_label = tk.Label(master=tweak_frame, text="Choosing color for pixel counting")
    color_label.grid(column=0, row=0, columnspan=3)
    row = add_color_sliders(state['color'], tweak_frame, partial(color, img))

    button_b1 = tk.Button(tweak_frame, text='Back', command=partial(contours_tab, img, tweak_frame))
    button_n2 = tk.Button(tweak_frame, text='Next', command=partial(seeds_tab, img, tweak_frame))
    button_b1.grid(column=0, row=row)
    button_n2.grid(column=2, row=row)


def contours_tab(img, tweak_frame):
    clear(tweak_frame)
    blur(img, None)

    morph_label = tk.Label(master=tweak_frame, text="Choosing parameters for contour recognition")
    morph_label.grid(column=0, row=0, columnspan=3)
    morph_slider_lbl = FormatLabel(master=tweak_frame, textvariable=state['settings']['morph'])
    morph_name_lbl = tk.Label(master=tweak_frame, text="morph:")
    morph_slider = ttk.Scale(master=tweak_frame, from_=1, to=5, command=partial(blur, img),
                             variable=state['settings']['morph'])
    morph_name_lbl.grid(column=0, row=1)
    morph_slider.grid(column=1, row=1)
    morph_slider_lbl.grid(column=2, row=1)

    gauss_slider_lbl = FormatLabel(master=tweak_frame, textvariable=state['settings']['gauss'])
    gauss_name_lbl = tk.Label(master=tweak_frame, text='gauss:')
    gauss_slider = ttk.Scale(master=tweak_frame, from_=1, to=5, command=partial(blur, img),
                             variable=state['settings']['gauss'])
    gauss_name_lbl.grid(column=0, row=2)
    gauss_slider.grid(column=1, row=2)
    gauss_slider_lbl.grid(column=2, row=2)

    canny_top_slider_lbl = FormatLabel(master=tweak_frame, textvariable=state['settings']['canny_top'])
    canny_top_name_lbl = tk.Label(master=tweak_frame, text='canny_top:')
    canny_top_slider = ttk.Scale(master=tweak_frame, from_=0, to=255, command=partial(blur, img),
                                 variable=state['settings']['canny_top'])
    canny_top_name_lbl.grid(column=0, row=3)
    canny_top_slider.grid(column=1, row=3)
    canny_top_slider_lbl.grid(column=2, row=3)

    button_n1 = tk.Button(tweak_frame, text='Next', command=partial(colors_tab, img, tweak_frame))
    button_n1.grid(column=2, row=4)


def tweak_image(w):
    window = tk.Toplevel(w)
    window.title('Tweak image')
    window.geometry('900x700')
    img_arr = cv.imread(state['fname'])
    img_arr_0 = cv.imread(state['fname'], 0)
    state['img_arr'] = img_arr  #.copy()
    state['img_arr_0'] = img_arr_0
    state['img_resized'] = ImageTk.PhotoImage(Image.fromarray(imutils.resize(img_arr, height=500)))
    state['img_mask'] = np.zeros_like(img_arr)

    img_frame = tk.Frame(master=window)
    img = tk.Label(master=img_frame)
    img.pack(fill=tk.BOTH, expand=True)
    img_frame.pack()
    tweak_frame = tk.Frame(master=window)
    tweak_frame.pack()

    contours_tab(img, tweak_frame)
