import tkinter as tk
from tkinter.filedialog import askopenfilename, askdirectory, asksaveasfilename
from PIL import ImageTk, Image
import cv2 as cv
import imutils
from imutils import contours
import numpy as np
import pandas as pd
import os
import random
from functools import partial
import json
import logging


state = {
    'settings': {
        'morph': 5,
        'gauss': 3,
        'canny_top': 160
    },
    'color': {
        'h_bottom': 50,
        'h_top': 255,
        's_bottom': 0,
        's_top': 255,
        'v_bottom': 100,
        'v_top': 255,
    },
    'roots': {
        'h_bottom': 0,
        'h_top': 255,
        's_bottom': 0,
        's_top': 255,
        'v_bottom': 0,
        'v_top': 255,
    },
    'leaves': {
        'h_bottom': 0,
        'h_top': 255,
        's_bottom': 0,
        's_top': 255,
        'v_bottom': 0,
        'v_top': 255,
    },
    'seed': {
        'h_bottom': 0,
        'h_top': 20,
        's_bottom': 100,
        's_top': 255,
        'v_bottom': 100,
        'v_top': 255,
    },
    'paths': {
        'out_dir': os.getcwd()
    },
    'rotation': 0,
    'paper_area_thresold': 5000,
    'paper_area': 0,
    'germ_thresh': 10,
    'progress': 0,
    'seed_margin_width':100
}

STATE_SYNTAX_VERSION = 1
STATE_SYNTAX_VERSION_KEY = 'syntax_version'
CONTOUR_AREA_THRESHOLD = 1000
FORMAT='{:.0f}'

class FormatLabel(tk.Label):

    def __init__(self, master=None, cnf={}, **kw):

        # default values
        self._format = FORMAT
        self._textvariable = None
        self.two_n_plus1 = None

        # get new format and remove it from `kw` so later `super().__init__` doesn't use them (it would get error message)
        if 'format' in kw:
            self._format = kw['format']
            del kw['format']

        # get `textvariable` to assign own function which set formatted text in Label when variable change value
        if 'textvariable' in kw:
            self._textvariable = kw['textvariable']
            self._trace_id = self._textvariable.trace('w', self._update_text)
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

    def destroy(self):
        self._textvariable.trace_vdelete('w', self._trace_id)
        super().destroy()


class LoggingToGUI(logging.Handler):
    # https://stackoverflow.com/a/18194597/1258041
    def __init__(self, console):
        logging.Handler.__init__(self)
        self.console = console

    def emit(self, message):
        formattedMessage = self.format(message)

        self.console.configure(state=tk.NORMAL)
        self.console.insert(tk.END, formattedMessage + '\n')
        self.console.configure(state=tk.DISABLED)
        self.console.see(tk.END)


def save_state(fname):
    state_dict = pythonize_state_dict()
    for k in ['template', 'img_arr', 'img_arr_0', 'progress']:
        state_dict.pop(k, None)
    state_dict[STATE_SYNTAX_VERSION_KEY] = STATE_SYNTAX_VERSION
    with open(fname, 'w') as f:
        json.dump(state_dict, f)


def update_dict(old, new):
    for k, v in new.items():
        if isinstance(v, dict):
            update_dict(old[k], v)
        elif k in old and hasattr(old[k], 'set'):
            old[k].set(v)
        else:
            old[k] = v


def load_state(fname, label_dict):
    with open(fname) as f:
        d = json.load(f)
    update_dict(state, d)
    set_state_variables(state)
    update_labels(label_dict)
    conditions.check_conditions()
    conditions.update_conditions()


def load_state_headless(fname):
    set_state_variables(state, headless=True)
    with open(fname) as f:
        d = json.load(f)
    update_dict(state, d)


def update_labels(label_dict):
    inp = state.get('paths', {}).get('input')
    if inp:
        set_label('input', label_dict['input'], inp)

    template = state.get('paths', {}).get('template_file')
    if template:
        read_template_file(template, label_dict['template'])

    outdir = state.get('paths', {}).get('out_dir')
    if outdir:
        set_label('outdir', label_dict['outdir'], outdir)

    rotate = state.get('rotation')
    if rotate:
        set_label('rotation', label_dict['rotation'], rotate)


def set_label(kind, label, value):
    mapping = {
        'input': lambda fname: f'Selected image directory: {os.path.basename(fname)}.',
        'template': lambda fname: f'Selected seed template: {os.path.basename(fname)}.',
        'outdir': lambda fname: f'Output directory: {os.path.basename(fname)}.',
        'rotation': lambda angle: f'Rotation angle: {angle.get()}.'
    }
    label['text'] = mapping[kind](value)


def load_state_dialog(label_dict):
    fname = askopenfilename(title="Load settings...", filetypes=[('JSON files', '*.json'), ('All files', '*')])
    if fname:
        load_state(fname, label_dict)


def save_state_dialog():
    fname = asksaveasfilename(title="Save settings as...", defaultextension='.json')
    if fname:
        save_state(fname)


class ConditionManager:
    CONDITIONS = {
        'rotate': ['input'],
        'tweak': ['input', 'template'],
        'run': ['input', 'template', 'paper_area', 'germ_thresh']
    }
    def __init__(self):
        self.satisfied = set()

    def register(self, widgets):
        self.widgets = widgets
        for k, w in widgets.items():
            if k in self.CONDITIONS:
                for widget in w:
                    widget['state'] = tk.DISABLED

    def satisfies(self, condition):
        """Decorator that makes a function call satisfy the corresponding condition.
        Changes the status of widgets after a successful call."""
        def decorator(func):
            def wrapped(*args, **kwargs):
                ret = func(*args, **kwargs)
                if ret:
                    self.satisfied.add(condition)
                    self.update_conditions()
                return ret
            return wrapped
        return decorator

    def update_conditions(self):
        for key, v in self.CONDITIONS.items():
            if all(w in self.satisfied for w in v) and key in self.widgets:
                status = tk.NORMAL
            else:
                status = tk.DISABLED
            for w in self.widgets[key]:
                w['state'] = status

    def check_conditions(self):
        if state.get('paths', {}).get('input'):
            self.satisfied.add('input')
        if state.get('paths', {}).get('template_file'):
            self.satisfied.add('template')
        if state.get('paper_area').get():
            self.satisfied.add('paper_area')
        if state.get('germ_thresh').get():
            self.satisfied.add('germ_thresh')


conditions = ConditionManager()


def trace_entry(key):
    def callback(name, index, op):
        try:
            assert state[key].get() != 0
        except (AssertionError, tk.TclError):
            if key in conditions.satisfied:
                conditions.satisfied.remove(key)
        else:
            conditions.satisfied.add(key)
        conditions.update_conditions()
    return callback


def random_file(path_to_file_folder):
    a=random.choice(os.listdir(path_to_file_folder))
    while (a=='template')|(a=='.ipynb_checkpoints')|(not os.path.isdir(os.path.join(path_to_file_folder,a))):
        a=random.choice(os.listdir(path_to_file_folder))
    path_to_file = os.path.join(path_to_file_folder, a+'/')
    b = random.choice(os.listdir(path_to_file))
    while (b=='.ipynb_checkpoints'):
        b=random.choice(os.listdir(path_to_file))
    path_to_file = os.path.join(path_to_file, b)
    return path_to_file


def set_state_variables(d, headless=False):
    if not headless:
        factory = tk.IntVar
    else:
        factory = PseudoIntVar

    for k, v in d.items():
        if isinstance(v, dict):
            set_state_variables(v, headless)
        if isinstance(v, (int, float)):
            d[k] = factory(value=v)
        # if isinstance(v, str):
        #     d[k] = tk.StringVar(value=v)


class PseudoIntVar:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


def pythonize_state_dict(d=None):
    out = {}
    if d is None:
        d = state
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = pythonize_state_dict(v)
        elif hasattr(v, 'get'):
            out[k] = v.get()
        else:
            out[k] = v
    return out


@conditions.satisfies('input')
def get_image_dirname(label):
    fname = askdirectory(title='Raw image directory')
    if fname:
        state['paths']['input'] = fname
        set_label('input', label, fname)
    return fname


def read_template_file(fname, label):
    state['template'] = cv.imdecode(np.fromfile(fname, dtype=np.uint8), cv.IMREAD_GRAYSCALE)
    # IMREAD_GRAYSCALE has 0 enum of imread modes
    set_label('template', label, fname)


@conditions.satisfies('template')
def get_template_file(label):
    fname = askopenfilename(title='Seed template file')
    if fname:
        state['paths']['template_file'] = fname
        read_template_file(fname, label)
    return fname


def get_out_dirname(label):
    fname = askdirectory(title='Output directory')
    if fname:
        state['paths']['out_dir'] = fname
        set_label('outdir', label, fname)
    return fname


def blur(img_widget, event):
    morph = state['settings']['morph'].get()
    gauss = state['settings']['gauss'].get()
    canny_bottom = 0
    canny_top = state['settings']['canny_top'].get()

    src = state['img_arr'].copy()

    bl = cv.GaussianBlur(src, (gauss, gauss), 0)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (morph, morph))
    closed = cv.morphologyEx(bl, cv.MORPH_CLOSE, kernel)
    canny = cv.Canny(closed, canny_bottom, canny_top)
    closed = cv.morphologyEx(canny, cv.MORPH_CLOSE, kernel)
    contours0 = cv.findContours(closed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
    (contours0, _) = contours.sort_contours(contours0)
    real_conts = []
    src = cv.cvtColor(src, cv.COLOR_BGR2RGB)

    for cont in contours0:
        center, radius = cv.minEnclosingCircle(cont)
        if ((cv.contourArea(cont) > CONTOUR_AREA_THRESHOLD) and
                (center[0] > src.shape[1] // 4) & (center[0] < 2 * src.shape[1] // 3)):
            cv.drawContours(src, [cont], -1, (255, 0, 0), -2)
            real_conts.append(cont)

    src = src.astype('uint8')
    src = imutils.resize(src, height=500)
    obj = ImageTk.PhotoImage(Image.fromarray(src))
    img_widget.image = obj
    img_widget['image'] = obj


def color(img_widget, hsv, event):

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

    thresh = thresh.astype('uint8')
    thresh = imutils.resize(thresh, height=420)
    obj_color = ImageTk.PhotoImage(Image.fromarray(thresh))
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
    thresh = imutils.resize(thresh, height=420)
    obj_color = ImageTk.PhotoImage(Image.fromarray(thresh))
    img_widget.image = obj_color
    img_widget['image'] = obj_color


def rotate_pic(img, rotate, resize=False):
    rotate_dict = {
        90: cv.ROTATE_90_CLOCKWISE,
        180: cv.ROTATE_180,
        270: cv.ROTATE_90_COUNTERCLOCKWISE}

    if rotate:
        img = cv.rotate(img, rotate_dict[rotate])
    if resize:
        img = imutils.resize(img, height=300)
    return img


def choose_rotation(angle, img, img_widget,label):
    img = rotate_pic(img, angle, True)
    i = ImageTk.PhotoImage(Image.fromarray(img))
    img_widget.image = i
    img_widget['image'] = i
    set_label('rotation', label, state['rotation'])


def rotation(w, label):
    window = tk.Toplevel(w)
    window.title('Rotate image')
    window.geometry('600x400')
    control_frame = tk.Frame(master=window)

    text_lbl = tk.Label(control_frame, text='Choose the angle to rotate your photos, as in the example')
    img_frame = tk.Frame(master=window)

    path_to_ethalon = os.path.dirname(os.path.abspath(__file__))
    ethalon = cv.imdecode(np.fromfile(os.path.join(path_to_ethalon, 'ethalon.png'), dtype=np.uint8), cv.IMREAD_COLOR)
#     ethalon = cv.imread(os.path.join(path_to_ethalon, 'ethalon.png'))
    ethalon = cv.cvtColor(ethalon, cv.COLOR_BGR2RGB)
    ethalon = ethalon.astype('uint8')
    ethalon = imutils.resize(ethalon, height=300)
    obj = ImageTk.PhotoImage(Image.fromarray(ethalon))

    test_img = cv.imdecode(np.fromfile(random_file(state['paths']['input']), dtype=np.uint8), cv.IMREAD_COLOR)
#     test_img = cv.imread(random_file(state['paths']['input']))
    test_img = cv.cvtColor(test_img, cv.COLOR_BGR2RGB)
    test_img = test_img.astype('uint8')
    test_img = imutils.resize(test_img, height=300)
    obj2 = ImageTk.PhotoImage(Image.fromarray(test_img))

    maxsize = max(test_img.shape[:2])
    window.geometry(f'{maxsize * 2}x{maxsize + 100}')

    img1 = tk.Label(master=img_frame, width=maxsize, height=maxsize)
    img1.image = obj
    img1['image'] = obj

    img2 = tk.Label(master=img_frame, width=maxsize, height=maxsize)
    img2.image = obj2
    img2['image'] = obj2
    choose_rotation(state['rotation'].get(), test_img, img2,label)

    buttons = []
    for i, val in enumerate([0, 90, 180, 270]):
        buttons.append(tk.Radiobutton(control_frame, text=str(val), command=lambda j=val: choose_rotation(j, test_img, img2, label),
                                      variable=state['rotation'], value=val))

    set_button = tk.Button(window, text='Save', command=window.destroy)
    img_frame.grid(column=0, row=0, sticky=tk.W + tk.E)
    control_frame.grid(column=0, row=1)

    img1.grid(column=0, row=0)
    img2.grid(column=1, row=0)

    text_lbl.grid(column=0, row=0, columnspan=4)
    for i, b in enumerate(buttons):
        b.grid(column=i, row=1)
    set_button.grid(column=0, row=2, columnspan=4)
    window.columnconfigure(0, weight=1)


def clear(w): # clear all the wigets
    for c in w.grid_slaves():
        c.destroy()


def set_params(parameter):
    for i in state[parameter]:
        state[parameter][i] = state['color'][i].get()


def add_color_sliders(d, frame, command, startrow=1):
    for i, param in enumerate('hsv'):
        row = startrow + 2 * i
        name = tk.Label(master=frame, text=param)
        name.grid(column=0, row=row, rowspan=2)
        for j, suffix in enumerate(['bottom', 'top']):
            var = d[f'{param}_{suffix}']
            label = tk.Label(master=frame, textvariable=var)
            slider = tk.Scale(master=frame, from_=0, to=255, variable=var, orient='horizontal', resolution=1, showvalue=False, command=command)
            slider.grid(column=j + 1, row=row)
            label.grid(column=j + 1, row=row + 1)
    return row + 1


def seeds_tab(img, tweak_frame):
    clear(tweak_frame)
    seed(img, None)

    color_label = tk.Label(master=tweak_frame, text="Choosing color for seed excluding")
    color_label.grid(column=0, row=0, columnspan=3)
    row = 1 + add_color_sliders(state['seed'], tweak_frame, partial(seed, img))
    button_b2 = tk.Button(tweak_frame, text='Back', command=partial(colors_tab, img, tweak_frame))
    button_end = tk.Button(tweak_frame, text='Done', command=tweak_frame.winfo_toplevel().destroy)
    button_b2.grid(column=0, row=row)
    button_end.grid(column=2, row=row)


def colors_tab(img, tweak_frame):
    src = state['img_arr'].copy()
    template = state['template']
    template = rotate_pic(template, state['rotation'].get())
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

    clear(tweak_frame)
    color(img, hsv, None)

    color_label = tk.Label(master=tweak_frame, text="Choosing color for pixel counting")
    color_label.grid(column=0, row=0, columnspan=3)
    row = add_color_sliders(state['color'], tweak_frame, partial(color, img, hsv)) + 1

    button_b1 = tk.Button(tweak_frame, text='Set sprouts', command=partial(set_params, 'leaves'))
    button_n2 = tk.Button(tweak_frame, text='Set roots', command=partial(set_params, 'roots'))
    button_b1.grid(column=0, row=row)
    button_n2.grid(column=2, row=row)
    button_b1 = tk.Button(tweak_frame, text='Back', command=partial(contours_tab, img, tweak_frame))
    button_n2 = tk.Button(tweak_frame, text='Next', command=partial(seeds_tab, img, tweak_frame))
    button_b1.grid(column=0, row=row+1)
    button_n2.grid(column=2, row=row+1)


def contours_tab(img, tweak_frame):
    clear(tweak_frame)
    blur(img, None)

    morph_label = tk.Label(master=tweak_frame, text="Choosing parameters for contour recognition")
    morph_label.grid(column=0, row=0, columnspan=3)
    morph_slider_lbl = tk.Label(master=tweak_frame, textvariable=state['settings']['morph'])
    morph_name_lbl = tk.Label(master=tweak_frame, text="morph:")
    morph_slider = tk.Scale(master=tweak_frame, from_=1, to=13, resolution=2, orient='horizontal', showvalue=False, command=partial(blur, img),
                             variable=state['settings']['morph'])
    morph_name_lbl.grid(column=0, row=1)
    morph_slider.grid(column=1, row=1)
    morph_slider_lbl.grid(column=2, row=1)

    gauss_slider_lbl = tk.Label(master=tweak_frame, textvariable=state['settings']['gauss'])
    gauss_name_lbl = tk.Label(master=tweak_frame, text='gauss:')
    gauss_slider = tk.Scale(master=tweak_frame, from_=1, to=13, resolution=2, orient='horizontal', showvalue=False, command=partial(blur, img),
                             variable=state['settings']['gauss'])
    gauss_name_lbl.grid(column=0, row=2)
    gauss_slider.grid(column=1, row=2)
    gauss_slider_lbl.grid(column=2, row=2)

    canny_top_slider_lbl = tk.Label(master=tweak_frame, textvariable=state['settings']['canny_top'])
    canny_top_name_lbl = tk.Label(master=tweak_frame, text='canny_top:')
    canny_top_slider = tk.Scale(master=tweak_frame, from_=0, to=255, orient='horizontal', showvalue=False, command=partial(blur, img),
                                 variable=state['settings']['canny_top'])
    canny_top_name_lbl.grid(column=0, row=3)
    canny_top_slider.grid(column=1, row=3)
    canny_top_slider_lbl.grid(column=2, row=3)

    button_n1 = tk.Button(tweak_frame, text='Next', command=partial(colors_tab, img, tweak_frame))
    button_n1.grid(column=2, row=4)


def tweak_image(w):
    window = tk.Toplevel(w)
    window.title('Tweak image')
    window.geometry('900x650')

    file_name = random_file(state['paths']['input'])
    img_arr = cv.imdecode(np.fromfile(file_name, dtype=np.uint8), cv.IMREAD_COLOR) ## IMREAD_UNCHANGED has -1 enum of imread modes
    img_arr = rotate_pic(img_arr, state['rotation'].get())
    img_arr_0 = cv.imdecode(np.fromfile(file_name, dtype=np.uint8), cv.IMREAD_GRAYSCALE)## IMREAD_GRAYSCALE has 0 enum of imread modes
    img_arr_0 = rotate_pic(img_arr_0, state['rotation'].get())
    state['img_arr'] = img_arr
    state['img_arr_0'] = img_arr_0

    img_frame = tk.Frame(master=window)
    img = tk.Label(master=img_frame)
    img.pack(fill=tk.BOTH, expand=True)
    img_frame.pack()
    tweak_frame = tk.Frame(master=window)
    tweak_frame.pack()

    contours_tab(img, tweak_frame)
