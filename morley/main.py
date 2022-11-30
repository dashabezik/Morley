import tkinter as tk
from tkinter import ttk
import tkinter.scrolledtext as st
from functools import partial
import os, sys
from . import gui, Morley

# sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# import setup

def main():
    window = tk.Tk()
    path_to_icon =os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logo.png') 
    window.tk.call('wm', 'iconphoto', window._w, tk.PhotoImage(file=path_to_icon))
    # Add image file
    background_image=tk.PhotoImage(path_to_icon)
    background_label = tk.Label(window, image=background_image)
    background_label.place(x=0, y=0, relwidth=1, relheight=1)
    background_label.image = background_image
    
    window.title('Morley GUI 0.0.3')
    window.geometry('900x400')

    gui.set_state_variables(gui.state)

    # https://stackoverflow.com/a/54068050/1258041
    try:
        try:
            window.tk.call('tk_getOpenFile', '-foobarbaz')
        except tk.TclError:
            pass

        window.tk.call('set', '::tk::dialog::file::showHiddenBtn', '1')
        window.tk.call('set', '::tk::dialog::file::showHiddenVar', '0')
    except:
        pass

    main_frame = tk.Frame(master=window)

    raw_dir_lbl = tk.Label(master=main_frame, text="No directory selected", justify='left')
    get_raw_dir_btn = tk.Button(master=main_frame, text="Select image directory",
        command=partial(gui.get_image_dirname, raw_dir_lbl), width=20)

    template_lbl = tk.Label(master=main_frame, text="Template file not selected", justify='left')
    get_template_btn = tk.Button(master=main_frame, text="Select seed template", command=partial(gui.get_template_file, template_lbl), width=20)

    out_dir_lbl = tk.Label(master=main_frame, text="Output directory: " + gui.state['paths']['out_dir'], justify='left')
    get_out_dir_btn = tk.Button(master=main_frame, text="Select output directory",
        command=partial(gui.get_out_dirname, out_dir_lbl), width=20)

    tweak_lbl = tk.Label(master=main_frame, text="Tweaking image", justify='left')
    tweak_btn = tk.Button(master=main_frame, text="Tweak image", command=partial(gui.tweak_image, window), width=20)

    rotation_lbl = tk.Label(master=main_frame, text="Rotating image" , justify='left')
    rotation_btn = tk.Button(master=main_frame, text="Rotate image", command=partial(gui.rotation, window), width=20)

    paper_size = tk.Entry(master=main_frame, width=20, textvariable=gui.state['paper_area'])
    paper_size_lbl = tk.Label(master=main_frame, text="Paper size, mm^2", width=20)

    germ_threshold = tk.Entry(master=main_frame, text="Germination threshold, mm", width=20, textvariable=gui.state['germ_thresh'])
    germ_threshold_lbl = tk.Label(master=main_frame, text="Germination threshold, mm", width=28)

    report_area = st.ScrolledText(main_frame, width=40, height=20)

    report_area.insert(tk.END,'... LOGGING WINDOW ... \n')
    report_area.update()

    pb = ttk.Progressbar(main_frame, orient=tk.HORIZONTAL, mode='determinate', length=100)
    pb_lbl = tk.Label(main_frame, text='0%')

    run_btn = tk.Button(master=main_frame, text="RUN", command=partial(
        Morley.search, main_frame, report_area, pb, pb_lbl), width=20)

    gui.conditions.register({
        'rotate': [rotation_btn],
        'run': [run_btn],
        'tweak': [tweak_btn]
        })
    for name in ['germ_thresh', 'paper_area']:
        gui.state[name].trace('w', gui.trace_entry(name))
        gui.state[name].set(gui.state[name].get())  # trigger trace functions

    main_frame.grid(sticky=tk.N+tk.E+tk.S+tk.W)
    get_raw_dir_btn.grid(column=0, row=0)
    raw_dir_lbl.grid(column=1, row=0)

    template_lbl.grid(row=1, column=1)
    get_template_btn.grid(row=1, column=0)

    get_out_dir_btn.grid(column=0, row=2)
    out_dir_lbl.grid(column=1, row=2)

    rotation_btn.grid(column=0, row=5)
    rotation_lbl.grid(column=1, row=5)

    tweak_btn.grid(column=0, row=6)
    tweak_lbl.grid(column=1, row=6)

    paper_size_lbl.grid(column=1, row=7)
    paper_size.grid(column=0, row=7)

    germ_threshold_lbl.grid(column=1, row=8)
    germ_threshold.grid(column=0, row=8)
    run_btn.grid(column=1, row=9)

    report_area.grid(column=2, row=0, rowspan=9)
    pb.grid(column=2, row=10)
    pb_lbl.grid(column=2, row=11)

    for col in range(3):
        main_frame.columnconfigure(col, weight=1)

    window.rowconfigure(0, weight=1)
    window.columnconfigure(0, weight=1)
    window.mainloop()


if __name__ == '__main__':
    main()
