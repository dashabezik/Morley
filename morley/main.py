import tkinter as tk
from tkinter import ttk
import tkinter.scrolledtext as st
import tklogging
import logging
from functools import partial
import os
import sys
import threading
from . import gui, Morley, version

logger = tklogging.getLogger('')


def main():
    window = tk.Tk()
    path_to_icon = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logo.png')
    window.tk.call('wm', 'iconphoto', window._w, tk.PhotoImage(file=path_to_icon))
    # Add image file
    background_image = tk.PhotoImage(path_to_icon)
    background_label = tk.Label(window, image=background_image)
    background_label.place(x=0, y=0, relwidth=1, relheight=1)
    background_label.image = background_image

    window.title(f'Morley GUI v{version.version}')
    window.geometry('960x430')

    gui.set_state_variables(gui.state, window)

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

    save_btn = tk.Button(master=main_frame, text="Save settings", command=gui.save_state_dialog)
    tweak_lbl = tk.Label(master=main_frame, text="Set recognition parameters: \n blur and color ranges", justify='left')
    tweak_btn = tk.Button(master=main_frame, text="Recognition settings", command=partial(gui.tweak_image, window), width=20)

    rotation_lbl = tk.Label(master=main_frame, text="Rotation angle: " + str(gui.state['rotation'].get()), justify='left')
    rotation_btn = tk.Button(master=main_frame, text="Rotate image", command=partial(gui.rotation, window, rotation_lbl), width=20)

    label_dict = {'input': raw_dir_lbl, 'template': template_lbl, 'outdir': out_dir_lbl, 'rotation': rotation_lbl}
    load_btn = tk.Button(master=main_frame, text="Load settings", command=partial(gui.load_state_dialog, label_dict))


    paper_size = tk.Entry(master=main_frame, width=20, textvariable=gui.state['paper_area'])
    paper_size_lbl = tk.Label(master=main_frame, text="Paper size, mm^2", width=20)

    germ_threshold = tk.Entry(master=main_frame, text="Germination threshold, mm", width=20, textvariable=gui.state['germ_thresh'])
    germ_threshold_lbl = tk.Label(master=main_frame, text="Germination threshold, mm", width=28)

    report_area = st.ScrolledText(main_frame, width=60, height=20, state=tk.DISABLED)
    Handler = tklogging.get_handler(report_area)

    log_t = threading.Thread(target=tklogging.socket_listener_worker,
        args=(logger, logging.handlers.DEFAULT_TCP_LOGGING_PORT, Handler),
        name='morley-listener')
    log_t.start()

    pb = ttk.Progressbar(main_frame, orient=tk.HORIZONTAL, mode='determinate', length=100, variable=gui.state['progress'])
    pb_lbl = gui.FormatLabel(main_frame, textvariable=gui.state['progress'], format='{}%')

    run_t = threading.Thread(target=Morley.search, name='morley-worker')
    run_t.daemon = True

    run_btn = tk.Button(master=main_frame, text="RUN", command=run_t.start, width=20)

    gui.conditions.register({
        'rotate': [rotation_btn],
        'run': [run_btn],
        'tweak': [tweak_btn]
        })
    for name in ['germ_thresh', 'paper_area']:
        gui.state[name].trace('w', gui.trace_entry(name))
        gui.state[name].set(gui.state[name].get())  # trigger trace functions

    main_frame.grid(sticky=tk.N+tk.E+tk.S+tk.W)
    get_raw_dir_btn.grid(column=0, row=1)
    raw_dir_lbl.grid(column=1, row=1)

    template_lbl.grid(row=2, column=1)
    get_template_btn.grid(row=2, column=0)

    get_out_dir_btn.grid(column=0, row=3)
    out_dir_lbl.grid(column=1, row=3)

    rotation_btn.grid(column=0, row=6)
    rotation_lbl.grid(column=1, row=6)
    load_btn.grid(column=0, row=0)

    tweak_btn.grid(column=0, row=7)
    tweak_lbl.grid(column=1, row=7)
    save_btn.grid(column=1, row=0)

    paper_size_lbl.grid(column=1, row=8)
    paper_size.grid(column=0, row=8)

    germ_threshold_lbl.grid(column=1, row=9)
    germ_threshold.grid(column=0, row=9)
    run_btn.grid(column=1, row=10)

    report_area.grid(column=2, row=0, rowspan=10)
    pb.grid(column=2, row=11)
    pb_lbl.grid(column=2, row=12)

    for col in range(3):
        main_frame.columnconfigure(col, weight=1)

    window.rowconfigure(0, weight=1)
    window.columnconfigure(0, weight=1)
    window.mainloop()

    tklogging.tklogging.tcpserver.abort = 1
    tklogging.tklogging.tcpserver.server_close()
    sys.exit()


if __name__ == '__main__':
    main()
