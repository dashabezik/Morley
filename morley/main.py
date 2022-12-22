import tkinter as tk
from tkinter import ttk
import tkinter.scrolledtext as st
import logging
from functools import partial
import os
import sys
import threading
import argparse
from . import gui, Morley, version

logger = logging.getLogger()


def start_gui():
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
    
    seed_margin_width = tk.Entry(master=main_frame, text="Germination threshold, mm", width=20, textvariable=gui.state['seed_margin_width'])
    seed_margin_width_lbl = tk.Label(master=main_frame, text="Seed margin width, %", width=28)

    report_area = st.ScrolledText(main_frame, width=60, height=20, state=tk.DISABLED)
    handler = gui.LoggingToGUI(report_area)
    formatter = logging.Formatter('{levelname:>8}: {asctime} {message}',
                datefmt='[%H:%M:%S]', style='{')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    logger.addHandler(stream_handler)
    stream_handler.setFormatter(formatter)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    pb = ttk.Progressbar(main_frame, orient=tk.HORIZONTAL, mode='determinate', length=100, variable=gui.state['progress'])
    pb_lbl = gui.FormatLabel(main_frame, textvariable=gui.state['progress'], format='{}%')

    worker = threading.Thread(target=Morley.search, name='morley-worker')
    run_btn = tk.Button(master=main_frame, text="RUN", command=worker.start, width=20)

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
    
    seed_margin_width_lbl.grid(column=1, row=10)
    seed_margin_width.grid(column=0, row=10)
    
    run_btn.grid(column=1, row=11)

    report_area.grid(column=2, row=0, rowspan=11)
    pb.grid(column=2, row=12)
    pb_lbl.grid(column=2, row=13)

    for col in range(3):
        main_frame.columnconfigure(col, weight=1)

    window.rowconfigure(0, weight=1)
    window.columnconfigure(0, weight=1)
    window.mainloop()
    if worker.is_alive():
        worker.join()
    sys.exit()


def main():
    if len(sys.argv) == 1:
        return start_gui()

    formatter = logging.Formatter('{levelname:>8}: {asctime} {message}',
                datefmt='[%H:%M:%S]', style='{')
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    logger.addHandler(stream_handler)
    stream_handler.setFormatter(formatter)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logger.setLevel(logging.DEBUG)

    parser = argparse.ArgumentParser(description="Morley CLI: run a Morley analysis in headless mode. ",
        epilog="To use a GUI instead, run morley without any arguments.")
    parser.add_argument('config', help="A JSON file with settings. You can obtain one by saving settings from GUI mode. "
        "Other arguments override the values in config.")
    parser.add_argument('-i', '--input', help="Input directory.")
    parser.add_argument('-t', '--template', help="Template file.")
    parser.add_argument('-o', '--output-dir', help="Output directory.")
    parser.add_argument('-r', '--rotation', help='Input photo rotation.', type=int, choices=[0, 90, 180, 270])
    parser.add_argument('-a', '--paper-area', help="Paper area, mm^2.", type=int)
    parser.add_argument('-g', '--threshold', help="Germination threshold, mm.", type=int)
    args = parser.parse_args()
    gui.load_state_headless(args.config)
    if args.input:
        gui.state['paths']['input'] = args.input
    if args.template:
        gui.state['paths']['template_file'] = args.template
    if args.output_dir:
        gui.state['paths']['out_dir'] = args.output_dir
    if args.rotation:
        gui.state['rotation'].set(args.rotation)
    if args.paper_area:
        gui.state['paper_area'].set(args.paper_area)
    if args.threshold:
        gui.state['germ_thresh'].set(args.threshold)

    gui.conditions.check_conditions()

    missing = [c for c in gui.conditions.CONDITIONS['run'] if c not in gui.conditions.satisfied]
    if missing:
        logger.error("The following parameters are missing: %s", ', '.join(missing))
        sys.exit(1)
    Morley.search()


if __name__ == '__main__':
    main()
