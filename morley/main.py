import tkinter as tk
from functools import partial
from . import gui



def main():
    window = tk.Tk()
    window.title('Plants GUI')
    window.geometry('900x600')

    # gui.state['settings']['slider'] = tk.DoubleVar(value=1.0)
    gui.set_state_variables(gui.state)
    gui.state['settings']['slider'] = tk.IntVar(value=1.0)

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

    files_frame = tk.Frame()

    raw_dir_lbl = tk.Label(master=files_frame, text="No directory selected", justify='left')
    get_raw_dir_btn = tk.Button(master=files_frame, text="Select image directory",
        command=partial(gui.get_image_dirname, raw_dir_lbl), width=20)


    get_raw_dir_btn.grid(column=0, row=0)  #pack(side=tk.LEFT, anchor=tk.E)
    raw_dir_lbl.grid(column=1, row=0)  #pack(side=tk.LEFT, padx=15, anchor=tk.W)

    template_lbl = tk.Label(master=files_frame, text="Template file not selected", justify='left')
    get_template_btn = tk.Button(master=files_frame, text="Select seed template", command=partial(gui.get_template_file, template_lbl), width=20)

    template_lbl.grid(row=1, column=1)
    get_template_btn.grid(row=1, column=0)  #pack(side=tk.TOP)

    paper_size = tk.Entry(text ="Paper size, mm^2")
    paper_size.grid(column = 1, row = 7)                 #.pack(anchor=tk.NW, padx=8, pady= 8)
#     gui.state['paper_size'] = paper_size.get()
#     print(paper_size.get())
    paper_size_btn = tk.Label(master=files_frame, text="Paper size", width=20)
    paper_size_btn.grid(column = 0, row = 9) 
    
    
    out_dir_lbl = tk.Label(master=files_frame, text="Output directory: " + gui.state['paths']['out_dir'], justify='left')
    get_out_dir_btn = tk.Button(master=files_frame, text="Select output directory",
        command=partial(gui.get_out_dirname, out_dir_lbl), width=20)

    get_out_dir_btn.grid(column=0, row=2)  #pack(side=tk.LEFT, anchor=tk.E)
    out_dir_lbl.grid(column=1, row=2)  #pack(side=tk.LEFT, padx=15, anchor=tk.W)


    files_frame.grid(column=0, row=0, columnspan=2, rowspan=3)

    status_frame = tk.Frame()
    status_frame.grid(column=2, row=0, rowspan=3)

    
    
    raw_file_lbl = tk.Label(master=files_frame, text="No file selected", justify='left')
    get_raw_file_btn = tk.Button(master=files_frame, text="Select image file",
        command=partial(gui.get_file, raw_file_lbl), width=20)


    get_raw_file_btn.grid(column=0, row=4)  #pack(side=tk.LEFT, anchor=tk.E)
    raw_file_lbl.grid(column=1, row=4)  #pack(side=tk.LEFT, padx=15, anchor=tk.W)
    
    
    
    tweak_file_lbl = tk.Label(master=files_frame, text="Tweaking image" , justify='left')
    tweak_file_btn = tk.Button(master=files_frame, text="Tweak image", command=partial(gui.tweak_image, window), width=20)
    tweak_file_btn.grid(column=0, row=5)  #pack(side=tk.LEFT, anchor=tk.E)
    tweak_file_lbl.grid(column=1, row=5)  #pack(side=tk.LEFT, padx=15, anchor=tk.W)
    
    
    rotation_file_lbl = tk.Label(master=files_frame, text="Rotating image" , justify='left')
    rotation_file_btn = tk.Button(master=files_frame, text="Rotate image", command=partial(gui.rotation, window), width=20)
    rotation_file_btn.grid(column=0, row=6)  #pack(side=tk.LEFT, anchor=tk.E)
    rotation_file_lbl.grid(column=1, row=6)  #pack(side=tk.LEFT, padx=15, anchor=tk.W)

    window.mainloop()

if __name__ == '__main__':
    main()
