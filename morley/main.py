import tkinter as tk
from functools import partial
from . import gui



def main():
    window = tk.Tk()
    window.title('Plants GUI')
    window.geometry('600x200')

    gui.state['settings']['slider'] = tk.DoubleVar(value=1.0)
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

    frame = tk.Frame()

    selected_file_lbl = tk.Label(master=frame, text="No file selected", justify='left')

    get_file_btn = tk.Button(master=frame, text="Select image file",
        command=partial(gui.get_image_filename, selected_file_lbl), width=20)


    get_file_btn.pack(side=tk.LEFT, anchor=tk.E)
    selected_file_lbl.pack(side=tk.LEFT, padx=15, anchor=tk.W)

    frame.pack(side=tk.TOP)

    tweak_file_btn = tk.Button(master=window, text="Tweak image", command=partial(gui.tweak_image, window), width=20)
    tweak_file_btn.pack(side=tk.TOP)

    window.mainloop()


if __name__ == '__main__':
    main()
