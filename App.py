import tkinter as tk
from idlelib.browser import file_open
from logging import raiseExceptions
from modulefinder import Module
from tkinter.filedialog import Open
from tkinter.filedialog import asksaveasfilename

import cv2 as cv
from typing import Union, Any

__all__ = ["MachineVisionApp"]


class MainWindow(tk.Frame):
    """
        MainWindow is a tkinter Frame that serves as the main interface for the application.

        Methods:
            open_image():
                Opens a file dialog to upload an image in color mode.
                Displays the selected image in a new OpenCV window.

            open_image_gray():
                Opens a file dialog to upload an image in grayscale mode.
                Displays the selected image in a new OpenCV window.
    """

    def __init__(self, parent, controller):
        super().__init__(parent)
        self.filename = None
        self.master = controller

        label = tk.Label(self, text="Main Page", font=("Segoe UI", 16))
        label.pack(pady=10, padx=10)

        open_image_gray = tk.Button(self, text="Upload Image", command=self.open_image)
        open_image_gray.pack(pady=10, padx=10)

        open_image_gray = tk.Button(self, text="Upload Image Gray", command=self.open_image_gray)
        open_image_gray.pack(pady=10, padx=10)

    def open_image(self):
        file_types = [('Images', '*.jpg *.tif *.bmp *.png *.jpeg *.webp')]
        dlg = Open(self, filetypes=file_types, title='Open Image')
        self.filename = dlg.show()

        if self.filename:
            # Check if the "Image In" window exists
            if cv.getWindowProperty('Image In', cv.WND_PROP_VISIBLE) >= 1:
                cv.destroyWindow('Image In')
            self.master.image = cv.imread(self.filename, cv.IMREAD_COLOR)
            cv.imshow('Image In', self.master.image)

    def open_image_gray(self):
        file_types = [('Images', '*.jpg *.tif *.bmp *.png *.jpeg *.webp')]
        dlg = Open(self, filetypes=file_types, title='Open Image Gray')
        self.filename = dlg.show()

        if self.filename:
            # Check if the "Image In" window exists
            if cv.getWindowProperty('Image In', cv.WND_PROP_VISIBLE) >= 1:
                cv.destroyWindow('Image In')
            self.master.image = cv.imread(self.filename, cv.IMREAD_GRAYSCALE)
            cv.imshow('ImageIn', self.master.image)


class ErrorWindow(tk.Frame):
    """
    ErrorWindow is a GUI component for displaying error messages using Tkinter.

    Methods:
        update_register():
            Updates the error label's text to display the current list of errors
            stored in the instance.
    """

    def __init__(self, parent, error: str = "No error"):
        super().__init__(parent)
        self.error = []
        self.error_text = tk.Label(self, text=error, font=("Segoe UI", 16))
        self.error_text.pack(pady=10, padx=10)

    def update_register(self):
        self.error_text.config(text="\n".join(self.error))


class MachineVisionApp(tk.Tk):
    """
    Machine Vision Main GUI Application, with included image transformation function
    access through Menu.
    """

    def __init__(self):
        super().__init__()
        # Class private variable
        self.image = None
        self.result_image = None

        self.title("Machine Vision Application")
        self.geometry("400x300")

        # Create a container for the frames
        container = tk.Frame(self)
        container.pack(fill=tk.BOTH, expand=True)

        # Main windows
        self.main_window = MainWindow(container, self)
        self.main_window.pack(fill=tk.BOTH, expand=True)

        # menu
        self.menu = tk.Menu(self)
        self.config(menu=self.menu)
        self.menus = {}

        # error register
        self.error_register = ErrorWindow(self)

    def transformers(self, module: Union[Module, any], group: str = "Image Processing"):
        try:
            # check if the given module have __all__ attribute
            getattr(module, '__all__')

            # Create new menu group if it not exist yet
            if group not in self.menus:
                self.menus[group] = tk.Menu(self.menu, tearoff=0)

            # Wrapper
            def define_callback(tf_: callable):
                def callback(image=self.image):
                    print(tf_.__name__)
                    self.result_image = tf_(self.image)
                    cv.imshow('Image out', self.result_image)

                return callback

            # Include function to the GUI Menu
            for function in module.__all__:
                transform_function = getattr(module, function)
                self.menus[group].add_command(label=transform_function.__name__,
                                              command=define_callback(transform_function))

            self.menu.add_cascade(label=group, menu=self.menus[group])

        except AttributeError:
            # If module not have the attribute, we should se that module name in the error windows (Frame)
            self.error_register.error.append(f"{module.__name__} not have attribute __all__, please define it!")
            self.error_register.update_register()
            self.error_register.tkraise()


if __name__ == "__main__":
    # Manual test application
    app = MachineVisionApp()
    app.mainloop()