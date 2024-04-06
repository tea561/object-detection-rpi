import threading
from tkinter import *
from PIL import Image, ImageTk

class App(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.start()

    def callback(self):
        self.root.quit()

    def run(self):
        self.root = Tk()
        self.root.protocol("WM_DELETE_WINDOW", self.callback)

        self.root.title("Object detection")
        self.root.geometry('600x600')

        self.root.mainloop()