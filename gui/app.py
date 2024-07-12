import pickle

import numpy as np

from tkinter import *

from sketchpad import Sketchpad
from labeled_progressbar import LabeledProgressbar

from neural_network.utils import fixed_normalization


class Sidebar:
    def __init__(self, parent):
        self.frame = Frame(parent)
        self.frame.grid(row=0, column=1, sticky=W+E+N+S, padx=3, pady=5)

        self.frame_top = Frame(self.frame)
        self.frame_top.pack(fill=BOTH, expand=True)

        self.prob_vars = []
        for i in range(10):
            Label(self.frame_top, text=f"{i}:", font=("Consolas", 10)).grid(row=i, column=0)
            self.prob_vars.append(DoubleVar(value=0.0))
            LabeledProgressbar(self.frame_top, maximum=1, variable=self.prob_vars[i]).grid(row=i, column=1, pady=3)

        self.frame_bottom = Frame(self.frame)
        self.frame_bottom.pack(fill=X)

        self.predict_btn = Button(self.frame_bottom, text="Predict")
        self.predict_btn.pack(fill=X, expand=True)

        self.clear_btn = Button(self.frame_bottom, text="Clear")
        self.clear_btn.pack(fill=X, expand=True)


class MainView:
    def __init__(self, parent):
        self.frame = Frame(parent, width=560, height=560)
        self.frame.grid(row=0, column=0)

        self.canvas = Sketchpad(self.frame, width=560, height=560, bg="#DDDDDD", borderwidth=0, highlightthickness=0)
        self.canvas.pack()


class App:
    def __init__(self, root):
        with open("../model.pickle", "rb") as fd:
            self.nn = pickle.load(fd)

        self.root = root
        self.root.resizable(False, False)
        self.view = MainView(self.root)
        self.sidebar = Sidebar(self.root)

        self.sidebar.predict_btn.configure(command=self.on_predict_button_click)
        self.sidebar.clear_btn.configure(command=self.on_clear_button_click)

    def on_predict_button_click(self):
        content = self.view.canvas.get_content(28, 28)
        content = fixed_normalization(content, 0, 255, 0, 1)
        result = np.round(self.nn.predict(content), 2).squeeze()
        for var, res in zip(self.sidebar.prob_vars, result):
            var.set(res)

    def on_clear_button_click(self):
        self.view.canvas.delete("all")
        prob_vars = self.sidebar.prob_vars
        for var in prob_vars:
            var.set(0.0)
