import io

from tkinter import *

import numpy as np
from PIL import Image


class Sketchpad(Canvas):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.bind("<Button-1>", self.save_start)
        self.bind("<B1-Motion>", self.draw_line)

    def save_start(self, event):
        self.last_x, self.last_y = event.x, event.y

    def draw_line(self, event):
        self.create_line(self.last_x, self.last_y, event.x, event.y, width=40, capstyle=ROUND)
        self.save_start(event)

    def get_content(self, width, height):
        self.update()
        ps = self.postscript(colormode="gray")
        image = Image.open(io.BytesIO(ps.encode('utf-8')))
        image = image.convert("L")
        image = image.resize((width, height))
        output = np.array(image)
        output = output.reshape(width * height)
        output = 255 - output
        return output

