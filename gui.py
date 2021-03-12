import tensorflow as tf
import tkinter as tk
from tkinter import *
import win32gui
from PIL import ImageGrab, ImageGrab
import numpy as np

model = tf.keras.models.load_model('aj_mnist.h5')

def predict_digit(img):
    #resize image to 28x28 pixels
    img = img.resize((28,28))

    #convert rgb to grayscale
    img = img.convert('L')
    img = np.array(img)
    img = img
    #reshaping to support our model input and normalizing
    img = img.reshape(1,28,28,1)

    #predicting the class
    res = model.predict([img])
    a = {"a":np.argmax(res),"b":np.max(res)}
    return a["a"], a['b']

class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)

        self.x = self.y = 0

        #creating elements
        self.canvas = tk.Canvas(self, width = 448, height= 448, bg= 'white', cursor='cross')
        self.label = tk.Label(self, text = 'Thinking..', font = ('Helvetica', 48))
        self.classify_btn = tk.Button(self, text = 'Recognize', command = self.classify_handwriting)
        self.button_clear = tk.Button(self, text = 'Clear', command = self.clear_all)

        #grid structure
        self.canvas.grid(row= 0, column = 0, pady = 2, sticky=W)
        self.label.grid(row= 0, column = 1,pady = 2,padx =2)
        self.classify_btn.grid(row= 1,column = 1, pady=2,padx=2)
        self.button_clear.grid(row = 1,column = 0,pady=2)

        #self cavas bind
        self.canvas.bind("<B1-Motion>", self.draw_lines)
    
    #function definitions
    def clear_all(self):
        self.canvas.delete("all")

    def classify_handwriting(self):
        #get the handle of the canvas
        HWND = self.canvas.winfo_id()

        rect = win32gui.GetWindowRect(HWND)
        im = ImageGrab.grab(rect)

        digit, acc = predict_digit(im)
        self.label.configure(text = str(digit)+', '+str(int(acc*100))+'%')

    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y 
        r = 8
        self.canvas.create_oval(self.x-r, self.y-r, self.x+r, self.y+r, fill='black')   

app = App()
mainloop()

#testing commit