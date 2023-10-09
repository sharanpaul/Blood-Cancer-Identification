import tkinter as tk
from tkinter import filedialog
from tkinter.filedialog import askopenfile
from PIL import Image, ImageTk
import base64
import urllib.request
import array
# Importing the libraries
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.models import *
from tensorflow.keras.preprocessing import image
import os
import numpy as np
import matplotlib.pyplot as plt


def upload_file():
    global img
    global filename
    f_types = [('Jpg Files', '*.jpg')]
    filename = filedialog.askopenfilename(filetypes=f_types)
    img = ImageTk.PhotoImage(file=filename)
    b2 =tk.Button(my_w,image=img)
    # using Button
    b2.grid(row=3,column=1, padx=5,pady=5)
    print(filename)


def predict():
    print('Detecting')
    ft=0
    st=0
    lt=0
    rt=0
    ut=0
    h=""
    out=""
    outv=5
    img = image.load_img(filename,target_size=(224,224))
    img = image.img_to_array(img,dtype='uint8')
    img = np.expand_dims(img,axis=0)
    ### flattening
    ypred1 = model1.predict_classes(img)
    ypred1=ypred1.round()
    ypred2 = model1.predict_classes(img)
    ypred2=ypred2.round()
    ypred3 = model1.predict_classes(img)
    ypred3=ypred3.round()
    ypred4 = model2.predict_classes(img)
    ypred4=ypred4.round()
    ypred5 = model2.predict_classes(img)
    ypred5=ypred5.round()
    ypred6 = model2.predict_classes(img)
    ypred6=ypred6.round()
    if ypred1[0]==0:
        ft+=1
    elif ypred1[0]==1:
        st+=1
    if ypred2[0]==0:
        ft+=1
    elif ypred2[0]==1:
        st+=1
    if ypred3[0]==0:
        ft+=1
    elif ypred3[0]==1:
        st+=1
    if ypred4[0]==0:
        ft+=1
    elif ypred4[0]==1:
        st+=1
    if ypred5[0]==0:
        ft+=1
    elif ypred5[0]==1:
        st+=1
    if ft>st:
        out = "There is no Blood Cancer Predicted"
        outv=0
    elif st>ft:
        out = "Blood Cancer predicted"
        outv=1
        ft=0
        st=0
        lt=0
        rt=0
        ut=0
    result.config(text=out)
    print(out)


model1 = load_model("mnn-1.h5")
model1.summary()
model2 = load_model("mnn-2.h5")
model2.summary()
print("success")
my_w = tk.Tk()
my_w.geometry("500x500")
# Size of the window
my_w.title('Blood Cancer Identification')
my_font1=('times', 18, 'bold')
l1 = tk.Label(my_w,text='Give Blood Smear Images',width=30,font=my_font1)
l1.grid(row=1,column=1)
b1 = tk.Button(my_w, text='Upload File', width=20,command = lambda:upload_file())
b1.grid(row=2,column=1, padx=5, pady=5)
b3 = tk.Button(my_w, text='Predict Output', width=20,command = lambda:predict())
b3.grid(row=6,column=1, padx=5, pady=5)

result = tk.Label(my_w, text="", font=my_font1)
result.grid(row=8, column=1)

my_w.mainloop()
