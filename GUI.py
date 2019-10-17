import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras import Model
import os
import datetime

from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label


class MammographyApp(App):

    def build(self):
        superbox = BoxLayout(orientation='vertical')

        imagebox = BoxLayout(orientation='horizontal')
        label1 = Label(text="Images")
        imagebox.add_widget(label1)

        buttonbox = BoxLayout(orientation='vertical')
        label2 = Label(text="Buttons")
        buttonbox.add_widget(label2)

        superbox.add_widget(imagebox)
        superbox.add_widget(buttonbox)

        #superbox.add_widget(FigureCanvasKivyAgg(plt.gcf()))

        return superbox

# if-statement to not run application when imported
if __name__ == "__main__":
    MammographyApp().run()