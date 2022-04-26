import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import *
import numpy as np
import threading
import random
 
class SOM:
 
    # Function here computes the winning vector
    # by Euclidean distance
    def winner(self, weights, sample):
        D0 = 0
        D1 = 0
        D2 = 0
        for i in range(len(sample)):
            D0 = D0 + math.pow((sample[i] - weights[0][i]), 2)
            D1 = D1 + math.pow((sample[i] - weights[1][i]), 2)
            D2 = D2 + math.pow((sample[i] - weights[2][i]), 2)
        winner = np.array([D0,D1,D2])
        mymin = winner.min()
        min_positions = [i for i, x in enumerate(winner) if x == mymin]
        return min_positions[0]
 
    # Function here updates the winning vector
    def update(self, weights, sample, J, alpha):
        for i in range(len(weights[0])):
            weights[J][i] = weights[J][i] + alpha * (sample[i] - weights[J][i])
 
        return weights
 
def print_axis():
    global ax
    plt.xlim(-1.5,1.5)
    plt.ylim(-1.5,1.5)

def plot_point(event):
    ix, iy = event.xdata, event.ydata
    X.append((ix, iy))
    ax.plot(ix,iy, 'hk')
    canvas.draw()

def clean_screen():
    global ax, X
    ax.cla()
    print_axis()
    X = []
    canvas.draw()
 
def dataTraining():
    global X, weights, ob

    epochs = 300
    alpha = 0.2

    m = len(X)-1

    while epochs > 0:
        r = random.randint(0,m)
        sample = X[r]
        J = ob.winner(weights, sample)
        weights = ob.update(weights, sample, J, alpha)
        epochs = epochs - 1
        print("Epoch: ",epochs)

    print("Done!")

def dataClasification():
    global X, ax, weights, ob
    m = len(X)
    ax.cla()
    print_axis()
    for i in range(m):
        sample = X[i]
        J = ob.winner(weights, sample)
        if J == 0:
            ax.plot(X[i][0],X[i][1], 'hg')
        elif J == 1:
            ax.plot(X[i][0],X[i][1], 'hb')
        else:
            ax.plot(X[i][0],X[i][1], 'hr')
    canvas.draw()

X = []
weights = [[random.random(), random.random()], [random.random(), random.random()], [random.random(), random.random()]]
ob = SOM()

#Inizializamos la grafica de matplotlib
fig, ax= plt.subplots(facecolor='#8D96DA')
print_axis()
mainwindow = Tk()
mainwindow.geometry('580x580')
mainwindow.wm_title('SOM')

#Colocamos la grafica en la interfaz
canvas = FigureCanvasTkAgg(fig, master = mainwindow)
canvas.get_tk_widget().place(x=0, y=0, width=580, height=580)
fig.canvas.mpl_connect('button_press_event', plot_point)

start_button = Button(mainwindow, text="Train", command=lambda:threading.Thread(target=dataTraining).start())
start_button.place(x=230, y=25)

clean_button = Button(mainwindow, text="Clean", command=clean_screen)
clean_button.place(x=280, y=25)

clean_button = Button(mainwindow, text="Test", command=dataClasification)
clean_button.place(x=335, y=25)

#Mostramos la interfaz
mainwindow.mainloop()

