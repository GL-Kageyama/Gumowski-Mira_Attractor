#======================================================================================
#--------------------------    Gumowski-Mira Attractor    -----------------------------
#======================================================================================

#----  value_G1 = X * mu + 2 * X * X * (1 - mu) / (1 + X * X)                      ----
#----  new_X = a * Y * (1 - b * Y * Y) + Y + value_G1                              ----

#----  value_G2 = new_X * mu + 2 * new_X * new_X * (1 - mu) / (1 + new_X * new_X)  ----
#----  new_Y = -X + value_G2                                                       ----

#======================================================================================

import numpy as np
import pandas as pd
import panel as pn
import datashader as ds
from numba import jit
from datashader import transfer_functions as tf
from colorcet import palette_n

#--------------------------------------------------------------------------------------

ps = {k:p[::-1] for k, p in palette_n.items()}

pn.extension()

#--------------------------------------------------------------------------------------

@jit(nopython=True)
def Gumowski(x, mu):
    g = x*mu + 2*x*x*(1 - mu) / (1 + x*x)
    return g

@jit(nopython=True)
def GumowskiMira_trajectory(a, b, mu, x0, y0, n):
    x, y = np.zeros(n), np.zeros(n)
    x[0], y[0] = x0, y0
    
    for i in np.arange(n-1):

        x[i+1] = a*y[i]*(1 - b*y[i]*y[i]) + y[i] + Gumowski(x[i], mu)

        y[i+1] = -x[i] + Gumowski(x[i+1], mu)
    
    return x, y

#--------------------------------------------------------------------------------------

def GumowskiMira_plot(a=0.01, b=0.05, mu=-0.8, n=100000, colormap=ps['bgyw']):
    
    cvs = ds.Canvas(plot_width=800, plot_height=800)
    x, y = GumowskiMira_trajectory(a, b, mu, 0, 0.5, n)
    agg = cvs.points(pd.DataFrame({'x':x, 'y':y}), 'x', 'y')
    
    return tf.shade(agg, cmap=colormap)

#--------------------------------------------------------------------------------------

pn.interact(GumowskiMira_plot, n=(1,1000000))

#--------------------------------------------------------------------------------------

# The value of this attractor can be changed freely.
# Try it in the jupyter notebook.

