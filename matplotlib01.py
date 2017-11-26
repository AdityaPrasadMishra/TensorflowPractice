#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 21:56:57 2017

@author: aditya
"""

import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(0,5,11)
y= x**2
z=x**3
#plt.plot(x,y,'r')
#plt.xlabel("X label")
#plt.ylabel("Y label")
#plt.title("Title")

#plt.subplot(1,2,1)
#plt.plot(x,y,'r')
#plt.subplot(1,2,2)
#plt.plot(y,x,'b')

#fig = plt.figure()
#axes = fig.add_axes([0.1,0.1,0.8,0.8])
#axes.plot(x,y)
#axes.set_xlabel('X label')
#axes.set_ylabel('Y label')

#axes.set_title('Title')
#axes1 = fig.add_axes([0.1,0.1,0.8,0.8])
#axes1.set_title("Smaller Plot")
#axes1.plot(x,y)

#
#axes2 = fig.add_axes([0.45,0.20,0.4,0.3])
#axes1.set_title("Larger Plot")
#axes2.plot(x,y)

fig,axes = plt.subplots(nrows=1,ncols=2)
plt.tight_layout()
#for cAxis in axes:
#    cAxis.plot(x,y)
axes[0].plot(x,y)
axes[0].plot(x,z)
axes[1].plot(y,z)
    
#axes.plot(x,y)
plt.show()