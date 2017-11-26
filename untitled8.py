#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 16:39:47 2017

@author: aditya
"""
import math
w1=3.0
w2=-3.0
w3=1.0
w4=-4.0
w5=-3.0
w6=5.0
x1=0
x2=1
H1 = (1/(1+math.exp(-float(w1*x1 +w2*x2 + 0.5))))
H2 = (1/(1+math.exp(-float(w3*x1 +w4*x2 + 0.5))))
result = (1/(1+math.exp(-float(w5*H1 +w6*H2 + 0.5))))
print (result)
