# -*- coding: utf-8 -*-
"""
@author: Alan Poulos
"""

print('Computing correlation coefficients')
exec(open("computeCorrelations.py").read())

print('Fitting regression model')
exec(open("regressionAnalysis.py").read())

print('Making figures')
exec(open("figures.py").read())