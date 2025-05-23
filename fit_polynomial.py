
import pandas as pd
import numpy as np
from scipy.interpolate import lagrange

df = pd.read_csv('data.csv')  

def make_points(y):
    data = df[df[f'{y}_loss'] <= 0.001]
    xd = data['x'].to_numpy()
    yd = data[f'{y}'].to_numpy()
    return xd,yd
     

def make_poly_func(y):
    xd, yd= make_points(y)
    coeff = np.polyfit(xd, yd, 4)
    poly = np.poly1d(coeff)
    print(poly)

make_poly_func('cos')
make_poly_func('sin')
make_poly_func('tan')