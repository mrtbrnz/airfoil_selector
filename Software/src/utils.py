# Utility functions
import os
import pickle
from numpy import genfromtxt
import numpy as np
from scipy import interpolate

def get_name(filename):
    return filename.split('.dat', 1)[0]

def pload(*f_names):
    """Pickle load"""
    f_name = os.path.join(*f_names)
    with open(f_name, "rb") as f:
        pickle_dict = pickle.load(f)
    return pickle_dict

def pdump(pickle_dict, *f_names):
    """Pickle dump"""
    f_name = os.path.join(*f_names)
    with open(f_name, "wb") as f:
        pickle.dump(pickle_dict, f)
        
def airfoil_as_feature(filename, nr_points=100, extra_derivatives=True):
    ''' Converts airfoil geometry into a feature trajectory '''
    
    data = genfromtxt(filename, delimiter='', skip_header=1, skip_footer=0)

    x_coor = data[:,0]
    y_coor = data[:,1]

    indx = np.argmin(x_coor)
    xu, yu = x_coor[:indx], y_coor[:indx]
    xd, yd = x_coor[indx:], y_coor[indx:]

    xu[np.argmin(xu)]=0.0
    xu[np.argmax(xu)]=1.0

    xd[np.argmin(xd)]=0.0
    xd[np.argmax(xd)]=1.0

    # dist = np.linspace(-1,1,nr_points) # Equal spacing
    dist = (-1+np.cos(np.linspace(0,np.pi,nr_points)))/-2. # Cosine Spacing

    fu = interpolate.interp1d(xu, yu)
    fd = interpolate.interp1d(xd, yd)

    yu = fu(dist)
    yd = fd(dist)

    # dyu = np.hstack([0,np.diff(yu)])
    # dyd = np.hstack([0,np.diff(yd)])
    # dyu = np.gradient(yu, xu)
    # dyd = np.gradient(yd, xd)
    dyu = np.gradient(yu)
    dyd = np.gradient(yd)
    
    if extra_derivatives:
        return np.hstack([10*yu, 10*yd, 10*dyu, 10*dyd])
    else:
        return np.hstack([10*yu, 10*yd])
