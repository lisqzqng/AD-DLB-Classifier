import numpy as np
import math
import time 
from processing.inv_exp import inv_exp

def InnerProd_Q(q1,q2):
    val = (q1.dot(q2.T)).trace()
    return val

def inner_prod_on_sphere(TX,TY):
    return None

def distance_on_sphere(X, Y):
    in_prod = InnerProd_Q(X,Y)
    if abs(in_prod) > 1:
        in_prod = 1
    return np.arccos(in_prod)

def exp(V,X):
    theta = np.trace(np.matmul(V,V.T))
    exp = np.cos(theta) * X + (np.sin(theta) / theta) * V 
    return exp

def pole_ladder (source, target,  V0):
    V = -inv_exp(target, midPoint(exp_map(source, V0), midPoint(source, target,1/2), 2))

    return V

def schilds_ladder (source, target,  V0):
    V = inv_exp(target,midPoint(source,midPoint(exp_map(source,V0),target, 1/2),2))

    return V


def exp_map(skeleton, X):
    sk = np.array(skeleton).dot(np.array(skeleton).T)
    tr = sk.trace()
    teta = np.sqrt(tr)
    expMap = ((math.cos(teta)) * X) + ((math.sin(teta) / teta) * skeleton)

    return expMap

def midPoint(x, y, t):
    m = exp_map(x, np.multiply(inv_exp(x, y), t))
    return m
