import numpy as np
import math
import time 
from processing.inv_exp import inv_exp
from processing.product import *
#Some of the functions here are experimental and might not deliver the expected results. 


def new_PTG(X0, V0, Y0): # on the tangent space
    s_t = inv_exp(X0,Y0) + inv_exp(Y0,X0)
    frob_sq = distance_on_sphere(X0,Y0)
    in_prod = InnerProd_Q(V0,inv_exp(Y0,X0))
    V = V0 - 2 * ( in_prod / frob_sq ) * (s_t)
    return V


"""
def new_inv_exp(X,Y):
    dist = distance_on_sphere(X,Y)
    upper = Y - InnerProd_Q(X,Y)*X
    V =  upper * (dist/ np.linalg.norm(upper,'fro'))


#def inv_exp(X, Y):    

    skeleton=X.dot(Y.T)
    tr=abs(skeleton.trace())
    #if np.isnan(tr):
    #    print('tr is nan')
    if tr>1:
        tr=1
    teta_invexp=math.acos(tr)
    if (math.sin(teta_invexp)<0.0001):
        teta_invexp=0.1

    invExp=(teta_invexp/math.sin(teta_invexp)) * (Y - (math.cos(teta_invexp))*X)     
    np_inv = np.array(invExp)

    return invExp
"""



def new_PTG2(X0, V, Y0): #on the manifold
    s_t = exp_map(X0,X0) + exp_map(Y0,Y0)  
    frob_sq = np.square(np.linalg.norm(exp_map(X0,X0) + exp_map(Y0,Y0)))
    in_prod = InnerProd_Q(V,Y0)
    V -= 2 * ( in_prod / frob_sq ) * (s_t)
    return V


def new_PTG1(source, V, ref): 
    s_t = source + ref
    frob_sq = np.square(np.linalg.norm(source + ref))
    in_prod = InnerProd_Q(V,ref)
    V -= 2 * ( in_prod / frob_sq ) * (s_t)
    return V


