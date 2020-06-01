from scipy.misc import derivative as dx
import sympy as sym
import numpy as np
import math

'''
This scipt contains the various gradient-descent methods
that are ready for matplotlib visualization
'''


def fx(val, x, y, z):
    return x * val**3 + y * val**2 + z * val


def fx_sym(val, x, y, z):
    return x * val**3 + y * val**2 + z * val


def gradient_descent(alpha, x_vals, theta0_val, theta1_val, theta2_val, y_vals):
    x, y, z = sym.symbols('x y z')
    gradsum_theta0 = 0
    gradsum_theta1 = 0
    gradsum_theta2 = 0
    for i, x_val in enumerate(x_vals):
        cost_func = (fx_sym(x_val, x, y, z) - y_vals[i])**2
        der_theta0 = sym.diff(cost_func, x)
        der_theta1 = sym.diff(cost_func, y)
        der_theta2 = sym.diff(cost_func, z)
        
        grad_theta0 = der_theta0.evalf(subs={x: theta0_val, y: theta1_val, z: theta2_val})
        grad_theta1 = der_theta1.evalf(subs={x: theta0_val, y: theta1_val, z: theta2_val})
        grad_theta2 = der_theta2.evalf(subs={x: theta0_val, y: theta1_val, z: theta2_val})

        gradsum_theta0 += grad_theta0 
        gradsum_theta1 += grad_theta1 
        gradsum_theta2 += grad_theta2
    
    grad_theta0 = alpha * gradsum_theta0 / len(x_vals)
    grad_theta1 = alpha * gradsum_theta1 / len(x_vals)
    grad_theta2 = alpha * gradsum_theta2 / len(x_vals)
    
    return grad_theta0, grad_theta1, grad_theta2
