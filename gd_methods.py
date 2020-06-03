from scipy.misc import derivative as dx
import sympy as sym
import numpy as np
import math

'''
This scipt contains the various gradient-descent methods
that are ready for matplotlib visualization
'''
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-10

def fx(val, x, y, z):
    return x * val**3 + y * val**2 + z * val


def fx_sym(val, x, y, z):
    return x * val**3 + y * val**2 + z * val


def cost_func(x, y, theta0, theta1, theta2, theta0_true, theta1_true, theta2_true, show='theta0_1'):
    """The cost function, J(theta0, theta1) describing the goodness of fit."""
    theta0 = np.atleast_3d(np.asarray(theta0))
    theta1 = np.atleast_3d(np.asarray(theta1))
    theta2 = np.atleast_3d(np.asarray(theta2))
    if show == 'theta0_1':
    	return np.average((y - fx(x, theta0, theta1, theta2_true))**2, axis=2)
    elif show == 'theta1_2':
    	return np.average((y - fx(x, theta0_true, theta1, theta2))**2, axis=2)
    elif show == 'theta0_2':
    	return np.average((y - fx(x, theta0, theta1_true, theta2))**2, axis=2)


def calc_gradient(lr, x_vals, theta0_val, theta1_val, theta2_val, y_vals):
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
    
    grad_theta0 = lr * gradsum_theta0 / len(x_vals)
    grad_theta1 = lr * gradsum_theta1 / len(x_vals)
    grad_theta2 = lr * gradsum_theta2 / len(x_vals)
    
    return grad_theta0, grad_theta1, grad_theta2


def gradient_descent(N, lr, x_vals, y_vals, theta0_true, theta1_true, theta2_true, show='theta0_1'):
    theta = [np.array((0, 0, 0))]
    J = [cost_func(x_vals, y_vals, *theta[0],
                    theta0_true, theta1_true, theta2_true)[0]]
    for j in range(N-1):
        last_theta = theta[-1]
        this_theta = np.empty((3,))
        grad_theta0, grad_theta1, grad_theta2 = calc_gradient(lr, x_vals, last_theta[0],
                                                              last_theta[1], last_theta[2], y_vals)
        print(theta[-1][0], theta[-1][1], theta[-1][2])
        this_theta[0] = last_theta[0] - grad_theta0
        this_theta[1] = last_theta[1] - grad_theta1
        this_theta[2] = last_theta[2] - grad_theta2
        theta.append(this_theta)
        J.append(cost_func(x_vals, y_vals, *this_theta,
                            theta0_true, theta1_true, theta2_true))
    
    return theta, J


def adam(N, lr, x_vals, y_vals, theta0_true, theta1_true, theta2_true, show='theta0_1'):
    theta = [np.array((0, 0, 0))]
    J = [cost_func(x_vals, y_vals, *theta[0],
                    theta0_true, theta1_true, theta2_true)[0]]
    for j in range(N-1):
        last_theta = theta[-1]
        this_theta = np.empty((3,))
        grad_theta0, grad_theta1, grad_theta2 = calc_gradient(lr, x_vals, last_theta[0],
                                                              last_theta[1], last_theta[2], y_vals)
        grad_arr = np.array([grad_theta0, grad_theta1, grad_theta2])

        if j == 0:
            v_t = grad_arr 
            s_t = np.power(grad_arr, 2)
        else:
            v_t = beta1 * v_t - (1- beta1) * grad_arr
            s_t = beta2 * s_t - (1- beta2) * np.power(grad_arr, 2)

        update_step = lr * v_t / np.sqrt(s_t + epsilon) * grad_arr
        this_theta[0] = last_theta[0] - update_step[0]
        this_theta[1] = last_theta[1] - update_step[1]
        this_theta[2] = last_theta[2] - update_step[2]
        theta.append(this_theta)
        J.append(cost_func(x_vals, y_vals, *this_theta,
                            theta0_true, theta1_true, theta2_true))

    return theta, J