import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import derivative as dx
from gd_methods import gradient_descent
from gd_methods import fx as hypothesis

parser = argparse.ArgumentParser('My program')
parser.add_argument('-x', '--one')
parser.add_argument('-y', '--two')
parser.add_argument('-z', '--three')
parser.add_argument('-type', '--show')
args = parser.parse_args()
theta0_true = int(args.one)
theta1_true = int(args.two)
theta2_true = int(args.three)
show = str(args.show)
print(theta0_true, type(theta0_true))

# N is the number of iterations
N = 8	

# The data to fit
m = 20
x = np.linspace(-1,1,m)
y = theta0_true * x**3 + theta1_true * x**2 + theta2_true * x
y = np.random.randn(m) + y

# The plot: LHS is the data, RHS will be the cost function.
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,6.15))
ax[0].scatter(x, y, marker='x', s=40, color='k')

def cost_func(theta0, theta1, theta2):
    """The cost function, J(theta0, theta1) describing the goodness of fit."""
    theta0 = np.atleast_3d(np.asarray(theta0))
    theta1 = np.atleast_3d(np.asarray(theta1))
    theta2 = np.atleast_3d(np.asarray(theta2))
    if show == 'theta0_1':
    	return np.average((y - hypothesis(x, theta0, theta1, theta2_true))**2, axis=2)
    elif show == 'theta1_2':
    	return np.average((y - hypothesis(x, theta0_true, theta1, theta2))**2, axis=2)
    elif show == 'theta0_2':
    	return np.average((y - hypothesis(x, theta0, theta1_true, theta2))**2, axis=2)

# First construct a grid of (theta0, theta1) parameter pairs and their
# corresponding cost function values.
theta0_grid = np.linspace(theta0_true-5, theta0_true+5,101)
theta1_grid = np.linspace(theta1_true-5, theta1_true+5,101)
theta2_grid = np.linspace(theta2_true-5, theta2_true+5,101)

if show == 'theta0_1':
	J_grid = cost_func(theta0_grid[np.newaxis,:,np.newaxis],
		               theta1_grid[:,np.newaxis,np.newaxis],
		               theta2_grid[np.newaxis,:,np.newaxis])
	# A labeled contour plot for the RHS cost function
	X, Y = np.meshgrid(theta0_grid, theta1_grid)
	contours = ax[1].contour(X, Y, J_grid, 30)
	ax[1].clabel(contours)
	# The target parameter values indicated on the cost function contour plot
	ax[1].scatter([theta0_true]*2,[theta1_true]*2,s=[50,10], color=['k','w'])
	
elif show == 'theta1_2':
	J_grid = cost_func(theta0_grid[np.newaxis,:,np.newaxis],
		               theta1_grid[np.newaxis,:,np.newaxis],
		               theta2_grid[:,np.newaxis,np.newaxis])
	# A labeled contour plot for the RHS cost function
	X, Y = np.meshgrid(theta1_grid, theta2_grid)
	contours = ax[1].contour(X, Y, J_grid, 30)
	ax[1].clabel(contours)
	# The target parameter values indicated on the cost function contour plot
	ax[1].scatter([theta1_true]*2,[theta2_true]*2,s=[50,10], color=['k','w'])	
	              
elif show == 'theta0_2':
	J_grid = cost_func(theta0_grid[np.newaxis,:,np.newaxis],
		               theta1_grid[np.newaxis,:,np.newaxis],
		               theta2_grid[:,np.newaxis,np.newaxis])
	# A labeled contour plot for the RHS cost function
	X, Y = np.meshgrid(theta0_grid, theta2_grid)
	contours = ax[1].contour(X, Y, J_grid, 30)
	ax[1].clabel(contours)
	# The target parameter values indicated on the cost function contour plot
	ax[1].scatter([theta0_true]*2,[theta2_true]*2,s=[50,10], color=['k','w'])
	
print(J_grid.shape)


# Take N steps with learning rate alpha down the steepest gradient,
# starting at (theta0, theta1) = (0, 0).

alpha = 0.7
theta = [np.array((0, 0, 0))]
J = [cost_func(*theta[0])[0]]
for j in range(N-1):
    last_theta = theta[-1]
    this_theta = np.empty((3,))
    grad_theta0, grad_theta1, grad_theta2 = gradient_descent(alpha, x,
                                                last_theta[0], last_theta[1], last_theta[2],
                                                y)
    print(theta[-1][0], theta[-1][1], theta[-1][2])
    this_theta[0] = last_theta[0] - grad_theta0
    this_theta[1] = last_theta[1] - grad_theta1
    this_theta[2] = last_theta[2] - grad_theta2
    theta.append(this_theta)
    J.append(cost_func(*this_theta))


# Annotate the cost function plot with coloured points indicating the
# parameters chosen and red arrows indicating the steps down the gradient.
# Also plot the fit function on the LHS data plot in a matching colour.
colors = ['#001ce5', '#0055e1', '#0d8ddd', '#00c2d9', '#00d5b4', '#00d27b',
          '#00ce44', '#d0ca0e', '#24c600', '#56c200', '#85bf00',
          '#001ce5', '#0055e1', '#0d8ddd', '#00c2d9', '#00d5b4', '#00d27b',
          '#00ce44', '#d0ca0e', '#24c600', '#56c200', '#85bf00']
ax[0].plot(x, hypothesis(x, *theta[0]), color=colors[0], lw=1,
           label=r'$\theta_0 = {:.3f}, \theta_1 = {:.3f}, \theta_2 = {:.3f}$'.format(*theta[0]))

for j in range(1,N):
	if show == 'theta0_1':
		ax[1].annotate('', xy=theta[j][:2], xytext=theta[j-1][:2],
		               arrowprops={'arrowstyle': '->', 'color': 'r', 'lw': 1},
		               va='center', ha='center')
	elif show == 'theta1_2':
		ax[1].annotate('', xy=theta[j][1:3], xytext=theta[j-1][1:3],
		               arrowprops={'arrowstyle': '->', 'color': 'r', 'lw': 1},
		               va='center', ha='center')	     
	elif show == 'theta0_2':
		ax[1].annotate('', xy=theta[j][::2], xytext=theta[j-1][::2],
		               arrowprops={'arrowstyle': '->', 'color': 'r', 'lw': 1},
		               va='center', ha='center')	    
		                                    
	ax[0].plot(x, hypothesis(x, *theta[j]), color=colors[j], lw=0.5, label=r'$\theta_0 = {:.3f}, \theta_1 = {:.3f}, \theta_2 = {:.3f}$'.format(*theta[j]))

if show == 'theta0_1':
	ax[1].scatter(*zip(*np.ndarray.tolist(np.array(theta)[:, :2])), c=colors[:N], s=20, lw=0)
elif show == 'theta1_2':
	ax[1].scatter(*zip(*np.ndarray.tolist(np.array(theta)[:, 1:3])), c=colors[:N], s=20, lw=0)
elif show == 'theta0_2':
	ax[1].scatter(*zip(*np.ndarray.tolist(np.array(theta)[:, ::2])), c=colors[:N], s=20, lw=0)

# Labels, titles and a legend.
if show =='theta0_1':
	ax[1].set_xlabel(r'$\theta_0$')
	ax[1].set_ylabel(r'$\theta_1$')
elif show =='theta1_2':
	ax[1].set_xlabel(r'$\theta_1$')
	ax[1].set_ylabel(r'$\theta_2$')
elif show =='theta0_2':
	ax[1].set_xlabel(r'$\theta_1$')
	ax[1].set_ylabel(r'$\theta_2$')
ax[1].set_title('Cost function')
ax[0].set_xlabel(r'$x$')
ax[0].set_ylabel(r'$y$')
ax[0].set_title('Data and fit')
axbox = ax[0].get_position()
# Position the legend by hand so that it doesn't cover up any of the lines.
ax[0].legend(loc=(axbox.x0+0.5*axbox.width, axbox.y0+0.1*axbox.height),
             fontsize='small')

plt.show()
#plt.savefig('2_fitting_cost_gd.jpg')
