import numpy as np

def c(p):
	'''
	inputs: p is array-like (x,y)
	returns: control action u [rad]
	'''
	L = .324 #length of wheelbase [m]
	l = np.sqrt(np.sum(p[0]**2+p[1]**2)) #lookahead distance [m]
	a = np.arctan(p[0]/p[1]) #angle of curvature [rad]
	u = np.arctan(2*L*np.sin(a)/l) #control action [rad]
	return u