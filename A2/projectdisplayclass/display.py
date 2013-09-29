import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

fig = plt.figure()
ax = plt.axes(xlim=(0,2)), ylim=(-2,2))
points = ax.plot([], [], lw=2)

def init():
	points.setdata([], [])
	return points

def animate(i):
	