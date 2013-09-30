import sys,os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
sys.path.append(['../','..\\'][os.name == 'nt'])

from searchAlgoriths import *

class matplotSearch:
	def __init__(self, w,h,table):
		self.w = w
		self.h = h
		self.table = table
		# self.problem = Problem('''''')

	def initMatPlot(self):
		self.fig = plt.figure()
		self.ax = plt.axes(xlim=(0,self.w),  ylim=(0,self.h))
		self.path = self.ax.plot([],[], 'ro-', lw=2)
		self.subPath = self.ax.plot([],[], 'b+', lw=2)

	def BFS(self):pass

	def star(self):pass

	def greed(self):pass

	def UCS(self):pass

	def hill(self):pass

	def runBFS(self):
		self.initMatPlot()
		plt.show()
		
	def runStar():pass
	def runGreed():pass
	def runUCS():pass
	def runHill():pass