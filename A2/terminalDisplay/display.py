import sys, os
sys.path.append(['../','..\\'][os.name == 'nt'])

from mapSearchProblem import *
from searchAlgorithms.searchAlgoriths import *

class TerminalSearch:
	def __init__(self, w,h,table):
		self.w = w
		self.h = h
		self.table = table
		initial = (0,0)
		goal = (w-1, h-1)
		self.problem = mapSearchProblem(initial, goal, w, h, self.table)
		self.map = self.table
	
	def nodeDepth(self, node):
		return node.depth
		
	def BFS(self):pass

	def star(self):pass

	def greed(self):pass

	def UCS(self):pass

	def hill(self):pass

	def runBFS(self):
		best_first_graph_search(self.problem, self.nodeDepth)

	def runStar():pass
	def runGreed():pass
	def runUCS():pass
	def runHill():pass