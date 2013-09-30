import sys
sys.path.append(['../','..\\'][os.name == 'nt'])

from mapSearchProblem import *
from searchAlgoriths import *

class TerminalSearch:
	def __init__(self, w,h,table, world):
		self.w = w
		self.h = h
		self.table = table
		initial = (0,0)
		goal = (w-1, h-1)
		self.problem = mapSearchProblem(initial, goal, w, h, world)
		self.map = self.table
		
		
	def BFS(self):pass

	def star(self):pass

	def greed(self):pass

	def UCS(self):pass

	def hill(self):pass

	def runBFS(self):
		pass
		best_first_graph_search(self.problem, node.depth)
	def runStar():pass
	def runGreed():pass
	def runUCS():pass
	def runHill():pass