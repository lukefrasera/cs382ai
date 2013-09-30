import sys, os, math
sys.path.append(['../','..\\'][os.name == 'nt'])

from mapSearchProblem import *
from searchAlgorithms.searchAlgoriths import *


def display(map):
	string = ''
	for l in range(len(map)):
		for w in range(len(map[0])):
			if map[l][w] < 0:
				string = string +'.' # expanded
			elif map[l][w] == 9:
				string = string +'0' # path
			elif map[l][w] > 0:
				string = string +'+' # obstacle
			elif map[l][w] == 0:
				string = string +' '
		string = string + '\n'
	return string

def cls():
	os.system(['clear','cls'][os.name == 'nt'])

class TerminalSearch:
	def __init__(self,w,h,table):
		self.w = w
		self.h = h
		self.table = table
		self.initial = (h-1,0)
		self.goal = (0, w-1)
		self.map = self.table
	
	def nodeDepth(self, node):
		return node.depth
	def distance(self, node):
		x = self.goal[0] - node.state[0]
		y = self.goal[1] - node.state[1]

		return math.sqrt(x*x + y*y)
		

	def runBFS(self):
		self.problem = mapProblem(self.initial, self.goal, self.w, self.h, self.map)
		node, map =  best_first_graph_search(self.problem, self.nodeDepth)

		if node == None:
			print 'No Path Possible'
			return -1

		for i in node.path():
			map[i.state[0]][i.state[1]] = 9
		cls()
		print display(map)

	def runStar(self):
		self.problem = mapProblem(self.initial, self.goal, self.w, self.h, self.map)
		node, map =  astar_search(self.problem, self.distance)

		if node == None:
			print 'No Path Possible'
			return -1

		for i in node.path():
			map[i.state[0]][i.state[1]] = 9
		cls()
		print display(map)

	def runGreed(self):
		self.problem = mapProblem(self.initial, self.goal, self.w, self.h, self.map)
		node, map =  best_first_graph_search(self.problem, self.distance)

		if node == None:
			print 'No Path Possible'
			return -1

		for i in node.path():
			map[i.state[0]][i.state[1]] = 9
		cls()
		print display(map)
	def runUCS(self):
		self.problem = mapProblem(self.initial, self.goal, self.w, self.h, self.map)
		node, map =  uniform_cost_search(self.problem)

		if node == None:
			print 'No Path Possible'
			return -1

		for i in node.path():
			map[i.state[0]][i.state[1]] = 9
		cls()
		print display(map)

	def runHill(self):
		self.problem = mapProblem(self.initial, self.goal, self.w, self.h, self.map)
		state=  hill_climbing(self.problem)

		if state == None:
			print 'No Path Possible'
			return -1

		print "Local Max: " + str(state)