import sys
sys.path.append('../')

from searchAlgoriths import *


class matplotSearch:
	def __init__(self, w,h,table):
		self.w = w
		self.h = h
		self.table = table

	def BFS(self, problem, frontier)
		frontier.append(Node(problem.initial))
		explored = set()
		while frontier:
        node = frontier.pop()
        if problem.goal_test(node.state):
            return node
        explored.add(node.state)
        frontier.extend(child for child in node.expand(problem)
                        if child.state not in explored
                        and child not in frontier)

	def star(self):pass

	def greed(self):pass

	def UCS(self):pass

	def hill(self):pass

	def runBFS(self):
		print 'BFS'

		
	def runStar():pass
	def runGreed():pass
	def runUCS():pass
	def runHill():pass