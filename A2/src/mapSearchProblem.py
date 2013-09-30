import sys, os
sys.path.append(['../','..\\'][os.name == 'nt'])

from searchAlgorithms.searchAlgoriths import Problem	

class mapSearchProblem(Problem):
	def __init__(self, initial, goal, rows, cols, _map):
		self.map = _map
		self.cols = cols
		self.rows = rows
		self.initial = initial
		self.goal = goal
	
	def actions(self, state):
		actionList = [(x,y) for x in range(-1,2) for y in range(-1,2) if x!=0 or y!=0]

		for action in reversed(actionList):
			x = action[0]+state[0]
			y = action[1]+state[1]

			if x>=self.cols or y>=self.rows or self.map[x][y] != 0:
				actionlist.remove(action)
		return actionList

	def result(self, state, action):
		for i in range(len(state)):
			state[i] = state[i] + action[i]
		return state