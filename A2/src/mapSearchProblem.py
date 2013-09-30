import sys, os, math
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

class mapProblem(Problem):
	def __init__(self, initial, goal=None, mRow=0, mCol=0, table=None):
		self.initial = initial; self.goal = goal
		self.x = mCol; self.y = mRow
		self.map = table

	def actions(self, state):
		actionList = [(x,y) for x in range(-1,2) for y in range(-1,2) if x!=0 or y!=0]
		for action in reversed(actionList):
			x = action[0]+state[0]
			y = action[1]+state[1]

			if x<0 or y<0 or x>=self.x or y>=self.y or self.map[x][y] !=0:
				actionList.remove(action)
		return actionList

	def result(self, state, action):
		x = state[0] + action[0]
		y = state[1] + action[1]
		return x,y

	def value(self, state):
		x = self.goal[0] - state[0]
		y = self.goal[1] - state[1]

		return 1/(math.sqrt(x*x + y*y)+1)