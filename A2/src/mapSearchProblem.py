import sys
sys.path.append('../')

from searchAlgoriths import Problem	

class mapSearchProblem(Problem)
	def __init__(self, initial, goal, rows, cols, map)
		self.map = map
		self.cols = cols
		self.rows = rows
		self.initial = initial
		self.goal = goal
	
	def actions(self, row, col)
		actionList = [(row + cols) * row, 
		return actionList