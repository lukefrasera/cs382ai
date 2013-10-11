

class connect4:
	def __init__(self, cols):
		self.cols = cols
	def genBoard(self):
		self.board=[[0 for x in range(6)] for y in range(self.cols)]
	def placePiece(self, col, mark):
		for i, j  in enumerate(self.board[col]):
			if j == 0:
				self.board[col][i]=mark
				return col, row
	def is_win(self, col, row):
		if col <= 3: xlow=0 else: xlow=col-3
		if row <= 3: ylow=0 else: ylow=row-3
		if col >= self.cols-3: xhigh=self.cols else: xhigh=col+3
		if row >= 6-3: yhigh=6 else: yhigh=row+3

		dirlist = [(1,0), (0,1), (1,1), (1,-1)]
		poslist = [(xlow, row), (col, ylow), (xlow, ylow), (xlow, yhigh)]
		piece = self.board[col][row]
		count = 0

		for i in range(len(dirlist)):
			x, y=poslist[i]
			for j in range(7):
				if self.baord[x,y]==piece:
					count += 1
				else:
					count = 0
				if count == 4:
					return True
		return False

	def is_draw(self):
		for col in range(self.cols):
			if self.board[col][6] != 0:
				return False
		return True

	def game(self):
		self.genBoard()
		player = 0

		while True:
			turn %= 2
			column = raw_input("selec Column")
			