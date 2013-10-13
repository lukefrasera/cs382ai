

class connect4:
	def __init__(self, cols):
		self.cols = cols
	def genBoard(self):
		self.board=[[0 for x in range(7)] for y in range(self.cols)]
	def placePiece(self, col, mark):
		for i, j  in enumerate(self.board[col]):
			if j == 0:
				self.board[col][i]=mark
				return i

	def find_victory(self, col, row, mRow, mCol):
		if self.board[row][col] != None:
			c = self.board[row][col]
			i = row + mRow
			j = col + mCol
			streak = 1
			for k in range(3):
				if i > 5 or j > 6 or i < 0 or j < 0:
					break
				if self.board[j][i] != c:
					break
				elif self.board[j][i] == c and k == 2:
					return c
				i += mRow
				j += mCol
			else:
				return None

	def is_game_over(self):
		lstDirs = [(1,0),(0,1),(1,1),(1,-1)]
		for i in range(self.cols):
			for j in range(7):
				for direction in lstDirs:
					if self.find_victory(i, j, direction[0], direction[1]) != None:
						return self.find_victory(i, j, direction[0], direction[1])


	def is_draw(self):
		for col in range(self.cols):
			if self.board[col][6] == 0:
				return False
		return True
	def isFullColumn(self, col):
		if self.board[col][6] != 0:
			return True
		return False

	def game(self):
		self.genBoard()
		turn = 0

		while True:
			turn %= 2
			colString = raw_input("select Column")
			try:
				col = int(colString)
				if col < 0 or col > self.cols:
					raise ValueError('Not within Range')
				if self.isFullColumn(col):
					raise ValueError('Column is Full')
				row = self.placePiece(col, turn+1)
				if self.is_game_over():
					print "winner"
					return turn
				elif self.is_draw():
					return -1
				turn += 1

			except ValueError as e:
				print "Value not a Positive Integer Number"


def main():
	connectFour = connect4(7)

	winner = connectFour.game()
	print "winner %s" %winner

if __name__ == '__main__':
	main()