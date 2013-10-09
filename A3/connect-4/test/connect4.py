class ConnectFour(object):
   def __init__(self):
      self.board = []
      for i in range(6):
         self.board.append([])
         for j in range(7):
            self.board[i].append(None)
 
   def get_position(self, row, column):
      if self.board[row][column] != None:
         if self.board[row][column] == 'X':
            return 1
         else:
            return 2
      else:
         return None
 
   def find_victory(self, row, col, mRow, mCol):
      if self.get_position(row, col) != None:
         c = self.get_position(row, col)
         i = row + mRow
         j = col + mCol
         streak = 1
         for k in range(3):
            if i > 5 or j > 6 or i < 0 or j < 0:
               break
            if self.get_position(i, j) != c:
               break
            elif self.get_position(i,j) == c and k == 2:
               return c
            i += mRow
            j += mCol
      else:
         return None
 
   def is_draw(self):
      for j in range(7):
         if self.board[5][j] != None:
            continue
         else:
            return False
      return True
 
   def is_game_over(self):
      lstDirs = [(1,0),(0,1),(1,1),(1,-1)]
      for i in range(6):
         for j in range(7):
            for direction in lstDirs:
               if self.find_victory(i, j, direction[0], direction[1]) != None:
                  return self.find_victory(i, j, direction[0], direction[1])
 
   def make_move(self, player, col):
      for i in range(6):
         if self.board[i][col] != None:
            pass
         else:
            if player == 1:
               self.board[i][col] = 'X'
               break
            else:
               self.board[i][col] = 'O'
               break
 
   def print_board(self):
      print '-' * 29
      print "| 0 | 1 | 2 | 3 | 4 | 5 | 6 |"
      print '-' * 29
      for row in range(5, -1, -1):
         s = "|"
         for col in range(7):
            p = self.get_position(row, col)
            if p != None:
               if p == 1:
                  s += " X |"
               elif p == 2:
                  s += " O |"
            else:
               s += "   |"
         print s
         print '-' * 29
 
   def getFullCols(self):
      lstFull = []
      for i in range(7):
         if self.board[5][i] != None:
            lstFull.append(i)
      return lstFull
 
class Human(object):
   def __init__(self, playernum):
      self.playernum = playernum
 
   def play_turn(self, board):
      print("It is your turn player number %d!\n" %self.playernum)
      print ('*' * 9) + "Before Turn" + ('*' * 9)
      board.print_board()
      while True:
         col = raw_input("Type in the column number you would like to put your chip in: ")
         fullCols = board.getFullCols()
         if len(col) == 1 and col in '0123456' and int(col) not in fullCols:
            col = int(col)
            print ('*' * 9) + "After Turn" + ('*' * 9)
            board.make_move(self.playernum, col)
            break
         else:
            print "That input was invalid. Remember that only a value between 0 and 6 may be input"
      board.print_board()
 
def play_game(board, player1, player2):
   print "Welcome to the Connect 4 game!\n"
   turn = 1
   while True:
      print "TURN %d:\n" %turn
      if turn % 2 == 1:
         player1.play_turn(board)
      else:
         player2.play_turn(board)
      if board.is_game_over() != None:
         print "Player %d has won the game!\n" %board.is_game_over()
         break
      elif board.is_draw():
         print "It's a draw!\n"
         break
      else:
         turn +=1
 
#Program execution start begins here
p1 = Human(1)
p2 = Human(2)
game = ConnectFour()
play_game(game, p1, p2)