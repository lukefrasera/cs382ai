import random

class Map:
    def __init__(self, l, w):
        '''Creates a blank map of a 2D world without obstactles. l columns and w rows'''
        self.maxRows = w
        self.maxCols = l
        self.blank()

    def __getitem__(self, state):
        print "STOP GETTING MY ITEM!"

    def blank(self):
        self.map = [[0 for x in range (self.maxCols)] for x in range(self.maxRows)]

    def makeRectObstacle(self):
        '''Make a random rectangular obstacle'''
        row = random.randint(0, self.maxRows - 2)
        col = random.randint(0, self.maxCols - 2)
        height = random.randint(1, self.maxRows - 1 - row)
        width  = random.randint(1, self.maxCols - 1 - col)

        self.changeMap(row, col, height, width)
        return (row, col, height, width)
        
        
    def changeMap(self, row, col, height, width):
        '''Marks up a rectangle in our 2D map'''
        for x in range (row, row+height):
            for y in range(col, col+width):
                self.map[x][y] += 1

    def path(self, path):
        pass

    def copyWorldMap(self, anotherWorld):
        '''Copies world, obstacles and all'''
        for r in range(self.maxRows):
            for c in range(self.maxCols):
                self.map[r][c] = anotherWorld.map[r][c]

    def display(self):
        '''Display for curses'''
        string = ''
        for l in range(self.maxRows):
            for w in range(self.maxCols):
                if self.map[l][w] < 0:
                    string = string +'.' # expanded
                elif self.map[l][w] == 9:
                    string = string +'0' # path
                elif self.map[l][w] > 0:
                    string = string +'+' # obstacle
        return string

    def __repr__(self):
        '''For printing'''
        out = ''
        for l in range(self.maxRows):
            for w in range(self.maxCols):
                if self.map[l][w] == 0:
                    out = out + ' '
                elif self.map[l][w] < 0:
                    out = out + str(-self.map[l][w])
                else:
                    out = out + str(self.map[l][w])
            out = out + " |\n"
        return out
                
# For testing this code
if __name__ == "__main__" :
    random.seed()
    maxCols = 80
    maxRows = 30
    world = Map(maxCols, maxRows)
    for x in range(5):
        tmp = world.makeRectObstacle()
    world.display()
