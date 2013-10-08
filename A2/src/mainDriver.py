# MainDriver.py
import sys, os, random
sys.path.append(['../','..\\'][os.name == 'nt'])

# import dirplay class
from mapGenerator import Map
from terminalDisplay.display import TerminalSearch

def cls():
    os.system(['clear','cls'][os.name == 'nt'])

def main():
	random.seed()
	rows = random.randint(20, 40)
	cols = random.randint(50, 90)

	ourMap = Map(cols, rows)
	for i in range(5):
		ourMap.makeRectObstacle()
	ourMap = ourMap.map

	plot = TerminalSearch(cols, rows, ourMap)
	#main program loop
	# while(True):
	# clear console
	cls()
	# Generate Terrain
	# Generate menu cases
	input = raw_input("Choose Algorithm:\n\
1) BFS   :\n\
2) A*    :\n\
3) Greedy:\n\
4) UCS   :\n\
5) Hill  :\n\
6) Quit  :\n")
	
		# BFS
	if(input == '1'):
		plot.runBFS()
		raw_input("PRESS ENTER TO CONTINUE")
		# A*
	if(input == '2'):
		plot.runStar()
		raw_input("PRESS ENTER TO CONTINUE")
		# Greedy
	if(input == '3'):
		plot.runGreed()
		raw_input("PRESS ENTER TO CONTINUE")
		# uniform cost search
	if(input == '4'):
		plot.runUCS()
		raw_input("PRESS ENTER TO CONTINUE")
		# hill climbing search
	if(input == '5'):
		plot.runHill()
		raw_input("PRESS ENTER TO CONTINUE")
			

if __name__ == '__main__':
	main()