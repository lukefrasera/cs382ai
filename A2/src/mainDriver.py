# MainDriver.py
import sys, os
sys.path.append('../')

# import dirplay class
from mapGenerator import Map
from matplotdisplay.search import matplotSearch

def cls():
    os.system(['clear','cls'][os.name == 'nt'])

def main():

	plot = matplotSearch(10,10,[])
	#main program loop
	while(True):
		# clear console
		# cls()
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
			# A*
		if(input == '2'):
			plot.runStar()
			# Greedy
		if(input == '3'):
			plot.runGreed()
			# uniform cost search
		if(input == '4'):
			plot.runUCS()
			# hill climbing search
		if(input == '5'):
			plot.runHill()
		if(input == '6'):
			break
			

if __name__ == '__main__':
	main()