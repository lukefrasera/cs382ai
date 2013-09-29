# MainDriver.py
# import dirplay class
import sys, os
from mapGenerator import Map

sys.path.append('../')
from projectdisplayclass import plot
# IMPORT DISPLAY CLASS

def cls():
    os.system(['clear','cls'][os.name == 'nt'])

def main():
	#main program loop
	while(True):
		# Generate Terrain
		# clear console
		cls()
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
			plot.runBFS(w,h,table)
			# A*
		if*input == '2'):
			plot.runStar(w,h,table)
			# Greedy
		if(input == '3'):
			plot.runGreed(w,h,table)
			# uniform cost search
		if(input == '4'):
			plot.runUCS(w,h,table)
			# hill climbing search
		if(input == '5'):
			plot.runHill(w,h,table)

if __name__ == '__main__':
	main()