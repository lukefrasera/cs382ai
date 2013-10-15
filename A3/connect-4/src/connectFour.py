import sys, os, getopt
sys.path.append(['../','..\\'][os.name == 'nt'])
from include.abstractscene import glApplication
from include.connect4Scene import connect4Scene

def main(argv):
	# PARSE ARGUMENTS
	try:
		opts, args = getopt.getopt(argv, "d:An:c:r:", 
				["depth=", "alphabeta=", "connect=","columns=", "rows="])
	except getopt.GetoptError as e:
		print "Argument Error: %s" % e
		sys.exit(2)

	# SET DEFUALT VALUES
	depth	=9
	pruning	=True
	k		=4
	columns	=6
	rows	=7

	# HANDLE COMMANDLINE OPTIONS
	for opt, arg, in opts:
		if   opt in ("-d", "--depth"):
			depth = int(arg)

		elif opt in ("-A", "--alphabeta"):
			pruning = True

		elif opt in ("-n", "--connect"):
			k = int(arg)

		elif opt in ("-c", "--columns"):
			columns = int(arg)

		elif opt in ("-r", "--rows"):
			rows = int(arg)
	
	# START GAME AND DISPLAY
	connectGame = connect4Scene(depth=depth, ab=pruning, k=k, cols=columns, rows=rows)
	app = glApplication(connectGame)



if __name__ == '__main__':
	main(sys.argv[1:])
