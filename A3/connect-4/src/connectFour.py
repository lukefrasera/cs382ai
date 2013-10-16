import sys, os, getopt
sys.path.append(['../','..\\'][os.name == 'nt'])
from include.minimaxSearch import *

def main(argv):
	# PARSE ARGUMENTS
	try:
		opts, args = getopt.getopt(argv, "d:An:c:r:", 
				["depth=", "alphabeta=", "connect=","columns=", "rows="])
	except getopt.GetoptError as e:
		print "Argument Error: %s" % e
		sys.exit(2)

	# SET DEFUALT VALUES
	dLimit = False
	depth	=9
	pruning	=False
	k		=4
	columns	=7
	rows	=6

	# HANDLE COMMANDLINE OPTIONS
	for opt, arg, in opts:
		if   opt in ("-d", "--depth"):
			depth = int(arg)
			dLimit = True

		elif opt in ("-A", "--alphabeta"):
			pruning = True

		elif opt in ("-n", "--connect"):
			k = int(arg)

		elif opt in ("-c", "--columns"):
			columns = int(arg)

		elif opt in ("-r", "--rows"):
			rows = int(arg)
	
	# START GAME AND DISPLAY
	# play_game(TicTacToe(), alphabeta_search, query_player)

	
	if pruning == True:
		if dLimit == True:
			print play_game_depth(ConnectFour(h=columns, v=rows, k=k), depth, alphabeta_player, human_player)
		else:
			print play_game(ConnectFour(h=columns, v=rows, k=k), alphabeta_full_search_player, human_player)
	elif dLimit== True:
		print play_game_depth(ConnectFour(h=columns, v=rows, k=k), depth, minimax_decision_depth_player, human_player)
	else:
		print play_game(ConnectFour(h=columns, v=rows, k=k), minimax_decision_player, human_player)

	



if __name__ == '__main__':
	main(sys.argv[1:])
