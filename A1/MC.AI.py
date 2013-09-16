def isValidCase(a,b):
	# Check that a and be are with in range
	A = float(a)/3
	B = float(b)/3
	if A<0 or A>1  or B<0 or B>1:
		return False
	# interpret other side of river
	x = 3 - a
	y = 3 - b

	# Check for invalid state
	if (a<b and a!=0) or (x<y and x!=0):
		return False
	return True

def isValidState(state):
	return isValidCase(state[0],state[1])

# This functionw will take any tuple and return
# the negated version of that same tuple
def negateTuple(L):
	x = ()
	for i in L:
		x += (i * -1,)
	return x

# This function will generate all the possible branhes
# a node can have in the recursive tree
def genPossibilityList(a,b,c):
	# i have commented out the list generation of all
	# possibilties and hard coded the result for speed

	#L = [(x,y,1) for x in range(a+1) for y in range(b+1) if x+y<=2 and (x is not 0 or y is not 0)]
	L = [(0,1,1), (0,2,1), (1,0,1), (1,1,1), (2,0,1)]

	# Here i filter the poissiblility list further by 
	# checking against the side on which the boat is 
	# to attempt and further limit branching
	if c is 0:
		a = 3-a
		b = 3-b
		X = [i for i in L if i[0]<=a and i[1] <=b]
		return list(map(negateTuple, X))		
	return [i for i in L if i[0]<=a and i[1]<=b]

#Keep a list of result
#[(state, N),...]

# updateOccurrence: This function maintains a list of the states
# that have been visisted already and the depth at which that state
# was reached. This list is compared against inorder to prevent redundant
# branches of already reached states that are longer that previously
# stored occurences.
def updateOccurence(state,N):
	global OcurencesList
	X = []
	new = True
	append = False

	for i in OcurencesList:
		if state == i[0]:
			new = False
			if N<i[1]:
				X.append((state,N))
				append = True
			else:
				X.append(i)
		else:
			X.append(i)
	if new:
		X.append((state,N))
		append = True
	OcurencesList = X
	return append


# This is the main recursive function that will return one of the optimal
# solutions to the missionaries and cannibals problem.
def MCSolveRecur(state, N):
	newState = updateOccurence(state,N)

	if not newState:
		return None

	if state==(0,0,0):
		return [state]

	if not isValidState(state):
		return None

	L = [MCSolveRecur(minus(state,i), N+1)for i in genPossibilityList(state[0],state[1],state[2])]


	for i in sorted(filter(None,L), key=lambda list: len(list)):
		if i[len(i)-1] != None:
			return [state] + i
	return None

# This function subtracts to tuples: size 3
def minus(xT,yT):
	a = xT[0] - yT[0]
	b = xT[1] - yT[1]
	c = xT[2] - yT[2]
	return (a,b,c)


def main():
	initialState = (3,3,1)
	global OcurencesList

	L = MCSolveRecur(initialState,0)

	print "\nOccurence List:"
	print OcurencesList
	print "\nAnswer:"
	print L

OcurencesList = []

if __name__ == '__main__':
	main()


# I would rework this inorder to make my code cleaner and more readabl and avoid some indicission of choices between
# recurrsive and iterative solutions. I would prefer to use Breadth first search with queues. python feels inefficient
# for something like this due to all the lists used and copying of lists needed. c++ would be more concrete.
# I used BFS because it provides a fast algorithm that will find the optimal solutions first given that 
# the optimal solutions will be the shortest branch resulting in a solution. I like using parts from dynamic
# programming and storing a list of previously disovered states seems like a good idea for speeding
# up the search because unless a discovered state is found in a shorter time there is no sense continueing
# along the branch or if it dicovered in the same amount of time this only has the potential to
# produce another optimal solution, but not one that is better.