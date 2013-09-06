def isValidCase(a,b):
	A = float(a)/3
	B = float(b)/3
	if A<0 or A>1  or B<0 or B>1:
		return False
	x = 3 - a
	y = 3 - b
	if (a<b and a!=0) or (x<y and x!=0):
		return False
	return True
	
def isValidState(state):
	return isValidCase(state[0],state[1])

def negateTuple(L):
	x = ()
	for i in L:
		x += (i * -1,)
	return x

def genPossibilityList(a,b,c):
	#L = [(x,y,1) for x in range(a+1) for y in range(b+1) if x+y<=2 and (x is not 0 or y is not 0)]
	L = [(0,1,1), (0,2,1), (1,0,1), (1,1,1), (2,0,1)]
	if c is 0:
		a = 3-a
		b = 3-b
		X = [i for i in L if i[0]<=a and i[1] <=b]
		return list(map(negateTuple, X))		
	return [i for i in L if i[0]<=a and i[1]<=b]

#Keep a list of result
#[(state, N),...]

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


def minus(xT,yT):
	a = xT[0] - yT[0]
	b = xT[1] - yT[1]
	c = xT[2] - yT[2]
	return (a,b,c)


def main():
	initialState = (3,3,1)
	global OcurencesList

	L = MCSolveRecur(initialState,0)

	print OcurencesList
	print "\nAnswer:"
	print L

OcurencesList = []

if __name__ == '__main__':
	main()


# you never subtracted to get next value 	