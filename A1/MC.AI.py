def isValidCase(a,b):
	A = float(a)/3
	B = float(b)/3

	if A<0 or A>1  or B<0 or B>1:
		return False
	x = 3 - a
	y = 3 - b

	if (a<b and a!=0) or (x<y and x!=0):
		return False

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

def updateOccurence(state,N,L):
	X = []
	new = False

	for i in L:
		if state == i[0] and N<i[1]:
			X.append((state,N))
			new = True
		X.append(i)

	return (X,new)


def MCSolveRecur(state, L):

	if state==(0,0,0):
		return L.append(state)
