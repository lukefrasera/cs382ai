// Problem class


class Problem
{
public:
	Problem(int *);
	~Problem();

	int ** actions(int *);
	int * result(int *);
	int path_cost(int , int*, int*, int*);
	int value(int *);
	/* data */
};

class Node{
public:
	Node( int*, Node*, int*, int);
	~Node();

	Node * expand(Problem);
	Node child_node(Problem, int*);
	int ** solution();
	int ** path();
};