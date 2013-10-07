#include <iostream>
#include <time>
using namespace std;

double eval(int *pj);
double hill_Climber(int *bitStream);
void menu(int *bitStream);
double GA(int *bitStream);
int *successors(int *previous);

int main()
{
	srand(time(NULL));
	int stream[150];
	for(int counter = 0; counter < 150; counter++)
	{
		stream[counter] = 1;
	}
	menu(stream);
	return 0;
}
//******************************************************************************
//Function name: menu
//Return type:      void
//Arguments:       int *
//Purpose:			  Display the menu and call the separate algorithms 
//						  depending on user selection.
//******************************************************************************
void menu(int *bitStream)
{
	int choice;
	bool exit = false;
	while(!exit)
	{
		cout << "Main menu:" << endl;
		cout << "1. Hill Climber" << endl;
		cout << "2. GA" << endl;
		cout << "3. Quit" << endl;
		cout << "Enter your choice" << endl;
		cin >> choice;
		switch(choice)
		{
			case 1:
				hillClimber(bitStream);
				break
			case 2:
				GA(bitStream);
				break;
			case 3:
				exit = true;
				break;
		}
	}
}
//******************************************************************************
//Function name: hill_Climber
//Return type:      double
//Arguments:       int *
//Purpose:			  Run eval on the bitStream and try to maximize the result
//						  through the use of a hill climber algorithm.
//******************************************************************************
double hill_Climber(int *bitStream)
{
	double result, previousResult;
	int *previous, *next;
	previous = new int[150];
	for(int hillCounter = 0; hillCounter < 150; hillCounter++)
	{
		previous[hillCounter] = bitStream[hillCounter];
	}	
	
	return result;
}
//******************************************************************************
//Function name: successors
//Return type:      int *
//Arguments:       int *
//Purpose:			  Helper function for hill_Climber. Calculates successor
//						  using a random number.
//******************************************************************************
int *successors(int *previous)
{
	int random, *next;
	next = new int[150];
	for(int sucCounter = 0; sucCounter < 150; sucCounter++)
	{
		next[sucCounter] = previous[sucCounter];
	}
	random = rand() % 150;  //generate a random number between 0 and 149
	//flip the bit. 
	if (next[random] == 0)
		next[random] = 1;
	else
		next[random] = 0;
	return next;
}