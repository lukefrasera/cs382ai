#include <iostream>
#include <stdlib.h>
#include <time.h>

using namespace std;

double eval(int *pj);
double hill_Climber(int *bitStream, int numIterations);
void menu(int *bitStream, int numIterations);
//double GA(int *bitStream);
int *successors(int *previous);


/*
typedef enum seeds_{
	SEED_0 = 9134028123
	SEED_1
	SEED_2
	SEED_3
	SEED_4
	SEED_5
	SEED_6
	SEED_7
	SEED_8
	SEED_9
}seeds;

unsigned int SEED[] = {
	0234928198,
	0340287349,
	9023028094,
	0202709292,
	0234234830,
	0948303342,
	9287699872,
	9472958309,
	2229874468,
	0948459893
}
*/
int main(int argc, char* argv[])
{
	int iterations;
	if(argc != 0)
	{
		iterations = (atoi(argv[1]));
	}
	else
	{
		//if no argument is passed, set iterations to default value
		iterations = 1000;
	}

	srand(time(NULL));
	int stream[150];
	for(int counter = 0; counter < 150; counter++)
	{
		stream[counter] = 1;
	}
	menu(stream, iterations);
	return 0;
}
//******************************************************************************
//Function name: menu
//Return type:      void
//Arguments:       int *
//Purpose:			  Display the menu and call the separate algorithms 
//						  depending on user selection.
//******************************************************************************
void menu(int *bitStream, int numIterations)
{
	int choice;
	bool exit = false;
	int run = 10;
	int runningTotal = 0;
	int average = 0;
	int result;
	while(!exit)
	{
		runningTotal = 0;
		cout << "Main menu:" << endl;
		cout << "1. Hill Climber" << endl;
		cout << "2. GA" << endl;
		cout << "3. Quit" << endl;
		cout << "Enter your choice" << endl;
		cin >> choice;
		switch(choice)
		{
			case 1:
				for(int counter = 0; counter < run; counter++)
				{
					result = hill_Climber(bitStream, numIterations);
					cout << "Attempt # " << counter + 1 << " result: " << result << endl;
					runningTotal += result;
				}
				average = runningTotal / 10;
				cout << "Average: " << average << endl;
				break;
			case 2:
//				GA(bitStream);
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
double hill_Climber(int *bitStream, int numIterations)
{
	double result, previousResult;
	int *previous, *next;
	previous = new int[150];
	for(int hillCounter = 0; hillCounter < 150; hillCounter++)
	{
		previous[hillCounter] = bitStream[hillCounter];
	}
	int counter = 0;
	previousResult = eval(previous);
	while(counter < numIterations - 1)
	{
		next = successors(previous);
		result = eval(next);
		if(result >= previousResult)
		{
			for(int copyCounter = 0; copyCounter < 150; copyCounter++)
			{
				previous[copyCounter] = next[copyCounter];
				previousResult = result;
			}
			delete [] next;
		}
		counter++;
	}
	return previousResult;
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
