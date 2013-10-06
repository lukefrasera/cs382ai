#include <iostream>
using namespace std;

double eval(int *pj);
double hill_Climber(int *bitStream);
void menu(int *bitStream);

int main()
{
	int stream[150];
	for(int counter = 0; counter < 150; counter++)
	{
		stream[counter] = 1;
	}
	menu(stream);
	return 0;
}

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

double hill_Climber(int *bitStream)
{
	double result;
	return result;
}