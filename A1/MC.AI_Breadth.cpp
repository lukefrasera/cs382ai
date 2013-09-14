#include <iostream>
#include <queue>
#include <stack>
#include <vector>

int BFS(){
	std::queue< std::vector<short> > BFSQue;

	// PRIME QUEUE
	std::vector<short>vec3(3);
	vec3[0] = 3;vec3[1] = 3;vec3[2] = 1;

	BFSQue.push(vec3);


	// ENTER BFS LOOP
	while(!BFSQue.empty()){
		vec3 = BFSQue.front();
		BFSQue.pop();

		// CHECK IF RESULT IS COMPLETE
		if(isFinished(vec3)){
			return 1;
		}
		if(isValidState(vec3)){
			// ADD ALL BRANCHES TO QUEUE
		}
	}

	std::cout << BFSQue.size() << std::endl;
}

int main(){

	BFS();
	return 0;
}