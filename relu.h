#ifndef RELU
#define RELU

#include <vector>

using namespace std;

class ReLU {
private:

public:
	ReLU() {}
	void feedForward(vector<vector<double>> &input);
};

void ReLU::feedForward(vector<vector<double>> &input) {
	for (int y = 0; y < input.size(); y++) {
		for (int x = 0; x < input[y].size(); x+=1) {
			input[y][x] = (0 < input[y][x]) ? input[y][x] : 0;
		}
	}
}

#endif