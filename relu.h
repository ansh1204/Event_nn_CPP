#ifndef RELU
#define RELU

#include <vector>

using namespace std;

class ReLU {
private:

public:
	ReLU() {}
	vector<vector<vector<double>>> feedForward(vector<vector<vector<double>>> &input);
};

vector<vector<vector<double>>> ReLU::feedForward(vector<vector<vector<double>>> &input) {
	vector<vector<vector<double>>> net_out;
	for(int i = 0; i < input.size(); i++) {
		vector<vector<double>> temp_net_out;
		temp_net_out.clear();
		for (int y = 0; y < input[i].size(); y++) {
			temp_net_out.push_back({});
			for (int x = 0; x < input[i][y].size(); x+=1) {
				int val = (0 < input[i][y][x]) ? input[i][y][x] : 0;
				temp_net_out.back().push_back(val);
			}
		}
		net_out.push_back(temp_net_out);
	}
	return net_out;
}

#endif