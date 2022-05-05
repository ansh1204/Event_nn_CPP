#ifndef RELU
#define RELU

#include <vector>

using namespace std;

class ReLU {
private:

public:
	ReLU() {}
	vector<vector<vector<float>>> feedForward(vector<vector<vector<float>>> &input);
};

vector<vector<vector<float>>> ReLU::feedForward(vector<vector<vector<float>>> &input) {
	vector<vector<vector<float>>> net_out;
	for(int i = 0; i < input.size(); i++) {
		vector<vector<float>> temp_net_out;
		temp_net_out.clear();
		for (int y = 0; y < input[i].size(); y++) {
			temp_net_out.push_back({});
			for (int x = 0; x < input[i][y].size(); x+=1) {
				float val = (0 < input[i][y][x]) ? input[i][y][x] : 0;
				temp_net_out.back().push_back(val);
			}
		}
		net_out.push_back(temp_net_out);
	}
	return net_out;
}

#endif