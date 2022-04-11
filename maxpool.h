#ifndef MAXPOOL
#define MAXPOOL

#include <vector>
#include<float.h>
#include<algorithm>

using namespace std;

class MaxPool {
private:
	unsigned stride;
	unsigned kern;
public:
	MaxPool() {}
	MaxPool(unsigned k, unsigned stri);
	vector<vector<vector<float>>> feedForward(vector<vector<vector<float>>> &input);
};
MaxPool::MaxPool(unsigned k, unsigned stri) {
	stride = stri;
	kern = k;
}



vector<vector<vector<float>>> MaxPool::feedForward(vector<vector<vector<float>>> &input) {
	vector<vector<vector<float>>> net_out;
	for(int i = 0; i < input.size(); i++) {
		vector<vector<float>> temp_net_out;
		temp_net_out.clear();
        for (int y = 0; y <= input[i].size() - kern; y += stride) {
            temp_net_out.push_back({});
            for (int x = 0; x <= input[i][y].size() - kern; x += stride) {
                float out = input[i][y][x];
                for (int fY = 0; fY < kern; fY++) {
                    for (int fX = 0; fX < kern; fX++) {
                            if (y + fY < input[i].size() && x + fX < input[i][y].size()) {
                                out = max(out,input[i][y + fY][x + fX]);
                            }
                    }
                }
                temp_net_out.back().push_back(out);
            }
        }
		net_out.push_back(temp_net_out);
	}
    return net_out;
}

#endif