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
	vector<vector<double>> feedForward(vector<vector<double>> &input);
};
MaxPool::MaxPool(unsigned k, unsigned stri) {
	stride = stri;
	kern = k;
}


vector<vector<double>> MaxPool::feedForward(vector<vector<double>> &input) {
	vector<vector<double>> net_out;
	net_out.clear();
	for (int y = 0; y <= input.size() - kern; y += stride) {
		net_out.push_back({});
		for (int x = 0; x <= input[y].size() - kern; x += stride) {
			double out = input[y][x];
			for (int fY = 0; fY < kern; fY++) {
				for (int fX = 0; fX < kern; fX++) {
					if (y + fY < input.size() && x + fX < input[y].size()) {
						out = max(out,input[y + fY][x + fX]);
					}
				}
			}
			net_out.back().push_back(out);
		}
	}
	return net_out;
}

#endif