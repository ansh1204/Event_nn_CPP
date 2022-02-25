#ifndef CONVOLUTION
#define CONVOLUTION

#include <vector>
using namespace std;

class ConvNet {
private:
	vector<vector<double>> weights;
	unsigned stride;
	unsigned kern;
    unsigned pad;
    unsigned inC;
    unsigned outC;
public:
	ConvNet() {}
	ConvNet(unsigned inChannels, unsigned outChannels, unsigned k, unsigned padding, unsigned stri);
	vector<vector<double>> feedForward(vector<vector<double>> &input);
};
ConvNet::ConvNet(unsigned inChannels, unsigned outChannels, unsigned k, unsigned padding, unsigned stri) {
    pad = padding;
    inC = inChannels;
    outC = outChannels;
	stride = stri;
	kern = k;
	// Weight initialization
	for (int i = 0; i < kern; i++) {
		weights.push_back({});
		for (int j = 0; j < kern; j++) {
			weights.back().push_back(rand() / double(RAND_MAX));
		}
	}
}


vector<vector<double>> ConvNet::feedForward(vector<vector<double>> &input) {
	vector<vector<double>> net_out;
    net_out.clear();
	for (int y = 0; y <= input.size() - kern; y += stride) {
		net_out.push_back({});
		for (int x = 0; x <= input[y].size() - kern; x += stride) {
			double out = 0.0;
			for (int fY = 0; fY < kern; fY++) {
				for (int fX = 0; fX < kern; fX++) {
					if (y + fY < input.size() && x + fX < input[y].size()) {
						out += weights[fY][fX] * input[y + fY][x + fX];
					}
				}
			}
			net_out.back().push_back(out);
		}
	}
    return net_out;
}

#endif