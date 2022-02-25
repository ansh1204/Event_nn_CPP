#include <iostream>
#include <vector>
using namespace std;

class ConvNet {
private:
	vector<vector<double>> weights;
	vector<vector<double>> net_out;
	vector<vector<double>> net_input;
	unsigned stride;
	double eta;
	unsigned kern;
public:
	ConvNet() {}
	ConvNet(unsigned k, unsigned stri, double e);
	void feedForward(vector<vector<double>> &input);
};
ConvNet::ConvNet(unsigned k, unsigned stri, double e) {
	// Weight initialization
	for (int i = 0; i < kern; i++) {
		weights.push_back({});
		for (int j = 0; j < kern; j++) {
			weights.back().push_back(rand() / double(RAND_MAX));
		}
	}
	stride = stri;
	eta = e;
	kern = k;
}


void ConvNet::feedForward(vector<vector<double>> &input) {
	net_out.clear();
	net_input = input;
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
}