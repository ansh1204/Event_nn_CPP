#ifndef CONVOLUTION
#define CONVOLUTION

#include <vector>
using namespace std;

class ConvNet {
private:
    vector<vector<double>> feedForwardHelper(vector<vector<vector<double>>> &image, int idx);
	vector<vector<vector<double>>> weights;
	unsigned stride;
	unsigned kern;
    unsigned p;
    unsigned inC;
    unsigned outC;
public:
	ConvNet() {}
	ConvNet(unsigned inChannels, unsigned outChannels, unsigned k, unsigned padding, unsigned stri);
	vector<vector<vector<double>>> feedForward(vector<vector<vector<double>>> &input);
    void pad(vector<vector<vector<double>>> &input);
};
ConvNet::ConvNet(unsigned inChannels, unsigned outChannels, unsigned k, unsigned padding, unsigned stri) {
    p = padding;
    inC = inChannels;
    outC = outChannels;
	stride = stri;
	kern = k;
	// Weight initialization
    for(int k = 0; k < outChannels; k++) {
        vector<vector<double>> temp_weights;
    for (int i = 0; i < kern; i++) {
		temp_weights.push_back({});
		for (int j = 0; j < kern; j++) {
			temp_weights.back().push_back(rand() / double(RAND_MAX));
		}
	}
    weights.push_back(temp_weights);
    }
}


void padHelper(vector<vector<double>> &input) {
    int r = input.size();
    int c = input[0].size();
    for(int i = 0; i < r; i++) {
        input[i].push_back(0.0);
        input[i].insert(input[i].begin(), 0.0);
    }
    vector<double> temp(c + 2, 0);
    input.insert(input.begin(), temp);
    input.push_back(temp);
}

void ConvNet::pad(vector<vector<vector<double>>> &input) {
    for(int i = 0; i < input.size(); i++) {
        padHelper(input[i]);
    }
}
vector<vector<double>> ConvNet::feedForwardHelper(vector<vector<vector<double>>> &input, int idx) {
	vector<vector<double>> net_out;
        for (int y = 0; y <= input[0].size() - kern; y += stride) {
            net_out.push_back({});
            for (int x = 0; x <= input[0][y].size() - kern; x += stride) {
                double out = 0.0;
                for (int fY = 0; fY < kern; fY++) {
                    for (int fX = 0; fX < kern; fX++) {
                        for(int i = 0; i < input.size(); i++) {
                            if (y + fY < input[i].size() && x + fX < input[i][y].size()) {
                                out += weights[idx][fY][fX] * input[i][y + fY][x + fX];
                            }
                        }
                    }
                }   
                net_out.back().push_back(out);
            }
        }
    return net_out;
}

vector<vector<vector<double>>> ConvNet::feedForward(vector<vector<vector<double>>> &input) {
    pad(input);
	vector<vector<vector<double>>> net_out;
    net_out.clear();
    for(int k = 0; k < outC; k++) {
        net_out.push_back(feedForwardHelper(input, k));
	}
    return net_out;
}

#endif