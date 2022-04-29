#ifndef CONVOLUTION
#define CONVOLUTION

#include <vector>
using namespace std;

class ConvNet {
private:
    vector<vector<float>> feedForwardHelper(vector<vector<vector<float>>> &image, int idx);
	unsigned stride;
	unsigned kern;
    unsigned p;
    unsigned inC;
    unsigned outC;
    bool eventNN;
public:
    vector<vector<vector<vector<float>>>> weights;
    vector<float> bias;
	ConvNet() {}
	ConvNet(unsigned inChannels, unsigned outChannels, unsigned k, unsigned padding, unsigned stri, bool eventNN);
    vector<vector<vector<float>>> feedForward(vector<vector<vector<float>>> &input);
    vector<vector<vector<float>>> flush(vector<vector<vector<float>>> &input);
    void pad(vector<vector<vector<float>>> &input);
    vector<vector<vector<float>>> net_out;
};
ConvNet::ConvNet(unsigned inChannels, unsigned outChannels, unsigned k, unsigned padding, unsigned stri, bool isEvent)
    : weights(outChannels, vector<vector<vector<float>>>(inChannels, vector<vector<float>>(k, vector<float>(k, 0)))),
    bias(outChannels, 0){
    p = padding;
    inC = inChannels;
    outC = outChannels;
	stride = stri;
	kern = k;
    eventNN = isEvent;
}


void padHelper(vector<vector<float>> &input) {
    int r = input.size();
    int c = input[0].size();
    for(int i = 0; i < r; i++) {
        input[i].push_back(0.0);
        input[i].insert(input[i].begin(), 0.0);
    }
    vector<float> temp(c + 2, 0);
    input.insert(input.begin(), temp);
    input.push_back(temp);
}

void ConvNet::pad(vector<vector<vector<float>>> &input) {
    for(int i = 0; i < input.size(); i++) {
        padHelper(input[i]);
    }
}


vector<vector<vector<float>>> ConvNet::flush(vector<vector<vector<float>>> &input){
     net_out = vector<vector<vector<float>>>(outC, vector<vector<float>>(input[0].size(), vector<float>(input[0][0].size(), 0))); // size of input would be same as size of output because of padding
     // net_out initialization through bias
     for(int i = 0; i < outC; i++) {
         for(int j = 0; j < input[0].size(); j++) {
             for(int k = 0; k < input[0][0].size(); k++) {
                 net_out[i][j][k] = bias[i];
             }
         }
     }
     return feedForward(input);
}

vector<vector<vector<float>>> ConvNet::feedForward(vector<vector<vector<float>>> &input) {
    if(!eventNN) {
        net_out = vector<vector<vector<float>>>(outC, vector<vector<float>>(input[0].size(), vector<float>(input[0][0].size(), 0))); // size of input would be same as size of output because of padding
        // net_out initialization through bias
        for(int i = 0; i < outC; i++) {
            for(int j = 0; j < input[0].size(); j++) {
                for(int k = 0; k < input[0][0].size(); k++) {
                    net_out[i][j][k] = bias[i];
                }
            }
        }
    }
    pad(input);
    for (int i = 0; i < input.size(); i++) {
        for (int j = 0; j + kern <= input[0].size(); j++) {
            for (int k = 0; k + kern <= input[0][0].size(); k++) {
                if (input[i][j][k] == 0 && eventNN) {

                } else {
                    for (int oc = 0; oc < outC; oc++) {
                        for (int fx = 0; fx < kern; fx++) {
                            for (int fy = 0; fy < kern; fy++) {
                                if (j - fx >= 0 && k - fy >= 0) {
                                    net_out[oc][j - fx][k - fy] += input[i][j][k] * weights[oc][i][fx][fy];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return net_out;
}
//vector<vector<float>> ConvNet::feedForwardHelper(vector<vector<vector<float>>> &input, int idx) {
//	vector<vector<float>> net_out;
//        for (int y = 0; y <= input[0].size() - kern; y += stride) {
//            net_out.push_back({});
//            for (int x = 0; x <= input[0][y].size() - kern; x += stride) {
//                float out = 0.0;
//                for (int fY = 0; fY < kern; fY++) {
//                    for (int fX = 0; fX < kern; fX++) {
//                        for(int i = 0; i < input.size(); i++) {
//                            if (y + fY < input[i].size() && x + fX < input[i][y].size()) {
//                                out += weights[idx][i][fY][fX] * input[i][y + fY][x + fX];
//                            }
//                        }
//                    }
//                }
//                net_out.back().push_back(out);
//            }
//        }
//    return net_out;
//}

//vector<vector<vector<float>>> ConvNet::feedForward(vector<vector<vector<float>>> &input) {
//    pad(input);
//	vector<vector<vector<float>>> net_out;
//    net_out.clear();
//    for(int k = 0; k < outC; k++) {
//        net_out.push_back(feedForwardHelper(input, k));
//	}
//    return net_out;
//}

#endif