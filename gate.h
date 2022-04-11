#ifndef EVENT_NN_C_GATE_H

#include <vector>
using namespace std;
class Gate {
private:

public:
    Gate() {}
    Gate(unsigned thr);
    vector<vector<vector<float>>> feedForward(vector<vector<vector<float>>> &input);
    vector<vector<vector<float>>> flush(vector<vector<vector<float>>> &input);
    vector<vector<vector<float>>> d;
    vector<vector<vector<float>>> b;
    unsigned threshold;
};

Gate::Gate(unsigned thr) {
    threshold = thr;
}
vector<vector<vector<float>>> Gate::flush(vector<vector<vector<float>>> &input){
    d = vector<vector<vector<float>>>(input.size(), vector<vector<float>>(input[0].size(), vector<float>(input[0][0].size(), 0)));
    b = input;
    return b;
}
vector<vector<vector<float>>> Gate::feedForward(vector<vector<vector<float>>> &input) {
    vector<vector<vector<float>>> net_out(input.size(), vector<vector<float>>(input[0].size(), vector<float>(input[0][0].size(), 0)));;
    for(int i = 0; i < input.size(); i++) {
        for(int j = 0; j < input[0].size(); j++) {
            for(int k = 0; k < input[0][0].size(); k++) {
                d[i][j][k] = d[i][j][k] + input[i][j][k] - b[i][j][k];
                if(d[i][j][k] >= threshold) {
                    net_out[i][j][k] = d[i][j][k];
                    d[i][j][k] = 0;
                }
            }
        }
    }
    b = input;
    return net_out;
}
#define EVENT_NN_C_GATE_H

#endif //EVENT_NN_C_GATE_H
