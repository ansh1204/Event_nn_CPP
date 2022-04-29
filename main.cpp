//
#include <iostream>
#include <vector>
#include <stdint.h>
#include "vgg16.h"
#include "cnpy.h"
#include<complex>
#include <fstream>
#include <iterator>
#include <vector>
#include <bits/stdc++.h>
//
using namespace std;


void load_weights_helper(string a, vector<vector<vector<vector<float>>>> wt) {
    cnpy::NpyArray weights = cnpy::npz_load("/home/ansh/Desktop/IS/openpose_mpii.npz",a);
    float* temp = weights.data<float>();
    int n = weights.shape[0]*weights.shape[1]*weights.shape[2]*weights.shape[3];
    int outC = wt.size();
    int inC = wt[0].size();
    int h = wt[0][0].size();
    int w = wt[0][0][0].size();
    for(int i = 0; i < n; i++) {
        wt[i%outC][i%inC][i%h][i%w] = temp[i];
    }
}

void load_bias_helper(string a, vector<float> bias) {
    cnpy::NpyArray bs = cnpy::npz_load("/home/ansh/Desktop/IS/openpose_mpii.npz",a);
    float* temp = bs.data<float>();
    int n = bs.shape[0];
    for(int i = 0; i < n; i++) {
        bias[i] = temp[i];
    }
}

void load_weights(vgg network){
    load_weights_helper("conv1_1-0", network.conv1_1.weights);
    load_bias_helper("conv1_1-1", network.conv1_1.bias);
    
    load_weights_helper("conv1_2-0", network.conv1_2.weights);
    load_bias_helper("conv1_2-1", network.conv1_2.bias);
    
    load_weights_helper("conv2_1-0", network.conv2_1.weights);
    load_bias_helper("conv2_2-1", network.conv2_1.bias);
    
    load_weights_helper("conv2_2-0", network.conv2_2.weights);
    load_bias_helper("conv2_2-1", network.conv2_2.bias);
    
    load_weights_helper("conv3_1-0", network.conv3_1.weights);
    load_bias_helper("conv3_1-1", network.conv3_1.bias);
    
    load_weights_helper("conv3_2-0", network.conv3_2.weights);
    load_bias_helper("conv3_2-1", network.conv3_2.bias);
    
    load_weights_helper("conv3_3-0", network.conv3_3.weights);
    load_bias_helper("conv3_3-1", network.conv3_3.bias);
    
    load_weights_helper("conv4_1-0", network.conv4_1.weights);
    load_bias_helper("conv4_1-1", network.conv4_1.bias);
    
    load_weights_helper("conv4_2-0", network.conv4_2.weights);
    load_bias_helper("conv4_2-1", network.conv4_2.bias);

}



int main(){
   vgg network(0, false);

    load_weights(network);
    return 0;
}
