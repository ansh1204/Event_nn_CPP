#include <iostream>
#include <vector>
#include <stdint.h>
#include "vgg16.h"
#include <cnpy.h>
#include<complex>
#include <fstream>
#include <iterator>
#include <vector>
#include <bits/stdc++.h>
#include <iostream>
#include <typeinfo>

using namespace std;
using namespace std::chrono;


void load_weights_helper(string a, vector<vector<vector<vector<float>>>> &wt, string path) {
    cnpy::NpyArray weights = cnpy::npz_load(path,a);
    float* temp = weights.data<float>();
    int n = weights.shape[0]*weights.shape[1]*weights.shape[2]*weights.shape[3];
    int outC = wt.size();
    int inC = wt[0].size();
    int h = wt[0][0].size();
    int w = wt[0][0][0].size();
    int prod1 = inC*h*w;
    int prod2 = h*w;
    int prod3 = w;
    for(int i = 0; i < n; i++) {
        int idx1 = i/prod1;
        int idx2 = (i%prod1)/prod2;
        int idx3 = (i%prod2)/prod3;
        int idx4 = (i%prod3);
        wt[idx1][idx2][idx3][idx4] = temp[i];
    }
}

void load_bias_helper(string a, vector<float> &bias, string path) {
    cnpy::NpyArray bs = cnpy::npz_load(path,a);
    float* temp = bs.data<float>();
    int n = bs.shape[0];
    for(int i = 0; i < n; i++) {
        bias[i] = temp[i];
    }
}

void load_weights(vgg &network, string path){
    load_weights_helper("conv1_1-0", network.conv1_1.weights, path);
    load_bias_helper("conv1_1-1", network.conv1_1.bias, path);
    
    load_weights_helper("conv1_2-0", network.conv1_2.weights, path);
    load_bias_helper("conv1_2-1", network.conv1_2.bias, path);
    
    load_weights_helper("conv2_1-0", network.conv2_1.weights, path);
    load_bias_helper("conv2_1-1", network.conv2_1.bias, path);
    
    load_weights_helper("conv2_2-0", network.conv2_2.weights, path);
    load_bias_helper("conv2_2-1", network.conv2_2.bias, path);
    
    load_weights_helper("conv3_1-0", network.conv3_1.weights, path);
    load_bias_helper("conv3_1-1", network.conv3_1.bias, path);
    
    load_weights_helper("conv3_2-0", network.conv3_2.weights, path);
    load_bias_helper("conv3_2-1", network.conv3_2.bias, path);
    
    load_weights_helper("conv3_3-0", network.conv3_3.weights, path);
    load_bias_helper("conv3_3-1", network.conv3_3.bias, path);
    
    load_weights_helper("conv3_4-0", network.conv3_4.weights, path);
    load_bias_helper("conv3_4-1", network.conv3_4.bias, path);

    load_weights_helper("conv4_1-0", network.conv4_1.weights, path);
    load_bias_helper("conv4_1-1", network.conv4_1.bias, path);
    
    load_weights_helper("conv4_2-0", network.conv4_2.weights, path);
    load_bias_helper("conv4_2-1", network.conv4_2.bias, path);

}


void load_image(vector<vector<vector<vector<float>>>> &input, string path){
    cnpy::NpyArray weights = cnpy::npy_load(path);
    int n = weights.shape[0]*weights.shape[1]*weights.shape[2]*weights.shape[3];
    float* temp = weights.data<float>();
    int total_images = input.size();
    int channels = input[0].size();
    int height = input[0][0].size();
    int width = input[0][0][0].size();

    int prod1 = channels*height*width;
    int prod2 = height*width;
    int prod3 = width;
    for(int i = 0; i < n; i++){
        int idx1 = i/prod1;
        int idx2 = (i%prod1)/prod2;
        int idx3 = (i%prod2)/prod3;
        int idx4 = (i%prod3);
        input[idx1][idx2][idx3][idx4] = temp[i];
    }
    
}
int main(int argc, char** argv){

    string path = "/home/ansh/Desktop/IS/openpose_mpii.npz";
    bool eventNN = false;
    float threshold = 0.0;
    vgg network(threshold, eventNN);

    load_weights(network, path);
    int total_images = 201;
    int height = 152; 
    int width = 270;
    int channels = 3;
    if(eventNN) {
        // call flush first
        vector<float> temp(width, 0);
        vector<vector<float>> two(height, temp);
        vector<vector<vector<float>>> three(3, two);
        network.flush(three);
    }
    vector<vector<vector<vector<float>>>> input = vector<vector<vector<vector<float>>>>(total_images, vector<vector<vector<float>>>(3, vector<vector<float>>(height, vector<float>(width,0))));

    load_image(input, "/home/ansh/Desktop/IS/wave.npy");

    cout<<"STARTING"<<endl;
    auto start = high_resolution_clock::now();
    vector<vector<vector<float>>> net_out;
    for(int i = 0; i < 1; i++) {
        net_out = network.feedForward(input[i]);
        cout<<"=========================================================="<<endl;
    }
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "Time taken by function: "<< duration.count() << " microseconds" << endl;

    for(int i = 0; i < net_out[0].size(); i++) {
        for(int j = 0; j < net_out[0][0].size(); j++) {
            cout<<net_out[1][i][j]<<" ";
        }
        cout<<endl;
    }

    cout<<endl;
    cout<<endl;
    cout<<endl;
        
    for(int i = 0; i < net_out[0].size(); i++) {
        for(int j = 0; j < net_out[0][0].size(); j++) {
            cout<<net_out[50][i][j]<<" ";
        }
        cout<<endl;
    }
}
