
#include <iostream>
#include <vector>
#include <stdint.h>
#include "vgg16.h"
#include "cnpy.h"

using namespace std;

int main(){
    vgg layer(0, false);
    vector<float> temp(224, 1);
    vector<vector<float>> two(224, temp);
    vector<vector<vector<float>>> three(3, two);
    cout<<three.size()<<" ";
    cout<<three[0].size()<<" ";
    cout<<three[0][0].size()<<" ";
    cout<<endl;
    vector<vector<vector<float>>> output = layer.feedForward(three);
    // cout<<output.size()<<" ";
    // cout<<output[0].size()<<" ";
    // cout<<output[0][0].size()<<" ";
}