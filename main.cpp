#include <iostream>
#include <vector>
#include <stdint.h>
#include "vgg16.h"

using namespace std;

int main(){
    vgg layer;
    vector<double> temp(224, 1);
    vector<vector<double>> two(224, temp);
    vector<vector<vector<double>>> three(3, two);
    cout<<three.size()<<" ";
    cout<<three[0].size()<<" ";
    cout<<three[0][0].size()<<" ";
    cout<<endl;
    vector<vector<vector<double>>> output = layer.feedForward(three);
    // cout<<output.size()<<" ";
    // cout<<output[0].size()<<" ";
    // cout<<output[0][0].size()<<" ";
}