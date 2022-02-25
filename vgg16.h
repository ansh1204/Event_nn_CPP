#ifndef VGG
#define VGG

#include <vector>
#include "convolution.h"
#include "maxpool.h"
#include "relu.h"

using namespace std;

class vgg{
    private:
        ConvNet conv1_1;
        ConvNet conv1_2;
        ConvNet conv2_1;
        ConvNet conv2_2;
        ConvNet conv3_1;
        ConvNet conv3_2;
        ConvNet conv3_3;
        ConvNet conv4_1;
        ConvNet conv4_2;
        ConvNet conv4_3;
        ConvNet conv5_1;
        ConvNet conv5_2;
        ConvNet conv5_3;
        MaxPool maxpool;
        ReLU relu;

    public:
        vgg(){
            conv1_1 = ConvNet(3,64,3,1,1);
            conv1_2 = ConvNet(64,64,3,1,1);
            conv2_1 = ConvNet(64,128,3,1,1);
            conv2_2 = ConvNet(128,128,3,1,1);
            conv3_1 = ConvNet(128,256,3,1,1);
            conv3_2 = ConvNet(256,256,3,1,1);
            conv3_3 = ConvNet(256,256,3,1,1);
            conv4_1 = ConvNet(256,512,3,1,1);
            conv4_2 = ConvNet(512,512,3,1,1);
            conv4_3 = ConvNet(512,512,3,1,1);
            conv5_1 = ConvNet(512,512,3,1,1);
            conv5_2 = ConvNet(512,512,3,1,1);
            conv5_3 = ConvNet(512,512,3,1,1);
            maxpool = MaxPool(2,2);
            relu = ReLU();
        }
        vector<vector<double>> feedForward();
};

vector<vector<double>> vgg::feedForward(){
    vector<vector<double>> net_out;
    net_out.clear();

    return net_out;
}

#endif