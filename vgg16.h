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
        vgg();
        vector<vector<vector<double>>> feedForward(vector<vector<vector<double>>> &input);
};
vgg::vgg(){
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
vector<vector<vector<double>>> vgg::feedForward(vector<vector<vector<double>>> &input){
    vector<vector<vector<double>>> net_out;
    net_out = conv1_1.feedForward(input);
    net_out = relu.feedForward(net_out);
    cout<<net_out.size()<<endl;
    cout<<net_out[0].size()<<endl;
    cout<<net_out[0][0].size()<<endl;

    net_out = conv1_2.feedForward(net_out);
    net_out = relu.feedForward(net_out);
    cout<<net_out.size()<<endl;
    cout<<net_out[0].size()<<endl;
    cout<<net_out[0][0].size()<<endl;
    
    net_out = maxpool.feedForward(net_out);
    cout<<net_out.size()<<endl;
    cout<<net_out[0].size()<<endl;
    cout<<net_out[0][0].size()<<endl;
    
    net_out = conv2_1.feedForward(net_out);
    net_out = relu.feedForward(net_out);
    cout<<net_out.size()<<endl;
    cout<<net_out[0].size()<<endl;
    cout<<net_out[0][0].size()<<endl;
    
    net_out = conv2_2.feedForward(net_out);
    net_out = relu.feedForward(net_out);
    cout<<net_out.size()<<endl;
    cout<<net_out[0].size()<<endl;
    cout<<net_out[0][0].size()<<endl;
    
    net_out = maxpool.feedForward(net_out);
    cout<<net_out.size()<<endl;
    cout<<net_out[0].size()<<endl;
    cout<<net_out[0][0].size()<<endl;
    
    net_out = conv3_1.feedForward(net_out);
    net_out = relu.feedForward(net_out);
    cout<<net_out.size()<<endl;
    cout<<net_out[0].size()<<endl;
    cout<<net_out[0][0].size()<<endl;
    
    net_out = conv3_2.feedForward(net_out);
    net_out = relu.feedForward(net_out);
    cout<<net_out.size()<<endl;
    cout<<net_out[0].size()<<endl;
    cout<<net_out[0][0].size()<<endl;
    
    net_out = conv3_3.feedForward(net_out);
    net_out = relu.feedForward(net_out);
    cout<<net_out.size()<<endl;
    cout<<net_out[0].size()<<endl;
    cout<<net_out[0][0].size()<<endl;
    
    net_out = maxpool.feedForward(net_out);
    cout<<net_out.size()<<endl;
    cout<<net_out[0].size()<<endl;
    cout<<net_out[0][0].size()<<endl;
    
    net_out = conv4_1.feedForward(net_out);
    net_out = relu.feedForward(net_out);
    cout<<net_out.size()<<endl;
    cout<<net_out[0].size()<<endl;
    cout<<net_out[0][0].size()<<endl;
    
    net_out = conv4_2.feedForward(net_out);
    net_out = relu.feedForward(net_out);
    cout<<net_out.size()<<endl;
    cout<<net_out[0].size()<<endl;
    cout<<net_out[0][0].size()<<endl;
    
    net_out = conv4_3.feedForward(net_out);
    net_out = relu.feedForward(net_out);
    cout<<net_out.size()<<endl;
    cout<<net_out[0].size()<<endl;
    cout<<net_out[0][0].size()<<endl;
    
    net_out = maxpool.feedForward(net_out);
    cout<<net_out.size()<<endl;
    cout<<net_out[0].size()<<endl;
    cout<<net_out[0][0].size()<<endl;
    
    net_out = conv5_1.feedForward(net_out);
    net_out = relu.feedForward(net_out);
    cout<<net_out.size()<<endl;
    cout<<net_out[0].size()<<endl;
    cout<<net_out[0][0].size()<<endl;
    
    net_out = conv5_2.feedForward(net_out);
    net_out = relu.feedForward(net_out);
    cout<<net_out.size()<<endl;
    cout<<net_out[0].size()<<endl;
    cout<<net_out[0][0].size()<<endl;
    
    net_out = conv5_3.feedForward(net_out);
    net_out = relu.feedForward(net_out);
    cout<<net_out.size()<<endl;
    cout<<net_out[0].size()<<endl;
    cout<<net_out[0][0].size()<<endl;
    
    net_out = maxpool.feedForward(net_out);
    cout<<net_out.size()<<endl;
    cout<<net_out[0].size()<<endl;
    cout<<net_out[0][0].size()<<endl;
    
    return net_out;
}

#endif