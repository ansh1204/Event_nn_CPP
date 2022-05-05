#ifndef VGG
#define VGG

#include <vector>
#include "convolution.h"
#include "maxpool.h"
#include "relu.h"
#include "gate.h"

using namespace std;

class vgg{
    private:
        
    public:
        Gate gate1_1, gate1_2, gate2_1, gate2_2, gate3_1, gate3_2, gate3_3, gate3_4, gate4_1, gate4_2, gate4_3, gate5_1, gate5_2, gate5_3;
        ConvNet conv1_1, conv1_2, conv2_1, conv2_2, conv3_1, conv3_2, conv3_3, conv3_4, conv4_1, conv4_2, conv4_3, conv5_1, conv5_2, conv5_3;
        MaxPool maxpool;
        ReLU relu;
        bool eventNN;

        vgg(){}
        vgg(unsigned threshold, bool isEvent);
        vector<vector<vector<float>>> feedForward(vector<vector<vector<float>>> &input);
        vector<vector<vector<float>>> feedForwardStandard(vector<vector<vector<float>>> &input);
        vector<vector<vector<float>>> feedForwardEventNN(vector<vector<vector<float>>> &input);
        vector<vector<vector<float>>> flush(vector<vector<vector<float>>> &input);

};
vgg::vgg(unsigned threshold, bool isEvent){
    eventNN = isEvent;

    if(eventNN)
        gate1_1 = Gate(threshold);
    conv1_1 = ConvNet(3,64,3,1,1, isEvent);

    if(eventNN)
        gate1_2 = Gate(threshold);
    conv1_2 = ConvNet(64,64,3,1,1, isEvent);

    if(eventNN)
        gate2_1 = Gate(threshold);
    conv2_1 = ConvNet(64,128,3,1,1, isEvent);

    if(eventNN)
        gate2_2 = Gate(threshold);
    conv2_2 = ConvNet(128,128,3,1,1,isEvent);

    if(eventNN)
        gate3_1 = Gate(threshold);
    conv3_1 = ConvNet(128,256,3,1,1,isEvent);

    if(eventNN)
        gate3_2 = Gate(threshold);
    conv3_2 = ConvNet(256,256,3,1,1,isEvent);

    if(eventNN)
        gate3_3 = Gate(threshold);
    conv3_3 = ConvNet(256,256,3,1,1,isEvent);

    if(eventNN)
        gate3_4 = Gate(threshold);
    conv3_4 = ConvNet(256,256,3,1,1,isEvent);

    if(eventNN)
        gate4_1 = Gate(threshold);
    conv4_1 = ConvNet(256,512,3,1,1,isEvent);

    if(eventNN)
        gate4_2 =Gate(threshold);
    conv4_2 = ConvNet(512,512,3,1,1,isEvent);

    if(eventNN)
        gate4_3 = Gate(threshold);
    conv4_3 = ConvNet(512,512,3,1,1,isEvent);

    if(eventNN)
        gate5_1 = Gate(threshold);
    conv5_1 = ConvNet(512,512,3,1,1,isEvent);

    if(eventNN)
        gate5_2 = Gate(threshold);
    conv5_2 = ConvNet(512,512,3,1,1,isEvent);

    if(eventNN)
        gate5_3 = Gate(threshold);
    conv5_3 = ConvNet(512,512,3,1,1,isEvent);

    maxpool = MaxPool(2,2);
    relu = ReLU();

}

vector<vector<vector<float>>> vgg::feedForward(vector<vector<vector<float>>> &input) {
    if(eventNN){
        return feedForwardEventNN(input);
    } else {
        return feedForwardStandard(input);
    }
}

vector<vector<vector<float>>> vgg::feedForwardEventNN(vector<vector<vector<float>>> &input){
    vector<vector<vector<float>>> net_out;

    net_out = gate1_1.feedForward(input);
    net_out = conv1_1.feedForward(net_out);
    net_out = relu.feedForward(net_out);

    net_out = gate1_2.feedForward(net_out);
    net_out = conv1_2.feedForward(net_out);
    net_out = relu.feedForward(net_out);
    net_out = maxpool.feedForward(net_out);

    net_out = gate2_1.feedForward(net_out);
    net_out = conv2_1.feedForward(net_out);
    net_out = relu.feedForward(net_out);    
	
    net_out = gate2_2.feedForward(net_out);
    net_out = conv2_2.feedForward(net_out);
    net_out = relu.feedForward(net_out);
    net_out = maxpool.feedForward(net_out);
	
    net_out = gate3_1.feedForward(net_out);
    net_out = conv3_1.feedForward(net_out);
    net_out = relu.feedForward(net_out);
    
    net_out = gate3_2.feedForward(net_out);
    net_out = conv3_2.feedForward(net_out);
    net_out = relu.feedForward(net_out);
    
    net_out = gate3_3.feedForward(net_out);
    net_out = conv3_3.feedForward(net_out);
    net_out = relu.feedForward(net_out);

    net_out = gate3_4.feedForward(net_out);
    net_out = conv3_4.feedForward(net_out);
    net_out = relu.feedForward(net_out);
    net_out = maxpool.feedForward(net_out);

    net_out = gate4_1.feedForward(net_out);
    net_out = conv4_1.feedForward(net_out);
    net_out = relu.feedForward(net_out);

    net_out = gate4_2.feedForward(net_out);
    net_out = conv4_2.feedForward(net_out);
    net_out = relu.feedForward(net_out);
    
    return net_out;
}

vector<vector<vector<float>>> vgg::flush(vector<vector<vector<float>>> &input) {
    vector<vector<vector<float>>> net_out;

    net_out = gate1_1.flush(input);
    net_out = conv1_1.flush(net_out);
    net_out = relu.feedForward(net_out);

    net_out = gate1_2.flush(net_out);
    net_out = conv1_2.flush(net_out);
    net_out = relu.feedForward(net_out);

    net_out = maxpool.feedForward(net_out);

    net_out = gate2_1.flush(net_out);
    net_out = conv2_1.flush(net_out);
    net_out = relu.feedForward(net_out);

    net_out = gate2_2.flush(net_out);
    net_out = conv2_2.flush(net_out);
    net_out = relu.feedForward(net_out);
    net_out = maxpool.feedForward(net_out);

    net_out = gate3_1.flush(net_out);
    net_out = conv3_1.flush(net_out);
    net_out = relu.feedForward(net_out);

    net_out = gate3_2.flush(net_out);
    net_out = conv3_2.flush(net_out);
    net_out = relu.feedForward(net_out);

    net_out = gate3_3.flush(net_out);
    net_out = conv3_3.flush(net_out);
    net_out = relu.feedForward(net_out);

    net_out = gate3_4.flush(net_out);
    net_out = conv3_4.flush(net_out);
    net_out = relu.feedForward(net_out);
    net_out = maxpool.feedForward(net_out);

    net_out = gate4_1.flush(net_out);
    net_out = conv4_1.flush(net_out);
    net_out = relu.feedForward(net_out);

    net_out = gate4_2.flush(net_out);
    net_out = conv4_2.flush(net_out);
    net_out = relu.feedForward(net_out);

    return net_out;
}

vector<vector<vector<float>>> vgg::feedForwardStandard(vector<vector<vector<float>>> &input) {
    vector<vector<vector<float>>> net_out;
    
    net_out = conv1_1.feedForward(input);
    net_out = relu.feedForward(net_out);

    net_out = conv1_2.feedForward(net_out);
    net_out = relu.feedForward(net_out);
    net_out = maxpool.feedForward(net_out);

    net_out = conv2_1.feedForward(net_out);
    net_out = relu.feedForward(net_out);

    net_out = conv2_2.feedForward(net_out);
    net_out = relu.feedForward(net_out);
    net_out = maxpool.feedForward(net_out);

    net_out = conv3_1.feedForward(net_out);
    net_out = relu.feedForward(net_out);

    net_out = conv3_2.feedForward(net_out);
    net_out = relu.feedForward(net_out);

    net_out = conv3_3.feedForward(net_out);
    net_out = relu.feedForward(net_out);

    net_out = conv3_4.feedForward(net_out);
    net_out = relu.feedForward(net_out);
    net_out = maxpool.feedForward(net_out);

    net_out = conv4_1.feedForward(net_out);
    net_out = relu.feedForward(net_out);

    net_out = conv4_2.feedForward(net_out);
    net_out = relu.feedForward(net_out);

    return net_out;
}

#endif
