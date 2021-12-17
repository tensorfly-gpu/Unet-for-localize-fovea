import paddle
import paddle.nn as nn
from paddle.io import Dataset
from paddle.nn import Linear,Dropout,BatchNorm1D,LeakyReLU,Sigmoid,ReLU
import paddle.nn.functional as F
import paddle.tensor as tensor

import numpy as np

class MLP(nn.Layer):
    def __init__(self):
        super(MLP,self).__init__()

        self.fc2 = nn.Sequential(
            Linear(875,256),
            ReLU(),
            BatchNorm1D(256),
            Dropout(0.2),

            Linear(256, 64),
            ReLU(),
            BatchNorm1D(64),
            Dropout(0.2),

            Linear(64, 32),
            ReLU(),
            BatchNorm1D(32),
            Dropout(0.2),
        )

        self.fc1 = nn.Sequential(
            Linear(22,64),
            ReLU(),
            BatchNorm1D(64),
            Dropout(0.2),

            Linear(64, 32),
            ReLU(),
            BatchNorm1D(32),
            Dropout(0.2),
        )

        self.fc3 = nn.Sequential(
            Linear(32, 32),
            ReLU(),
            BatchNorm1D(32),
            Dropout(0.1),

            Linear(32, 16),
            ReLU(),
            BatchNorm1D(16),

            Linear(16, 2),
        )

    def data_precess(self,inputs):
        data1 = inputs.numpy()[:,0:22]
        data2 = inputs.numpy()[:,22:]

        data1 = paddle.to_tensor(data1)
        data2 = paddle.to_tensor(data2)

        return data1,data2

    def forward(self,inputs):
        data1,data2 = self.data_precess(inputs)
        x = self.fc1(data1)
        y = self.fc2(data2)
        output = x+y
        output = self.fc3(output)
        return output


#FCNresnet

#define convBN block
class ConvBNLayer(nn.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 stride=1,
                 groups=1,
                 act=None):
        super(ConvBNLayer, self).__init__()

        self._conv = nn.Conv2D(in_channels=num_channels,
                               out_channels=num_filters,
                               kernel_size=filter_size,
                               stride=stride,
                               padding=(filter_size-1)//2,
                               groups=groups,
                               bias_attr=False)

        self._batch_norm = paddle.nn.BatchNorm2D(num_filters)

        self.act = act

    def forward(self,inputs):
        y = self._conv(inputs)
        y = self._batch_norm(y)
        if self.act == 'leaky':
            y = F.leaky_relu(x=y,negative_slope=0.1)
        elif self.act == 'relu':
            y = F.relu(x=y)
        return y

#define residual block
class BottleneckBlock(nn.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 stride,
                 shortcut=True):
        super(BottleneckBlock, self).__init__()

        #build first conv 1x1
        self.conv0 = ConvBNLayer(num_channels=num_channels,
                                 num_filters=num_filters,
                                 filter_size=1,
                                 act='relu')

        #build second conv 3x3
        self.conv1 = ConvBNLayer(num_channels=num_filters,
                                 num_filters=num_filters,
                                 filter_size=3,
                                 stride=stride,
                                 act='relu')

        #bulid third conv 1x1,but out channels x 4
        self.conv2 = ConvBNLayer(num_channels=num_filters,
                                 num_filters=num_filters * 4,
                                 filter_size=1,
                                 act=None)

        if not shortcut:
            self.short = ConvBNLayer(num_channels=num_channels,
                                     num_filters=num_filters * 4,
                                     filter_size=1,
                                     stride=stride)

        self.shortcut = shortcut

        self._num_channels_out = num_filters * 4

    def forward(self,inputs):
        y = self.conv0(inputs)
        conv1 = self.conv1(y)
        conv2 = self.conv2(conv1)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)

        y = paddle.add(x=short,y=conv2)
        y = F.relu(y)
        return y

#U-net
class Conv_Block(nn.Layer):
    def __init__(self,in_channels,out_channels):
        super(Conv_Block, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2D(
                    in_channels,
                    out_channels,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                    padding_mode='reflect',bias_attr=False),
            nn.BatchNorm2D(out_channels),
            nn.Dropout2D(p=0.2),
            nn.ReLU(),
            nn.Conv2D(
                    out_channels,
                    out_channels,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                    padding_mode='reflect', bias_attr=False),
            nn.BatchNorm2D(out_channels),
            nn.Dropout2D(p=0.2),
            nn.ReLU(),
        )

    def forward(self,x):
        return self.layer(x)

class DownSample(nn.Layer):
    def __init__(self,channels):
        super(DownSample, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2D(
                in_channels=channels,
                out_channels=channels,
                kernel_size=5,
                stride=2,
                padding=2,
                padding_mode='reflect', bias_attr=False),
            nn.BatchNorm2D(channels),
            nn.ReLU(),
        )
    def forward(self,x):
        return self.layer(x)

class UpSample(nn.Layer):
    def __init__(self,channels):
        super(UpSample, self).__init__()
        self.layer = nn.Conv2D(channels,channels//2,kernel_size=1,stride=1)

    def forward(self,x,y):
        x = F.interpolate(x,scale_factor=2,mode='nearest')
        x = self.layer(x)
        return paddle.concat((x,y),axis=1)    #NCHW

class Unet(nn.Layer):
    def __init__(self):
        super(Unet, self).__init__()
        self.c1 = Conv_Block(in_channels=3,out_channels=64)
        self.d1 = DownSample(64)
        self.c2 = Conv_Block(in_channels=64, out_channels=128)
        self.d2 = DownSample(128)
        self.c3 = Conv_Block(in_channels=128, out_channels=256)
        self.d3 = DownSample(256)
        self.c4 = Conv_Block(in_channels=256, out_channels=512)
        self.d4 = DownSample(512)
        self.c5 = Conv_Block(in_channels=512, out_channels=1024)

        self.u1 = UpSample(1024)
        self.c6 = Conv_Block(in_channels=1024, out_channels=512)
        self.u2 = UpSample(512)
        self.c7 = Conv_Block(in_channels=512, out_channels=256)
        self.u3 = UpSample(256)
        self.c8 = Conv_Block(in_channels=256, out_channels=128)
        self.u4 = UpSample(128)
        self.c9 = Conv_Block(in_channels=128, out_channels=64)

        self.out = nn.Conv2D(in_channels=64,out_channels=1,kernel_size=5,stride=1,padding=2,padding_mode='reflect',bias_attr=False)
        self.TH = nn.ReLU()

    def forward(self,x):
        r1 = self.c1(x)
        r2 = self.c2(self.d1(r1))
        r3 = self.c3(self.d2(r2))
        r4 = self.c4(self.d3(r3))
        r5 = self.c5(self.d4(r4))

        o1 = self.c6(self.u1(r5,r4))
        o2 = self.c7(self.u2(o1,r3))
        o3 = self.c8(self.u3(o2,r2))
        o4 = self.c9(self.u4(o3, r1))


        return self.TH(self.out(o4))


def train(model,train_loader,train_features,train_labels,valid_features,valid_labels):
    use_gpu = True
    paddle.set_device('gpu:0') if use_gpu else paddle.set_device('cpu')

    model.train()

    opt = paddle.optimizer.Adam(learning_rate=0.0001,parameters=model.parameters(),weight_decay=paddle.regularizer.L2Decay(coeff=0.005))
    EPOCH_NUM = 10000

    for epoch_id in range(EPOCH_NUM):

        for batch_id,data in enumerate(train_loader()):
            features,labels = data
            features = paddle.to_tensor(features)
            labels = paddle.to_tensor(labels)

            predicts = model(features)

            loss = F.square_error_cost(predicts,labels)

            avg_loss = paddle.mean(loss)
            avg_loss.backward()
            opt.step()

            opt.clear_grad()
        if epoch_id % 100 == 0:
            print('epoch_id:{},loss:{}'.format(epoch_id,avg_loss))
        if epoch_id % 1000 == 0:
            evaluate(model,train_features=train_features,train_labels=train_labels,valid_features=valid_features,valid_labels=valid_labels)

if __name__ == '__main__':
    x = paddle.randn((4,3,400,400))
    use_gpu = False
    paddle.set_device('gpu:0') if use_gpu else paddle.set_device('cpu')

    Unet = Unet()
    print(Unet(x).shape)