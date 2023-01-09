from torch import nn
import math
from yolov6.layers.common import BottleRep, RepVGGBlock, RepBlock, BepC3, SimSPPF, SPPF, ConvWrapper, MobileBlock, InvertedResidual, MV2Block, MobileViTBlock


class EfficientRep(nn.Module):
    '''EfficientRep Backbone
    EfficientRep is handcrafted by hardware-aware neural network design.
    With rep-style struct, EfficientRep is friendly to high-computation hardware(e.g. GPU).
    '''

    def __init__(
        self,
        in_channels=3,
        channels_list=None, #[64,128,256,512,1024]
        num_repeats=None, # [1,6,12,18,6]
        block=RepVGGBlock
    ):
        super().__init__()

        assert channels_list is not None
        assert num_repeats is not None

        self.stem = block(
            in_channels=in_channels,
            out_channels=channels_list[0],
            kernel_size=3,
            stride=2
        )

        self.ERBlock_2 = nn.Sequential(
            block(
                in_channels=channels_list[0],
                out_channels=channels_list[1],
                kernel_size=3,
                stride=2
            ),
            RepBlock(
                in_channels=channels_list[1],
                out_channels=channels_list[1],
                n=num_repeats[1],
                block=block,
            )
        )

        self.ERBlock_3 = nn.Sequential(
            block(
                in_channels=channels_list[1],
                out_channels=channels_list[2],
                kernel_size=3,
                stride=2
            ),
            RepBlock(
                in_channels=channels_list[2],
                out_channels=channels_list[2],
                n=num_repeats[2],
                block=block,
            )
        )

        self.ERBlock_4 = nn.Sequential(
            block(
                in_channels=channels_list[2],
                out_channels=channels_list[3],
                kernel_size=3,
                stride=2
            ),
            RepBlock(
                in_channels=channels_list[3],
                out_channels=channels_list[3],
                n=num_repeats[3],
                block=block,
            )
        )

        self.ERBlock_5 = nn.Sequential(
            block(
                in_channels=channels_list[3],
                out_channels=channels_list[4],
                kernel_size=3,
                stride=2,
            ),
            RepBlock(
                in_channels=channels_list[4],
                out_channels=channels_list[4],
                n=num_repeats[4],
                block=block,
            ),
            SimSPPF(
                in_channels=channels_list[4],
                out_channels=channels_list[4],
                kernel_size=5
            )
        )

    def forward(self, x):

        outputs = []
        x = self.stem(x)
        x = self.ERBlock_2(x)
        x = self.ERBlock_3(x)
        outputs.append(x)
        x = self.ERBlock_4(x)
        outputs.append(x)
        x = self.ERBlock_5(x)
        outputs.append(x)

        return tuple(outputs)


class CSPBepBackbone(nn.Module):
    """
    CSPBepBackbone module. 
    """

    def __init__(
        self,
        in_channels=3,
        channels_list=None,
        num_repeats=None,
        block=RepVGGBlock,
        csp_e=float(1)/2,
    ):
        super().__init__()

        assert channels_list is not None
        assert num_repeats is not None

        self.stem = block(
            in_channels=in_channels,
            out_channels=channels_list[0],
            kernel_size=3,
            stride=2
        )

        self.ERBlock_2 = nn.Sequential(
            block(
                in_channels=channels_list[0],
                out_channels=channels_list[1],
                kernel_size=3,
                stride=2
            ),
            BepC3(
                in_channels=channels_list[1],
                out_channels=channels_list[1],
                n=num_repeats[1],
                e=csp_e,
                block=block,
            )
        )

        self.ERBlock_3 = nn.Sequential(
            block(
                in_channels=channels_list[1],
                out_channels=channels_list[2],
                kernel_size=3,
                stride=2
            ),
            BepC3(
                in_channels=channels_list[2],
                out_channels=channels_list[2],
                n=num_repeats[2],
                e=csp_e,
                block=block,
            )
        )

        self.ERBlock_4 = nn.Sequential(
            block(
                in_channels=channels_list[2],
                out_channels=channels_list[3],
                kernel_size=3,
                stride=2
            ),
            BepC3(
                in_channels=channels_list[3],
                out_channels=channels_list[3],
                n=num_repeats[3],
                e=csp_e,
                block=block,
            )
        )

        channel_merge_layer = SimSPPF
        if block == ConvWrapper:
            channel_merge_layer = SPPF

        self.ERBlock_5 = nn.Sequential(
            block(
                in_channels=channels_list[3],
                out_channels=channels_list[4],
                kernel_size=3,
                stride=2,
            ),
            BepC3(
                in_channels=channels_list[4],
                out_channels=channels_list[4],
                n=num_repeats[4],
                e=csp_e,
                block=block,
            ),
            channel_merge_layer(
                in_channels=channels_list[4],
                out_channels=channels_list[4],
                kernel_size=5
            )
        )

    def forward(self, x):

        outputs = []
        x = self.stem(x)
        x = self.ERBlock_2(x)
        x = self.ERBlock_3(x)
        outputs.append(x)
        x = self.ERBlock_4(x)
        outputs.append(x)
        x = self.ERBlock_5(x)
        outputs.append(x)

        return tuple(outputs)

'''xyh 2022add mobilenetv1'''
class MobileNet(nn.Module):
    ''' MobileNet backbone model'''
    def __init__(self, 
    in_planes = 32,
    planes_list = None,
    block = MobileBlock
    ):
        super(MobileNet, self).__init__()

        assert planes_list is not None

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, 
        	stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        '''first layer is normal conv'''
        self.layers = self._make_layers(in_planes=32,planes_list=planes_list, block=MobileBlock)
        # self.linear = nn.Linear(1024, num_classes) do not need linear in backbone
        self.relu = nn.ReLU()

    def _make_layers(self, in_planes, planes_list, block):
        layers = []
        for x in planes_list:

            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(block(in_planes, out_planes, stride))
            in_planes = out_planes

        return nn.Sequential(*layers)

    def forward(self, x):
        output = []
        out = nn.functional.relu(self.bn1(self.conv1(x)))
        '''first layer is normal conv add bn'''
        for index, layer in enumerate(self.layers):

            out = layer(out)
            output.append(out) if index == 4 or index == 10 or index == 12 else None
            '''get the same shape output as repvgg backbone'''

        return tuple(output)

'''xyh  add mobilenetv2'''
class MobileNetV2(nn.Module):
    def __init__(
        self, 
        cfgs=None, 
        block = InvertedResidual,
        width_mult=1.
        ):
        super(MobileNetV2, self).__init__()
        # setting of inverted residual blocks
        assert cfgs is not None
        # building first layer
        input_channel = self._make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        #layers = [conv_3x3_bn(3, input_channel, 2)]
        self.preprocessing = self.conv_3x3_bn(3, input_channel, 2)
        layers = []
        # building inverted residual blocks
        for t, c, n, s in cfgs:
            output_channel = self._make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t))
                input_channel = output_channel
        #buliud layers
        self.features = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        output = []
        x = self.preprocessing(x)
        #x = self.features(x)
        #print(x.shape)
        for index, layer in enumerate(self.features):
            x = layer(x)
            if index == 9 or index == 12 or index == 15:
                output.append(x)
        #print("mobilenetV2 successfully used")
        return tuple(output)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def _make_divisible(self, v, divisor, min_value=None):
    
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v
    
    def conv_3x3_bn(self, inp, oup, stride):
        return nn.Sequential(
            nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU6(inplace=True)
    )

'''xyh add mobilevit'''
class MobileViT(nn.Module):
    def __init__(self, image_size, dims, channels, expansion=4, kernel_size=3, patch_size=(2, 2)):
        super().__init__()
        ih, iw = image_size
        ph, pw = patch_size
        assert ih % ph == 0 and iw % pw == 0
        assert channels is not None
        assert dims is not None

        L = [2, 4, 3]

        self.conv1 = self.Conv_BN_ReLU(3, channels[0], kernel=3, stride=2)

        self.mv2 = nn.ModuleList([])
        self.mv2.append(MV2Block(channels[0], channels[1], 1, expansion))
        self.mv2.append(MV2Block(channels[1], channels[2], 2, expansion))
        self.mv2.append(MV2Block(channels[2], channels[3], 1, expansion))
        self.mv2.append(MV2Block(channels[2], channels[3], 1, expansion))  # Repeat
        self.mv2.append(MV2Block(channels[3], channels[4], 2, expansion))
        self.mv2.append(MV2Block(channels[5], channels[6], 2, expansion))
        self.mv2.append(MV2Block(channels[7], channels[8], 2, expansion))

        self.mvit = nn.ModuleList([])
        self.mvit.append(MobileViTBlock(dims[0], L[0], channels[5], kernel_size, patch_size, int(dims[0] * 2)))
        self.mvit.append(MobileViTBlock(dims[1], L[1], channels[7], kernel_size, patch_size, int(dims[1] * 4)))
        self.mvit.append(MobileViTBlock(dims[2], L[2], channels[9], kernel_size, patch_size, int(dims[2] * 4)))

        self.post = nn.ModuleList([])
        self.post.append(MV2Block(channels[5], channels[-3], 1, expand_ratio=6))
        self.post.append(MV2Block(channels[7], channels[-2], 1, expand_ratio=6))
        self.post.append(MV2Block(channels[9], channels[-1], 1, expand_ratio=6))


        #self.conv2 = conv_1x1_bn(channels[-2], channels[-1])

        #self.pool = nn.AvgPool2d(ih // 32, 1)
        #self.fc = nn.Linear(channels[-1], num_classes, bias=False)

    def forward(self, x):
        output = []
        x = self.conv1(x)
        x = self.mv2[0](x)
        x = self.mv2[1](x)
        x = self.mv2[2](x)
        x = self.mv2[3](x)  # Repeat

        x = self.mv2[4](x)
        x = self.mvit[0](x)
        out = self.post[0](x)
        output.append(out)

        x = self.mv2[5](x)
        x = self.mvit[1](x)
        out = self.post[1](x)
        output.append(out)

        x = self.mv2[6](x)
        x = self.mvit[2](x)
        out = self.post[2](x)
        output.append(out)
        #x = self.conv2(x)

        #x = self.pool(x).view(-1, x.shape[1])
        #x = self.fc(x)
        return tuple(output)

    def Conv_BN_ReLU(self, inp, oup, kernel, stride=1):
        return nn.Sequential(
            nn.Conv2d(inp, oup, kernel_size=kernel, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU6(inplace=True)
        )


'''xyh add new mobilevitV2'''
class MobileViTV2(nn.Module):
    def __init__(self, image_size, dims, channels, expansion=4, kernel_size=3, patch_size=(2, 2)):
        super().__init__()
        ih, iw = image_size
        ph, pw = patch_size
        assert ih % ph == 0 and iw % pw == 0

        L = [2, 4, 3]

        self.conv1 = self.Conv_BN_ReLU(3, channels[0], kernel=3, stride=2)
        
        self.mv2 = nn.ModuleList([])
        self.mv2.append(MV2Block(channels[0], channels[1], 2, expansion))
        self.mv2.append(MV2Block(channels[1], channels[1], 1, expansion))  # Repeat
        self.mv2.append(MV2Block(channels[1], channels[2], 1, expansion))
        self.mv2.append(MV2Block(channels[2], channels[3], 1, expansion))
        self.mv2.append(MV2Block(channels[2], channels[3], 1, expansion))  # Repeat
        self.mv2.append(MV2Block(channels[3], channels[4], 2, expansion))
        self.mv2.append(MV2Block(channels[4], channels[4], 1, expansion))  # Repeat
        self.mv2.append(MV2Block(channels[5], channels[6], 2, expansion))
        self.mv2.append(MV2Block(channels[6], channels[6], 1, expansion))  # Repeat
        self.mv2.append(MV2Block(channels[7], channels[8], 2, expansion))
        self.mv2.append(MV2Block(channels[8], channels[8], 1, expansion))  # Repeat


        self.mvit = nn.ModuleList([])
        self.mvit.append(MobileViTBlock(dims[0], L[0], channels[5], kernel_size, patch_size, int(dims[0]*2)))
        self.mvit.append(MobileViTBlock(dims[1], L[1], channels[7], kernel_size, patch_size, int(dims[1]*4)))
        self.mvit.append(MobileViTBlock(dims[2], L[2], channels[9], kernel_size, patch_size, int(dims[2]*4)))
    
    def forward(self, x):

        output = []
        x = self.conv1(x)
        x = self.mv2[0](x)
        x = self.mv2[1](x)
        x = self.mv2[2](x)
        x = self.mv2[3](x)      # Repeat
        x = self.mv2[4](x)

        x = self.mv2[5](x)
        x = self.mv2[6](x)      # Repeat
        x = self.mvit[0](x)
        output.append(x)

        x = self.mv2[7](x)
        x = self.mv2[8](x)      # Repeat
        x = self.mvit[1](x)
        output.append(x)

        x = self.mv2[9](x)
        x = self.mv2[-1](x)      # Repeat
        x = self.mvit[2](x)
        output.append(x)
        
        return tuple(output)

    def Conv_BN_ReLU(self, inp, oup, kernel, stride=1):
        return nn.Sequential(
            nn.Conv2d(inp, oup, kernel_size=kernel, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU6(inplace=True)
        )


