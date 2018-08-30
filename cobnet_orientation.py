import torch.nn as nn

class CobNetOrientationModule(nn.Module):
    def __init__(self, base_model, orientation):

        super(CobNetOrientationModule, self).__init__()
        # This is the resnet50
        # Calling its forward function returns outputs of 5 scale block
        # This will give the inputs of the layers (one per scale)
        self.base_model = base_model

        self.orientation = orientation

        # Build it
        self.models = self.make_models()
        self.out_model = self.make_output_model()
        
    def forward(self, x):

        outputs = dict()
        for s in self.models.keys():
            outputs[s] = self.models[s](x)

        #Concatenate all outputs
        cat_outputs = torch.cat([v for v,_ in outputs.items()])

        return self.out_model(cat_outputs)

    def make_models(self):
        # From model:
        # https://github.com/kmaninis/COB/blob/master/models/deploy.prototxt
        #
        # This will be part of a stack of depth 5 (1 for each fine->coarse level)
        #
        # Layer 1-------------
        # conv2d:
        # weights learning rate multiplier: 1
        # weights decay multiplier: 1
        # bias learning rate multiplier: 2
        # bias decay multiplier: 0
        # n_outputs: 32, pad:1, kernel_size: 3
        # weight fillter: gaussian, std: 0.01
        #------------------------------------
        # Layer 2-------------
        # conv2d:
        # weights learning rate multiplier: 1
        # weights decay multiplier: 1
        # bias learning rate multiplier: 2
        # bias decay multiplier: 0
        # n_outputs: 4, pad:1, kernel_size: 3
        # weight fillter: gaussian, std: 0.01
        # Layer 3-------------
        # crop:

        model = dict()

        # From scale 0
        conv_1 = nn.Conv2d(
            in_channels=self.base_model.model.conv1.out_channels,
            out_channels=32,
            kernel_size=3,
            padding=1)
        conv_2 = nn.Conv2d(
            in_channels=conv_1.out_channels,
            out_channels=4,
            kernel_size=3,
            padding=1)

        model[0] = [conv_1, conv_2]

        # From scale 1
        conv_3 = nn.Conv2d(
            in_channels=self.base_model.model.layer1[-1].conv3.out_channels,
            out_channels=32,
            kernel_size=3,
            padding=1)
        conv_4 = nn.Conv2d(
            in_channels=conv_3.out_channels,
            out_channels=4,
            kernel_size=3,
            padding=1)
        deconv_1 = nn.ConvTranspose2d(
            in_channels=conv_4.out_channels,
            out_channels=4,
            stride=2,
            kernel_size=4)

        model[1] = [conv_3, conv_4, deconv_1]

        # From scale 2
        conv_5 = nn.Conv2d(
            in_channels=self.base_model.model.layer2[-1].conv3.out_channels,
            out_channels=32,
            kernel_size=3,
            padding=1)
        conv_6 = nn.Conv2d(
            in_channels=conv_5.out_channels,
            out_channels=4,
            kernel_size=3,
            padding=1)
        deconv_2 = nn.ConvTranspose2d(
            in_channels=conv_6.out_channels,
            out_channels=4,
            stride=4,
            kernel_size=8)

        model[2] = [conv_5, conv_6, deconv_2]

        # From scale 3
        conv_7 = nn.Conv2d(
            in_channels=self.base_model.model.layer3[-1].conv3.out_channels,
            out_channels=32,
            kernel_size=3,
            padding=1)
        conv_8 = nn.Conv2d(
            in_channels=conv_7.out_channels,
            out_channels=4,
            kernel_size=3,
            padding=1)
        deconv_3 = nn.ConvTranspose2d(
            in_channels=conv_8.out_channels,
            out_channels=4,
            stride=4,
            kernel_size=8)

        model[3] = [conv_7, conv_8, deconv_3]

        # From scale 4
        conv_9 = nn.Conv2d(
            in_channels=self.base_model.model.layer4[-1].conv3.out_channels,
            out_channels=32,
            kernel_size=3,
            padding=1)
        conv_10 = nn.Conv2d(
            in_channels=conv_9.out_channels,
            out_channels=4,
            kernel_size=3,
            padding=1)
        deconv_3 = nn.ConvTranspose2d(
            in_channels=conv_10.out_channels,
            out_channels=4,
            stride=16,
            kernel_size=32)

        model[4] = [conv_9, conv_10, deconv_3]

        # transform to sequential models
        for k in model.keys():
            model[k] = nn.Sequential(*model[k])

        #initialize convolutional layers
        for _,m in model.items():
            for t in m:
                if(isinstance(t, nn.Conv2d)):
                    nn.init.normal_(t.weight, mean=0, std=0.01)

        return model

    def make_output_model(self):
        # Will take concatenated outputs of main model, apply 2dconv and a sigmoid

        model = list()
    
        conv = nn.Conv2d(
            in_channels=4,
            out_channels=1,
            kernel_size=3,
            padding=1)

        sigmoid = nn.Sigmoid()

        return nn.Sequential(*[conv, sigmoid])
