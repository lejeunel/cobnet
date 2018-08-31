import torch
import torch.nn as nn
from torch import optim
from torchvision import transforms
from torchvision import models
import os
from skimage.transform import resize

# Base model for Convolutional Oriented Boundaries
# Removes the last fully convolutional layer
class MyResnet50(nn.Module):
    def __init__(self,
                 batch_size=1,
                 load_model_path='resnet50.pth',
                 cuda=True):

        super(MyResnet50, self).__init__()

        self.batch_size = batch_size

        # Download original pre-trained Resnet50 if not given
        if(not os.path.exists(load_model_path)):
            self.model = models.resnet50(pretrained=True)
            torch.save(self.model, load_model_path)
        else:
            self.model = torch.load(load_model_path)

        if(cuda):
            self.model.cuda()

        self.side0_shape = (self.batch_size, 64, 112, 112)
        self.side1_shape = (self.batch_size, 256, 56, 56)  
        self.side2_shape = (self.batch_size, 512, 28, 28)  
        self.side3_shape = (self.batch_size, 1024, 14, 14) 
        self.side4_shape = (self.batch_size, 2048, 7, 7) 

        # Initialize tensors
        self.side0_out = torch.zeros(self.side0_shape)
        self.side1_out = torch.zeros(self.side1_shape)
        self.side2_out = torch.zeros(self.side2_shape)
        self.side3_out = torch.zeros(self.side3_shape)
        self.side4_out = torch.zeros(self.side4_shape)

        # Remove avg pool and classification layer (fc)
        #self.model = nn.Sequential(*list(self.model.children())[:-2])

        # Freeze parameters as we don't backprop on it
        #for param in self.model.parameters():
        #    param.requires_grad = False

    def output_tensor_shape(self, idx):
        if(idx == 0):
            return self.side0_shape
        elif(idx == 1):
            return self.side1_shape
        elif(idx == 2):
            return self.side2_shape
        elif(idx == 3):
            return self.side3_shape
        elif(idx == 4):
            return self.side4_shape
        else:
            raise ValueError('Bad output layer requested in MyResNet50.')

    def hook(module, input, output):
        outputs.append(output)

    def output_layer(self, idx):
        if(idx == 0):
            return self.model.conv1
        elif(idx == 1):
            return self.model.layer1[-1].conv3
        elif(idx == 2):
            return self.model.layer2[-1].conv3
        elif(idx == 3):
            return self.model.layer3[-1].conv3
        elif(idx == 4):
            return self.model.layer4[-1].conv3
        else:
            raise ValueError('Bad output layer requested in MyResNet50.')

    def output_tensor(self, idx):
        if(idx == 0):
            return self.side0_out
        elif(idx == 1):
            return self.side1_out
        elif(idx == 2):
            return self.side2_out
        elif(idx == 3):
            return self.side3_out
        elif(idx == 4):
            return self.side4_out
        else:
            raise ValueError('Bad output tensor requested in MyResNet50.')

    def forward(self, x):
        # Returns outputs of conv1 and layer{1, ..., 4}

        # Set hooks
        def copy_side0_data(m, i, o):
            self.side0_out.copy_(o.data)
        def copy_side1_data(m, i, o):
            self.side1_out.copy_(o.data)
        def copy_side2_data(m, i, o):
            self.side2_out.copy_(o.data)
        def copy_side3_data(m, i, o):
            self.side3_out.copy_(o.data)
        def copy_side4_data(m, i, o):
            self.side4_out.copy_(o.data)

        # Attach hooks
        h_side0 = self.output_layer(0).register_forward_hook(copy_side0_data)
        h_side1 = self.output_layer(1).register_forward_hook(copy_side1_data)
        h_side2 = self.output_layer(2).register_forward_hook(copy_side2_data)
        h_side3 = self.output_layer(3).register_forward_hook(copy_side3_data)
        h_side4 = self.output_layer(4).register_forward_hook(copy_side4_data)

        # Pass input through model
        self.model(x)

        # Detach copy functions
        h_side0.remove()
        h_side1.remove()
        h_side2.remove()
        h_side3.remove()
        h_side4.remove()

        return [self.side0_out,
                self.side1_out,
                self.side2_out,
                self.side3_out,
                self.side4_out]

    def single_stage_loss(self, y_pred, y_true):

        y_pred = y_pred.data.numpy().ravel()

        # Calculate re-balancing factor
        beta = 1 - np.sum(Y_true)/np.prod(Y_true.shape)

        width, height = y_pred.shape[2:]

        # Resize y_true to size of current scale
        y_true_resized = resize(y_true, (width, height)).ravel()
        pos_term = np.sum(np.log(y_pred[Y_true_resized == 1]))
        neg_term = np.sum(np.log(y_pred[Y_true_resized == 0]))
        loss += -beta*pos_term - (1-beta)*neg_term

    def multi_stage_loss(self, Y_pred, y_true):
        # Y_pred is a list of predictions for each scale (tensors)
        # Y_true: numpy array (binary)

        loss = 0

        for y_pred in Y_pred:
            loss += single_stage_loss(y_pred, y_true)

        return loss

    #def train(self, x, y):
    #    # x: list of tensors (already normalized for imagenet)
    #    # y: corresponding contour groundtruths (numpy arrays)

    #    # Spawn 1 optimizer per scale
    #    learning_rate = 0.01
    #    n_epochs = 20
    #    optims = [optim.Adam(m.parameters(), lr=learning_rate)
    #              for m in [self.model.conv1,
    #                        self.model.layer1,
    #                        self.model.layer2,
    #                        self.model.layer3,
    #                        self.model.layer4]]
    #    # Begin!
    #    for epoch in range(1, n_epochs + 1):

    #        # Get training data for this cycle
    #        training_pair = variables_from_pair(random.choice(pairs))
    #        input_variable = training_pair[0]
    #        target_variable = training_pair[1]

    #        # Run the train function
    #        loss = train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)

    #        # Keep track of loss
    #        print_loss_total += loss
    #        plot_loss_total += loss

    #        if epoch == 0: continue

    #        if epoch % print_every == 0:
    #            print_loss_avg = print_loss_total / print_every
    #            print_loss_total = 0
    #            print_summary = '%s (%d %d%%) %.4f' % (time_since(start, epoch / n_epochs), epoch, epoch / n_epochs * 100, print_loss_avg)
    #            print(print_summary)

    #        if epoch % plot_every == 0:
    #            plot_loss_avg = plot_loss_total / plot_every
    #            plot_losses.append(plot_loss_avg)
    #            plot_loss_total = 0
