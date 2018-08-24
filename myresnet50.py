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
                 load_model_path=os.path.join('models', 'resnet50.pth'),
                 cuda=True):

        super(MyResnet50, self).__init__()

        # Download original pre-trained Resnet50 if not given
        if(not os.path.exists(load_model_path)):
            self.model = models.resnet50(pretrained=True)
            torch.save(self.model, load_model_path)
        else:
            self.model = torch.load(load_model_path)

        if(cuda):
            self.model.cuda()

        # Remove avg pool and classification layer (fc)
        #self.model = nn.Sequential(*list(self.model.children())[:-2])

        # Freeze parameters as we don't backprop on it
        #for param in self.model.parameters():
        #    param.requires_grad = False

    def hook(module, input, output):
        outputs.append(output)

    def forward(self, x):
    #    # Returns outputs of conv1 and layer{1, ..., 4}

        # Initialize tensors
        conv1_out = torch.zeros((1, 64, 112, 112))
        layer1_out = torch.zeros((1, 256, 56, 56))
        layer2_out = torch.zeros((1, 512, 28, 28))
        layer3_out = torch.zeros((1, 1024, 14, 14))
        layer4_out = torch.zeros((1, 2048, 7, 7))

        # Set hooks
        def copy_conv1_data(m, i, o):
            conv1_out.copy_(o.data)
        def copy_layer1_data(m, i, o):
            layer1_out.copy_(o.data)
        def copy_layer2_data(m, i, o):
            layer2_out.copy_(o.data)
        def copy_layer3_data(m, i, o):
            layer3_out.copy_(o.data)
        def copy_layer4_data(m, i, o):
            layer4_out.copy_(o.data)

        # Attach hooks
        h_conv1 = self.model.conv1.register_forward_hook(copy_conv1_data)
        h_layer1 = self.model.layer1.register_forward_hook(copy_layer1_data)
        h_layer2 = self.model.layer2.register_forward_hook(copy_layer2_data)
        h_layer3 = self.model.layer3.register_forward_hook(copy_layer3_data)
        h_layer4 = self.model.layer4.register_forward_hook(copy_layer4_data)

        # Pass input through model
        self.model(x)

        # Detach copy functions
        h_conv1.remove()
        h_layer1.remove()
        h_layer2.remove()
        h_layer3.remove()
        h_layer4.remove()

        return [conv1_out, layer1_out, layer2_out, layer3_out, layer4_out]

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

    def train(self, x, y):
        # x: list of tensors (already normalized for imagenet)
        # y: corresponding contour groundtruths (numpy arrays)

        # Spawn 1 optimizer per scale
        learning_rate = 0.01
        n_epochs = 20
        optims = [optim.Adam(m.parameters(), lr=learning_rate)
                  for m in [self.model.conv1,
                            self.model.layer1,
                            self.model.layer2,
                            self.model.layer3,
                            self.model.layer4]]
        # Begin!
        for epoch in range(1, n_epochs + 1):

            # Get training data for this cycle
            training_pair = variables_from_pair(random.choice(pairs))
            input_variable = training_pair[0]
            target_variable = training_pair[1]

            # Run the train function
            loss = train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)

            # Keep track of loss
            print_loss_total += loss
            plot_loss_total += loss

            if epoch == 0: continue

            if epoch % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print_summary = '%s (%d %d%%) %.4f' % (time_since(start, epoch / n_epochs), epoch, epoch / n_epochs * 100, print_loss_avg)
                print(print_summary)

            if epoch % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0
