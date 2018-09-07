import utils as utls
import numpy
import matplotlib.pyplot as plt
import numpy as np

class LossLogger:

    def __init__(self,
                 phase,
                 batch_size,
                 n_samples,
                 save_path,
                 print_mode=1,
                 print_skip_ratio=0.75):

        self.batch_size = batch_size
        self.n_samples = n_samples
        self.save_path = save_path
        self.print_mode = print_mode
        self.phase = phase
        self.print_skip_ratio = print_skip_ratio
        self.print_loss_every = int(n_samples*(1-print_skip_ratio))

        # This will be a dict with epoch as keys
        # values are (batch, loss) tuples
        self.loss = dict()

    def print_loss_batch(self, epoch, batch, loss):
        print('[{}] batch {}/{}: running_loss: {:.4f}'.format(
            self.phase,
            batch,
            self.n_samples,
            loss))


    def update(self, epoch, batch, loss):

        if(epoch not in self.loss.keys()):
            self.loss[epoch] = list()
        self.loss[epoch].append((batch, loss))

        if(self.print_mode == 1):
            losses = np.asarray(self.loss[epoch])[:, -1]
            if(losses.size % self.print_loss_every == 0):
                self.print_loss_batch(epoch, batch, loss)

    def get_loss(self, epoch):
        return np.sum(np.asarray(self.loss[epoch][:-1]))

    def print_epoch(self, epoch):

        if(epoch in self.loss.keys()):
            print('[{}] batch_size: {}, epoch: {}, loss: {}'.format(
                self.phase,
                self.batch_size,
                epoch,
                self.get_loss(epoch)))
            
    #def save(self, fname):
        
            

