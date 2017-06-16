import os
from dataset import Quora
import numpy as np
import argparse

from chainer import optimizers
import chainer.links as L
import chainer.functions as F
import chainer 
from tqdm import tqdm

from sobamchan.sobamchan_vocabulary import Vocabulary
from sobamchan.sobamchan_utility import Utility
from sobamchan.sobamchan_chainer import Model
from sobamchan.sobamchan_log import Log
from sobamchan.sobamchan_iterator import Iterator
util = Utility()

class TestModel(Model):
    def __init__(self, class_n, vocab_n, d, vocab):
        super(TestModel, self).__init__(
            embed=L.EmbedID(vocab_n, d),
            fc1=L.Linear(None, 10),
            fc2=L.Linear(None, 10),
            fc3=L.Linear(None, class_n),
        )
    def __call__(self, x, t, train=True):
        x = self.fwd(x, train)
        return F.softmax_cross_entropy(x, t), F.accuracy(x, t)
    def fwd(self, x, train):
        q1 = x[:, 0, :]
        q2 = x[:, 1, :]
        embed1 = self.embed(q1)
        embed2 = self.embed(q2)
        h = F.concat([embed1, embed2])
        h = F.tanh(self.fc1(h))
        h = F.tanh(self.fc2(h))
        h = self.fc3(h)
        return h


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', dest='bs', default=128, type=int)
    parser.add_argument('--epoch', dest='epoch', default=1, type=int)
    parser.add_argument('--gpu', dest='gpu', default=-1, type=int)
    parser.add_argument('--opath', dest='opath', required=True)

    return parser.parse_args()

def train(args):

    agps = get_args()
    
    bs = agps.bs
    epoch = agps.epoch
    gpu = agps.gpu
    opath = agps.opath
    
    
    train_x, train_t, test_x, test_t = Quora().get()
    train_n = len(train_x)
    test_n = len(test_x)
    
    vocab = Vocabulary()
 
    for d in train_x + test_x:
        try:
            vocab.new(d[0])
            vocab.new(d[1])
        except:
            print(d)
            break

    train_x = np.array([tuple([util.pad_to_max(vocab.encode(x[0].lower()), 30, 0), util.pad_to_max(vocab.encode(x[1].lower()), 30, 0)]) for x in train_x])
    train_t = np.array(train_t)
    test_x  = np.array([tuple([util.pad_to_max(vocab.encode(x[0].lower()), 30, 0), util.pad_to_max(vocab.encode(x[1].lower()), 30, 0)]) for x in test_x])
    test_t = np.array(test_t)
    
    optimizer = args['optimizer']
    class_n = 2
    vocab_n = len(vocab)
    d = 300
    model = args['model'](class_n, vocab_n, d, vocab)
    xp = model.check_gpu(gpu)
    optimizer.setup(model)
    
    train_loss_log = Log()
    test_loss_log = Log()
    test_acc_log = Log()
    
    for _ in tqdm(range(epoch)):
        
        order = np.random.permutation(train_n)
        train_x_iter = Iterator(train_x, bs, order, shuffle=False)
        train_t_iter = Iterator(train_t, bs, order, shuffle=False)
        
        loss_sum = 0
        for x, t in zip(train_x_iter, train_t_iter):
            model.cleargrads()
            x_n = len(x)
            x = model.prepare_input(x, dtype=xp.int32, xp=xp)
            t = model.prepare_input(t, dtype=np.int32, xp=xp)
            loss, _ = model(x, t, train=True)
            loss_sum += loss.data * x_n
            
            loss.backward()
            optimizer.update()
        loss_mean = float(loss_sum/train_n)
        train_loss_log.add(loss_mean)
        print('train_loss: {}'.format(loss_mean))
        
        order = np.random.permutation(test_n)
        test_x_iter = Iterator(test_x, bs, order, shuffle=False)
        test_t_iter = Iterator(test_t, bs, order, shuffle=False)
        loss_sum = 0
        acc_sum = 0
        for x, t in zip(test_x_iter, test_t_iter):
            model.cleargrads()
            x_n = len(x)
            x = model.prepare_input(x, dtype=xp.int32, xp=xp)
            t = model.prepare_input(t, dtype=xp.int32, xp=xp)
            loss, acc = model(x, t, train=False)
            loss_sum += loss.data * x_n
            acc_sum += acc.data * x_n
        loss_mean = float(loss_sum / test_n)
        acc_mean = float(acc_sum / test_n)
        test_loss_log.add(loss_mean)
        test_acc_log.add(acc_mean)
        print('test loss: {}'.format(loss_mean))
        print('test acc: {}'.format(acc_mean))

        
    opath = './results/{}'.format(opath)
    if not os.path.exists(opath):
        os.mkdir(opath)
        

    train_loss_log.save('{}/train_loss_log'.format(opath))
    train_loss_log.save_graph('{}/train_loss_log'.format(opath))
    test_loss_log.save('{}/test_loss_log'.format(opath))
    test_loss_log.save_graph('{}/test_loss_log'.format(opath))
    test_acc_log.save('{}/test_acc_log'.format(opath))
    test_acc_log.save_graph('{}/test_acc_log'.format(opath))



    
    
    
args = {}
args['optimizer'] = optimizers.AdaGrad()
args['model'] = TestModel
train(args)
