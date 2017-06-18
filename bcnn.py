import numpy as np
import chainer.functions as F
import chainer.links as L
import chainer

from sobamchan.sobamchan_chainer import Model

class BCNN(Model):

    def __init__(self, class_n, vocab_n, d, vocab):
        super(BCNN, self).__init__(
            embed=L.EmbedID(vocab_n, d),
            wide_cnn11=L.Convolution2D(1, 50, (1, 2), pad=(0, 2)),
            wide_cnn12=L.Convolution2D(1, 50, (1, 2), pad=(0, 2)),
            wide_cnn21=L.Convolution2D(50, 50, (1, 2), pad=(0, 2)),
            wide_cnn22=L.Convolution2D(50, 50, (1, 2), pad=(0, 2)),
            fc=L.Linear(None, class_n)
        )

    def __call__(self, x, t, train=True):
        y = self.fwd(x, train)
        return F.softmax_cross_entropy(y, t), F.accuracy(y, t)

    def fwd(self, x, train):
        q1 = self.embed(x[:, 0, :])
        q2 = self.embed(x[:, 1, :])
        b, embed_h, embed_w = q1.shape
        q1 = F.reshape(q1, (b, 1, embed_h, embed_w))
        q2 = F.reshape(q2, (b, 1, embed_h, embed_w))
        y1 = F.average_pooling_2d(F.tanh(self.wide_cnn11(q1)), (1, 3))
        y2 = F.average_pooling_2d(F.tanh(self.wide_cnn12(q2)), (1, 3))
        _, _, h, w = y1.shape
        y1 = F.average_pooling_2d(F.tanh(self.wide_cnn21(y1)), (1, w))
        y2 = F.average_pooling_2d(F.tanh(self.wide_cnn22(y2)), (1, w))
        y = F.concat([y1, y2])
        y = self.fc(y)
        return y
