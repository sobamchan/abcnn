import numpy as np
import chainer.functions as F
from chainer import Variable
from chainer import initializers
import chainer.links as L
import chainer

from sobamchan.sobamchan_chainer import Model

class AttentionLayer(chainer.Link):

    def __init__(self, w_shape=(300, 30)):
        super(AttentionLayer, self).__init__(
            W=w_shape,
        )
        self.W.data[...] = np.random.randn(w_shape[0], w_shape[1])

    def __call__(self, x):
        '''
        x.shape -> (b, c, h, w)
        '''
        xb, xc, xh, xw = x.shape
        wh, ww = self.W.shape
        return F.broadcast_to(self.W, (xb, xc, wh, ww))*x


class AttentionMatch1(Model):

    def __init__(self, w_shape=(300, 30), gpu=-1):
        super(AttentionMatch1, self).__init__(
            w1=AttentionLayer(w_shape),
            w2=AttentionLayer(w_shape),
        )

    def __call__(self, x1, x2, train=True):
        return self.fwd(x1, x2, train)

    def fwd(self, x1, x2, train=True):
        attention_matrix = self.match_score(x1, x2)
        attention_feature_map1 = F.swapaxes(self.w1(attention_matrix), 2, 3)
        attention_feature_map2 = F.swapaxes(self.w2(attention_matrix), 2, 3)
        x1 = F.concat((x1, attention_feature_map1), 1)
        x2 = F.concat((x2, attention_feature_map2), 1)
        return x1, x2

    def match_score(self, x1, x2):
        '''
        x1.shape == x2.shape == (b, c, h, w)
        '''
        # diff_matrix = F.squared_difference(x1, x2)
        diff_matrix = x1 * x2
        diff_matrix = F.swapaxes(diff_matrix, 2, 3)
        return diff_matrix


class ABCNN1(Model):

    def __init__(self, class_n, vocab_n, d, vocab, out_channels=5):
        super(ABCNN1, self).__init__(
            embed=L.EmbedID(vocab_n, d),
            am1=AttentionMatch1(),
            conv11=L.Convolution2D(2, out_channels, (1, 2), pad=(0, 2)),
            conv12=L.Convolution2D(2, out_channels, (1, 2), pad=(0, 2)),
            am2=AttentionMatch1(),
            conv21=L.Convolution2D(out_channels, out_channels, (1, 2), pad=(0, 2)),
            conv22=L.Convolution2D(out_channels, out_channels, (1, 2), pad=(0, 2)),
            fc=L.Linear(None, class_n),
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

        y1, y2 = self.am1(q1, q2)
        y1 = F.average_pooling_2d(F.tanh(self.conv11(y1)), (1, 3))
        y2 = F.average_pooling_2d(F.tanh(self.conv12(y2)), (1, 3))
        # y1, y2 = self.am2(y1, y2)
        # _, _, h, w = y1.shape
        # y1 = F.average_pooling_2d(F.tanh(self.wide_cnn21(y1)), (1, w))
        # y2 = F.average_pooling_2d(F.tanh(self.wide_cnn22(y2)), (1, w))
        y = F.concat([y1, y2])
        y = self.fc(y)
        return y
