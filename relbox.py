import theano
from theano import tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

class RelBox(object):
    def __init__(self, init_box, init_vocab=None, init_rel=None, lr=0.01, dense_vocab_rep=True, dense_rel_rep=False, tensor_activation=T.tanh, train_nce=False):
        self.B = theano.shared(value=init_box, name='B')
        word_dim, _, rel_dim = init_box.shape
        if dense_vocab_rep:
            self.vocab = theano.shared(value=init_vocab, name='V')
            word_activation = T.tanh
        else:
            self.vocab = T.eye(word_dim)
            word_activation = T.nnet.softmax

        if dense_rel_rep:
            self.rel = theano.shared(value=init_rel, name='R')
            rel_activation = T.tanh
        else:
            self.rel = T.eye(rel_dim)
            rel_activation = T.nnet.softmax

        self.lr = lr

        self.x_ind, self.y_ind, self.r_ind = T.iscalars('x_ind', 'y_ind', 'r_ind')
        x = self.vocab[self.x_ind]
        y = self.vocab[self.y_ind]
        r = self.rel[self.r_ind]
        # Assumption: Corresponding dimensions: 0 -> x, 1 -> y, 2 -> r
        # TODO: Where do we apply activations? Do we have to, at all?
        pred_xy = tensor_activation(T.tensordot(r, self.B, axes=(0,2)))
        self.pred_y = word_activation(T.tensordot(x, pred_xy, axes=(0,0)))
        self.pred_x = word_activation(T.tensordot(y, pred_xy, axes=(0,1)))
        pred_yr = tensor_activation(T.tensordot(x, self.B, axes=(0,0)))
        self.pred_r = rel_activation(T.tensordot(y, pred_yr, axes=(0,0)))

        if dense_vocab_rep:
            self.x_loss = self.mse(x, self.pred_x)
            self.y_loss = self.mse(y, self.pred_y)
        else:
            self.x_loss = self.ce(x, self.pred_x)
            self.y_loss = self.ce(y, self.pred_y)
        if dense_rel_rep:
            self.r_loss = self.mse(r, self.pred_r)
        else:
            self.r_loss = self.ce(r, self.pred_r)
        if train_nce:
            self.score = T.dot(y, T.tensordot(x, T.tensordot(r, self.B, axes=(0,2)), axes=(0,0)).T)
            srng = RandomStreams(seed=2345)
            rand_x_ind = srng.random_integers(low=0, high=word_dim-1)
            rand_y_ind = srng.random_integers(low=0, high=word_dim-1)
            rand_r_ind = srng.random_integers(low=0, high=rel_dim-1)
            rand_x = self.vocab[rand_x_ind]
            rand_y = self.vocab[rand_y_ind]
            rand_r = self.rel[rand_r_ind]
            rand_score = T.dot(rand_y, T.tensordot(rand_x, T.tensordot(rand_r, self.B, axes=(0,2)), axes=(0,0)).T)
            self.nce_loss = T.sum(T.maximum(0, 1 - self.score + rand_score))

        self.cost_inputs = [self.x_ind, self.y_ind, self.r_ind]
        self.params = [self.B]
        if dense_vocab_rep:
            self.params.append(self.vocab)

        if dense_rel_rep:
            self.params.append(self.rel)
        #self.params = [self.B, self.vocab, self.rel]
        #self.params = [self.B, self.vocab]

    # TODO: Other (better?) losses
    def mse(self, t, p):
        return T.mean((p - t) ** 2)

    def ce(self, t, p):
        return T.mean(T.nnet.binary_crossentropy(p, t))

    def get_x_pred(self):
        return theano.function([self.y_ind, self.r_ind], self.pred_x)

    def get_y_pred(self):
        return theano.function([self.x_ind, self.r_ind], self.pred_y)

    def get_r_pred(self):
        return theano.function([self.x_ind, self.y_ind], self.pred_r)

    def get_costs(self):
        return theano.function(self.cost_inputs, [self.x_loss, self.y_loss, self.r_loss])

    def update_for_x(self):
        gparams = T.grad(self.x_loss, self.params)
        updates_for_x = []
        for param, gparam in zip(self.params, gparams):
            updates_for_x.append((param, param - self.lr * gparam))
        return theano.function(self.cost_inputs, self.x_loss, updates=updates_for_x)

    def update_for_y(self):
        gparams = T.grad(self.y_loss, self.params)
        updates_for_y = []
        for param, gparam in zip(self.params, gparams):
            updates_for_y.append((param, param - self.lr * gparam))
        return theano.function(self.cost_inputs, self.y_loss, updates=updates_for_y)

    def update_for_r(self):
        gparams = T.grad(self.r_loss, self.params)
        updates_for_r = []
        for param, gparam in zip(self.params, gparams):
            updates_for_r.append((param, param - self.lr * gparam))
        return theano.function(self.cost_inputs, self.r_loss, updates=updates_for_r)

    def train_with_nce(self):
        gparams = T.grad(self.nce_loss, self.params)
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - self.lr * gparam))
        return theano.function(self.cost_inputs, self.nce_loss, updates=updates)



