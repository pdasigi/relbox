import theano, numpy
from theano import tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

class RelBox(object):
    def __init__(self, num_words, num_rels, vocab_embed_size, lr=0.01, tensor_activation=T.tanh, num_noise_samples=1):
        numpy_rng = numpy.random.RandomState(89677)
        theano_rng = RandomStreams(12783)
        rng_box_limit = 4 * numpy.sqrt(6. / (vocab_embed_size + vocab_embed_size + num_rels))
        rng_box_low = 0
        rng_box_high = rng_box_limit
        init_box = numpy.asarray(numpy_rng.uniform(low=rng_box_low, high=rng_box_high, size=(vocab_embed_size, vocab_embed_size, num_rels)))
        rng_proj_limit = 4 * numpy.sqrt(6. / (num_words + vocab_embed_size))
        rng_proj_low = 0
        rng_proj_high = rng_proj_limit
        init_dense_vocab = numpy.asarray(numpy_rng.uniform(low=rng_proj_low, high=rng_proj_high, size=(num_words, vocab_embed_size)))
        init_rev_dense_vocab = numpy.asarray(numpy_rng.uniform(low=rng_proj_low, high=rng_proj_high, size=(vocab_embed_size, num_words)))
        self.B = theano.shared(value=init_box, name='B')
        self.P = theano.shared(value=init_dense_vocab, name='P')
        self.P_hat = theano.shared(value=init_rev_dense_vocab, name='P_hat')
        self.vocab = T.eye(num_words)
        word_activation = T.nnet.softmax
        self.rel = T.eye(num_rels)
        rel_activation = T.nnet.softmax

        self.lr = lr

        self.x_ind, self.y_ind, self.r_ind = T.iscalars('x_ind', 'y_ind', 'r_ind')
        x = self.vocab[self.x_ind]
        self.x_rep = T.dot(x, self.P)
        y = self.vocab[self.y_ind]
        self.y_rep = T.dot(y, self.P)
        r = self.rel[self.r_ind]
        # Assumption: Corresponding dimensions: 0 -> x, 1 -> y, 2 -> r
        # TODO: Where do we apply activations? Do we have to, at all?
        pred_xy = tensor_activation(T.tensordot(r, self.B, axes=(0,2)))
        pred_y = T.dot(T.tensordot(self.x_rep, pred_xy, axes=(0,0)), self.P_hat)
        self.prob_y = word_activation(pred_y)
        pred_x = T.dot(T.tensordot(self.y_rep, pred_xy, axes=(0,1)), self.P_hat)
        self.prob_x = word_activation(pred_x)
        pred_yr = tensor_activation(T.tensordot(self.x_rep, self.B, axes=(0,0)))
        self.prob_r = rel_activation(T.tensordot(self.y_rep, pred_yr, axes=(0,0)))

        self.score = T.dot(y, T.dot(T.tensordot(self.x_rep, T.tensordot(r, self.B, axes=(0,2)), axes=(0,0)), self.P_hat).T)
        # y \times (((x \times P) \times (r \otimes B)) \times P_hat)
        rand_margin_score = T.constant(0)
        noise_log_likelihood = T.constant(0)
        # The noise distribution is one where words and the relation are independent of each other.  The probability of the right tuple and the corrupted tuple are both equal in this distribution.
        noise_prob = num_noise_samples/float(num_words * num_words * num_rels)
        rand_x_ind = theano_rng.random_integers(low=0, high=num_words-1)
        rand_y_ind = theano_rng.random_integers(low=0, high=num_words-1)
        rand_r_ind = theano_rng.random_integers(low=0, high=num_rels-1)
        rand_x = self.vocab[rand_x_ind]
        rand_x_rep = T.dot(rand_x, self.P)
        rand_y = self.vocab[rand_y_ind]
        rand_y_rep = T.dot(rand_y, self.P)
        rand_r = self.rel[rand_r_ind]
        rand_score = T.dot(rand_y, T.dot(T.tensordot(rand_x_rep, T.tensordot(rand_r, self.B, axes=(0,2)), axes=(0,0)), self.P_hat).T)
        for _ in range(num_noise_samples):
            rand_margin_score += rand_score
            noise_log_likelihood += T.log(noise_prob/(T.abs_(rand_score) + noise_prob))
        self.nce_margin_loss = T.maximum(0, 1 - self.score + rand_margin_score)
        
        # NCE negative log likelihood:-1 * {log(score/(score + num_noise_samples*noise_prob)) + \sum_{i=1}^k (log(noise_prob/(rand_score + noise_prob)))}
        self.nce_prob_loss = -(T.log(T.abs_(self.score)/(T.abs_(self.score) + noise_prob)) + noise_log_likelihood)
        self.cost_inputs = [self.x_ind, self.y_ind, self.r_ind]
        self.params = [self.B, self.P, self.P_hat]

        self.x_loss = self.ce(x, self.prob_x)
        self.y_loss = self.ce(y, self.prob_y)
        self.r_loss = self.ce(r, self.prob_r)

    # TODO: Other (better?) losses
    def mse(self, t, p):
        return T.mean((p - t) ** 2)

    def ce(self, t, p):
        return T.mean(T.nnet.binary_crossentropy(p, t))

    def get_x_pred(self):
        return theano.function([self.y_ind, self.r_ind], self.prob_x)

    def get_y_pred(self):
        return theano.function([self.x_ind, self.r_ind], self.prob_y)

    def get_r_pred(self):
        return theano.function([self.x_ind, self.y_ind], self.prob_r)

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

    def train_with_nce_margin(self):
        gparams = T.grad(self.nce_margin_loss, self.params)
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - self.lr * gparam))
        return theano.function(self.cost_inputs, self.nce_margin_loss, updates=updates)

    def train_with_nce_prob(self):
        gparams = T.grad(self.nce_prob_loss, self.params)
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - self.lr * gparam))
        return theano.function(self.cost_inputs, self.nce_prob_loss, updates=updates)
