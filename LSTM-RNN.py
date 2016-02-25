'''
Build a tweet sentiment analyzer
'''
from collections import OrderedDict
import cPickle as pkl
import sys
import time

import numpy
import theano
from theano import config
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import imdb
import common

# Set the random number generators' seeds for consistency
SEED = 123
numpy.random.seed(SEED)

def adadelta(lr, tparams, grads, x, mask, y, cost):
    """
    An adaptive learning rate optimizer

    """

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, mask, y], cost, updates=zgup + rg2up,
                                    name='adadelta_f_grad_shared')

    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]

    f_update = theano.function([lr], [], updates=ru2up + param_up,
                               on_unused_input='ignore',
                               name='adadelta_f_update')

    return f_grad_shared, f_update

class Lstm(object):
    """ docstring for Lstm """
    def __init__(self, dim_proj=128,  # word embeding dimension and LSTM number of hidden units.
        max_epochs=5000,  # The maximum number of epoch to run
        dispFreq=2,  # Display to stdout the training progress every N updates
        lrate=0.0001,  # Learning rate for sgd (not used for adadelta and rmsprop)
        n_words=10000,  # Vocabulary size
        optimizer=adadelta,
        encoder='lstm',  # TODO: can be removed must be lstm.
        saveFreq=100,  # Save the parameters after every saveFreq updates
        maxlen=100,  # Sequence longer then this get ignored
        batch_size=16,  # The batch size during training.
        dataset='imdb',

        # Parameter for extra option
        noise_std=0.,
        use_dropout=True,):

        self.model_options = locals().copy()
        
        self.loadData()
        self.createModel()

    def loadData(self):
        print 'Loading data'
        self.train = common.generate_ucf_dataset('frames')
        self.prepare_data = imdb.prepare_data

        ydim = numpy.max(self.train[1]) + 1

        self.model_options['ydim'] = ydim

    
        
    def createModel(self):
        # This create the initial parameters as numpy ndarrays.
        # Dict name (string) -> numpy ndarray

        print 'Building the Model'
        params = init_params(self.model_options)

        # This create Theano Shared Variable from the parameters.
        # Dict name (string) -> Theano Tensor Shared Variable
        # params and tparams have different copy of the weights.
        tparams = init_tparams(params)

        # use_noise is for dropout
        (self.use_noise, self.x, self.mask,
        self.y, self.f_pred_prob, self.f_pred, self.cost) = self.buildModel(tparams, self.model_options)

        f_cost = theano.function([self.x, self.mask, self.y], self.cost, name='f_cost')

        grads = tensor.grad(self.cost, wrt=tparams.values())
        self.f_grad = theano.function([self.x, self.mask, self.y], grads, name='f_grad')

        self.lr = tensor.scalar(name='lr')
        self.f_grad_shared, self.f_update = self.model_options['optimizer'](self.lr, tparams, grads,
                                            self.x, self.mask, self.y, self.cost)
    
    def buildModel(self,tparams, options):
        trng = RandomStreams(SEED)

        # Used for dropout.
        use_noise = theano.shared(numpy_floatX(0.))

        x = tensor.matrix('x', dtype='int64')
        mask = tensor.matrix('mask', dtype=config.floatX)
        y = tensor.vector('y', dtype='int64')

        n_timesteps = x.shape[0]
        n_samples = x.shape[1]

        emb = tparams['Wemb'][x.flatten()].reshape([n_timesteps,
                                                    n_samples,
                                                    options['dim_proj']])
        proj = lstm_layer(tparams, emb, options,prefix=options['encoder'],mask=mask)

        if options['encoder'] == 'lstm':
            proj = (proj * mask[:, :, None]).sum(axis=0)
            proj = proj / mask.sum(axis=0)[:, None]

        pred = tensor.nnet.softmax(tensor.dot(proj, tparams['U']) + tparams['b'])

        f_pred_prob = theano.function([x, mask], pred, name='f_pred_prob')
        f_pred = theano.function([x, mask], pred.argmax(axis=1), name='f_pred')

        off = 1e-8
        if pred.dtype == 'float16':
            off = 1e-6

        cost = -tensor.log(pred[tensor.arange(n_samples), y] + off).mean()

        return use_noise, x, mask, y, f_pred_prob, f_pred, cost
        
    def train_lstm(self):
        best_p = None

        print 'Start Training'

        uidx = 0  # the number of update done
        self.start_time = time.time()
        try:
            for eidx in xrange(self.model_options['max_epochs']):
                n_samples = 0

                # Get new shuffled index for the training set.
                kf = get_minibatches_idx(len(self.train[0]), self.model_options['batch_size'], shuffle=True)

                for _, train_index in kf:
                    uidx += 1
                    self.use_noise.set_value(1.)

                    # Select the random examples for this minibatch
                    y = [self.train[1][t] for t in train_index]
                    x = [self.train[0][t]for t in train_index]

                    # Get the data in numpy.ndarray format
                    # This swap the axis!
                    # Return something of shape (minibatch maxlen, n samples)
                    x, mask, y = self.prepare_data(x, y)
                    n_samples += x.shape[1]

                    cost = self.f_grad_shared(x, mask, y)
                    self.f_update(self.model_options['lrate'])

                    if numpy.mod(uidx, self.model_options['dispFreq']) == 0:
                        print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost

                    '''if numpy.mod(uidx, validFreq) == 0:
                        use_noise.set_value(0.)
                        train_err = pred_error(f_pred, prepare_data, train, kf)

                        print ('Train ', train_err)'''

                print 'Seen %d samples' % n_samples

        except KeyboardInterrupt:
            print "Training interupted"

        self.end_time = time.time()

def numpy_floatX(data):
    return numpy.asarray(data, dtype=config.floatX)


def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    """

    idx_list = numpy.arange(n, dtype="int32")

    if shuffle:
        numpy.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)

def zipp(params, tparams):
    """
    When we reload the model. Needed for the GPU stuff.
    """
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)


def unzip(zipped):
    """
    When we pickle the model. Needed for the GPU stuff.
    """
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params

def _p(pp, name):
    return '%s_%s' % (pp, name)


def init_params(options):
    """

    Global (not LSTM) parameter. For the embedding and the classifier.

    """
    params = OrderedDict()
    # embedding
    randn = numpy.random.rand(options['n_words'],
                              options['dim_proj'])
    params['Wemb'] = (0.01 * randn).astype(config.floatX)
    params = param_init_lstm(options,params, prefix=options['encoder'])
                                             
    # classifier
    params['U'] = 0.01 * numpy.random.randn(options['dim_proj'],
                                            options['ydim']).astype(config.floatX)
    params['b'] = numpy.zeros((options['ydim'],)).astype(config.floatX)

    return params


def load_params(path, params):
    pp = numpy.load(path)
    for kk, vv in params.iteritems():
        if kk not in pp:
            raise Warning('%s is not in the archive' % kk)
        params[kk] = pp[kk]

    return params


def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams


def get_layer(name):
    fns = layers[name]
    return fns


def ortho_weight(ndim):
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype(config.floatX)


def param_init_lstm(options, params, prefix='lstm'):
    """
    Init the LSTM parameter:

    :see: init_params
    """
    W = numpy.concatenate([ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj'])], axis=1)
    params[_p(prefix, 'W')] = W
    U = numpy.concatenate([ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj'])], axis=1)
    params[_p(prefix, 'U')] = U
    b = numpy.zeros((4 * options['dim_proj'],))
    params[_p(prefix, 'b')] = b.astype(config.floatX)

    return params


def lstm_layer(tparams, state_below, options, prefix='lstm', mask=None):
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    assert mask is not None

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    def _step(m_, x_, h_, c_):
        preact = tensor.dot(h_, tparams[_p(prefix, 'U')])
        preact += x_

        i = tensor.nnet.sigmoid(_slice(preact, 0, options['dim_proj']))
        f = tensor.nnet.sigmoid(_slice(preact, 1, options['dim_proj']))
        o = tensor.nnet.sigmoid(_slice(preact, 2, options['dim_proj']))
        c = tensor.tanh(_slice(preact, 3, options['dim_proj']))

        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * tensor.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c

    state_below = (tensor.dot(state_below, tparams[_p(prefix, 'W')]) +
                   tparams[_p(prefix, 'b')])

    dim_proj = options['dim_proj']
    rval, updates = theano.scan(_step,
                                sequences=[mask, state_below],
                                outputs_info=[tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_proj),
                                              tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_proj)],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps)
    return rval[0]


# ff: Feed Forward (normal neural net), only useful to put after lstm
#     before the classifier.
layers = {'lstm': (param_init_lstm, lstm_layer)}


def pred_probs(f_pred_prob, prepare_data, data, iterator, verbose=False):
    """ If you want to use a trained model, this is useful to compute
    the probabilities of new examples.
    """
    n_samples = len(data[0])
    probs = numpy.zeros((n_samples, 2)).astype(config.floatX)

    n_done = 0

    for _, valid_index in iterator:
        x, mask, y = prepare_data([data[0][t] for t in valid_index],
                                  numpy.array(data[1])[valid_index],
                                  maxlen=None)
        pred_probs = f_pred_prob(x, mask)
        probs[valid_index, :] = pred_probs

        n_done += len(valid_index)
        if verbose:
            print '%d/%d samples classified' % (n_done, n_samples)

    return probs


def pred_error(f_pred, prepare_data, data, iterator, verbose=False):
    """
    Just compute the error
    f_pred: Theano fct computing the prediction
    prepare_data: usual prepare_data for that dataset.
    """
    valid_err = 0
    for _, valid_index in iterator:
        x, mask, y = prepare_data([data[0][t] for t in valid_index],
                                  numpy.array(data[1])[valid_index],
                                  maxlen=None)
        preds = f_pred(x, mask)
        targets = numpy.array(data[1])[valid_index]
        valid_err += (preds == targets).sum()
    valid_err = 1. - numpy_floatX(valid_err) / len(data[0])

    return valid_err

if __name__ == '__main__':
    
    net = Lstm()
    net.train_lstm()
    
