from __future__ import print_function

#!/usr/bin/env python
# Example script for recurrent network usage in PyBrain.
__author__ = "Martin Felder"
__version__ = '$Id$'

from pylab import plot, hold, show
from scipy import sin, rand, arange
from pybrain.datasets            import ClassificationDataSet
from pybrain.structure.modules   import LSTMLayer, SoftmaxLayer
from pybrain.supervised          import BackpropTrainer
from pybrain.tools.validation    import testOnSequenceData
from pybrain.tools.shortcuts     import buildNetwork

import common

# create training and test data
DS = common.generate_ucf_dataset('frames')
X, y = DS

trndata = ClassificationDataSet(100, class_labels=['piano', 'guitar'])

for index in range(len(y)):
	trndata.appendLinked(X[index], y[index])

trndata._convertToOneOfMany( bounds=[0.,1.] )
# construct LSTM network - note the missing output bias
rnn = buildNetwork( trndata.indim, 5, trndata.outdim, hiddenclass=LSTMLayer, outclass=SoftmaxLayer, outputbias=False, recurrent=True)

# define a training method
#trainer = RPropMinusTrainer( rnn, dataset=trndata, verbose=True )
# instead, you may also try
trainer = BackpropTrainer( rnn, dataset=trndata, verbose=True, momentum=0.9, learningrate=0.00001 )

# carry out the training
for i in range(10):
    err = trainer.train()
    
    print("train error: %5.2f%%" % err)

print(rnn.activate(X[-1]))

