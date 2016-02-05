from __future__ import print_function

#!/usr/bin/env python
# Example script for recurrent network usage in PyBrain.
__author__ = "Marcos Teixeira"
__version__ = '$Id$'

from pylab import plot, hold, show
from scipy import sin, rand, arange
from pybrain.datasets            import SequenceClassificationDataSet
from pybrain.structure.modules   import LSTMLayer, SoftmaxLayer
from pybrain.supervised          import RPropMinusTrainer
from pybrain.tools.validation    import testOnSequenceData
from pybrain.tools.shortcuts     import buildNetwork  

# create training and test data
trndata = ClassificationDataSet(1,1, nb_classes=4)

trndata._convertToOneOfMany( bounds=[0.,1.] )
print (trndata)
exit(1)

# construct LSTM network - note the missing output bias
rnn = buildNetwork( trndata.indim, 5, trndata.outdim, hiddenclass=LSTMLayer, outclass=SoftmaxLayer, outputbias=False, recurrent=True)

# define a training method
trainer = RPropMinusTrainer( rnn, dataset=trndata, verbose=True )
# instead, you may also try
##trainer = BackpropTrainer( rnn, dataset=trndata, verbose=True, momentum=0.9, learningrate=0.00001 )

# carry out the training
for i in range(1):
    trainer.trainEpochs( 2 )
    trnresult = 100. * (1.0-testOnSequenceData(rnn, trndata))
    print("train error: %5.2f%%" % trnresult)

# just for reference, plot the first 5 timeseries
plot(trndata['input'][0:250,:],'-o')
hold(True)
plot(trndata['target'][0:250,0])
show()