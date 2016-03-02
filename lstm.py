from __future__ import print_function

#!/usr/bin/env python
# Example script for recurrent network usage in PyBrain.
__author__ = "Martin Felder"
__version__ = '$Id$'

import common
import numpy as np
from pylab import plot, hold, show
from scipy import sin, rand, arange
from sklearn.metrics import confusion_matrix
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.tools.validation    import testOnSequenceData
from sklearn.neighbors  		 import KNeighborsClassifier
from pylab import figure, ioff, clf, contourf, ion, draw, show
from pybrain.structure.modules   import LSTMLayer, SoftmaxLayer
from pybrain.datasets            import SequenceClassificationDataSet
from pybrain.supervised          import RPropMinusTrainer,BackpropTrainer
from sklearn.cross_validation 	 import train_test_split,StratifiedKFold

from sklearn.metrics import precision_score,recall_score,accuracy_score,f1_score,roc_auc_score


# create training and test data
DS = common.generate_ucf_dataset('frames')
X, y = DS

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)


trndata = SequenceClassificationDataSet(100,1, nb_classes=2)
tstdata = SequenceClassificationDataSet(100,1, nb_classes=2)

for index in range(len(y_train)):
	trndata.addSample(X_train[index], y_train[index])

for index in range(len(y_test)):
	tstdata.addSample(X_test[index], y_test[index])

trndata._convertToOneOfMany( bounds=[0.,1.] )
tstdata._convertToOneOfMany( bounds=[0.,1.] )
# construct LSTM network - note the missing output bias
rnn = buildNetwork( trndata.indim, 5, trndata.outdim, hiddenclass=LSTMLayer, outclass=SoftmaxLayer, outputbias=False, recurrent=True)

# define a training method
#trainer = RPropMinusTrainer( rnn, dataset=trndata, verbose=True )
# instead, you may also try
trainer = BackpropTrainer( rnn, dataset=trndata, momentum=0.1, verbose=True, weightdecay=0.01)

# carry out the training
for i in range(100):
    trainer.trainEpochs( 2 )
    trnresult = (1.0-testOnSequenceData(rnn, trndata))
    tstresult = (1.0-testOnSequenceData(rnn, tstdata))
    print("train error: %5.2f%%" % trnresult, ",  test error: %5.2f%%" % tstresult)

    out = rnn.activate(X_train[0])
    out = out.argmax(axis=0)

index=0
result = []
for x in X_test:
	result.append(rnn.activate(x).argmax())


mresult = confusion_matrix(y_test,result)
    
print (mresult)
print ("precision %4.2f"%(precision_score(y_test,result)))
print ("recall    %4.2f"%(recall_score(y_test,result)))
print ("f1        %4.2f"%(f1_score(y_test,result)))
print ("accuracy  %4.2fo"%(accuracy_score(y_test,result)))


