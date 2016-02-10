"""

//[]------------------------------------------------------------------------[]
//|                                                                          |
//|                      Long-Short Term Memory Module                       |
//|                               Version 1.0                                |
//|                                                                          |
//|              Copyright 2016-2020, Marcos Vinicius Teixeira               |
//|                          All Rights Reserved.                            |
//|                                                                          |
//[]------------------------------------------------------------------------[]
//
OVERVIEW: lstm.py
//  ================================
//  This module implement a Simple Long-Short Term Memory(LSTM), a Recurrent 
//  Neural Network to sequence data and it's an adaptation of the lstm code 
//  provided by pybrain for sequence classification.
//
"""
from __future__ import print_function


from pylab import plot, hold, show
from scipy import sin, rand, arange
from pybrain.datasets            import SequenceClassificationDataSet
from pybrain.structure.modules   import LSTMLayer, SoftmaxLayer
from pybrain.supervised          import BackpropTrainer
from pybrain.tools.validation    import testOnSequenceData
from pybrain.tools.shortcuts     import buildNetwork
import sys
import common


# Dimension of each position of the vector X
X_DIM = 1322

# Number of classes
N_CLASSES = 487

if __name__ == '__main__':

	if len(sys.argv) < 2:
		print ("Usage: python simple_lstm.py trainset_file")
		raise "Missing params"

	f = open(sys.argv[1])
	X, Y = common.generate_dataset(f)

	# Constructing the train data
	trndata = SequenceClassificationDataSet(X_DIM,1,nb_classes=N_CLASSES)
	for i in range(len(X)):
		trndata.newSequence()

		trndata.addSample(X[i],Y[i])

	trndata._convertToOneOfMany( bounds=[0.,1.] )

	# construct LSTM network
	rnn = buildNetwork( trndata.indim, 5, trndata.outdim, hiddenclass=LSTMLayer, outclass=SoftmaxLayer, outputbias=False, recurrent=True)

	# define a training method
	trainer = BackpropTrainer( rnn, dataset=trndata, verbose=True, momentum=0.9, learningrate=0.00001 )

	# Training the network
	for i in range(100):
	    trainer.trainEpochs(5)
	    trnresult = 100. * (1.0-testOnSequenceData(rnn, trndata))
	    print("train error: %5.2f%%" % trnresult)

	# optional plot
	plot(trndata['input'],'-o')
	hold(True)
	plot(trndata['target'])
	show()
