import gzip
import cPickle
import theano
import numpy
import theano.tensor as T

def loaddata(filename):
	with gzip.open(filename) as f:
		train_set, valid_set, test_set = cPickle.load(f)

	def shared_dataset(data_xy, borrow=True):
			data_x, data_y = data_xy
			shared_x = theano.shared(numpy.asarray(data_x,
													dtype=theano.config.floatX),
									borrow=borrow)
			shared_y = theano.shared(numpy.asarray(data_y, 
													dtype=theano.config.floatX),
									borrow=borrow)

			return shared_x, T.cast(shared_y, 'int32')

	test_set_x, test_set_y = shared_dataset(test_set)
	valid_set_x, valid_set_y = shared_dataset(valid_set)
	train_set_x, train_set_y = shared_dataset(train_set)

	rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), 
			(test_set_x, test_set_y)]

	return rval