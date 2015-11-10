import theano
from theano import tensor as T
from theano.tensor.nnet import conv
import numpy

rng = numpy.random.RandomState(1234)

input = T.tensor4(name='input')

w_shape = (2, 3, 9, 9)
w_bound = numpy.sqrt(3 * 9 * 9)
W = theano.shared( numpy.asarray(
			rng.uniform(
				low = -1.0 / w_bound, 
				high = 1.0 / w_bound,
				size = w_shp),
			dtype=input.dtyp), name = 'W')

b_shp = (2,)
b = theano.shared(numpy.asarray(
		rng.uniform(low= -0.5, high=0.5, size=b_shp),
		dtype=input.dtype), name = 'b')

# define that convolution output is calculated by conv ops 
conv_out  = conv.conv2d(input, W)

# apply bias and sigmoid activation
output = T.nnet.sigmoid(conv_out + b.dimshuffle('x', 0, 'x', 'x'))

f = theano.function([input], output)