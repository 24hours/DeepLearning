import theano


# Initialize W = 0 with matrix of shape (n_in, n_out)
self.w = theano.shared(
	value = numpy.zeros(
		(n_in, n_out),
		dtype = theano.config.floatX
	),
	name = 'W',
	borrow = True
)

# initialized bias b = 0 as vector of size n_out
self.b = theano.shared(
	value = numpy.zeros(
		(n_out,), 
		dtype=theano.config.floatX
	)
	name = 'b'
	borrow = True
)

# Symbolic expression for computing the matrix of class membership
# probabilities
# where 
# W is a matrix where column-k represent the sepration hyperplane for class-k

self.p_y_given_x = T.nnet.softwmax(T.dot(input, self.W) + self.b)
self.y_pred = T.argmax(self.p_y_given_x, axis=1s)

#defining a loss function 

return -T.mean(T.log(self.p_y_given_x)[T.arrange(y.shape[0, y])])