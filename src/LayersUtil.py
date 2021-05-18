import numpy as np
from scipy.ndimage import zoom

# This method is for convolution2d layer
class Convolution2D:
    # Initializing all parameters
    def __init__(self, inputs_channel, num_filters, kernel_size, padding, stride, learning_rate, name):
        self.numFilters = num_filters
        self.kernelSize = kernel_size
        self.inputChannels = inputs_channel
        self.padding = padding
        self.stride = stride
        self.learningRate = learning_rate
        self.layerName = name
        # defining weights shape and bias' shape
        self.weights = np.zeros((self.numFilters, self.inputChannels, self.kernelSize, self.kernelSize))
        self.bias = np.zeros((self.numFilters, 1))
        # initializing weights with random values 
        for i in range(0,self.numFilters):
            self.weights[i,:,:,:] = np.random.normal(loc=0, scale=np.sqrt(1./(self.inputChannels*self.kernelSize*self.kernelSize)), size=(self.inputChannels, self.kernelSize, self.kernelSize))


    # This function to add padding
    def zero_padding(self, inputs, size):
        width, height = inputs.shape[0], inputs.shape[1]
        # adding paddign to the input image
        padded_weight = 2 * size + width
        padded_height = 2 * size + height
        # innitializing padding part of the image with zeros
        out = np.zeros((padded_weight, padded_height))
        out[size:width+size, size:height+size] = inputs
        return out
 
    # This function if for forward propagation of conv2d layer
    def forward(self, inputs):
        # initializing input shape- channels, width, height
        inputChannels = inputs.shape[0]
        inputWidth = inputs.shape[1]+2*self.padding
        inputHeight = inputs.shape[2]+2*self.padding
        # initialize inputs with zeros
        self.inputs = np.zeros((inputChannels, inputWidth, inputHeight))
        # padding using the zero_padding method
        for c in range(inputs.shape[0]):
            self.inputs[c,:,:] = self.zero_padding(inputs[c,:,:], self.padding)
        # defining the new height and width
        WW = (inputWidth - self.kernelSize)//self.stride + 1
        HH = (inputHeight - self.kernelSize)//self.stride + 1
        # initialize feature maps with new shapes
        feature_maps = np.zeros((self.numFilters, WW, HH))
        # updating feature maps with convolution layer logic
        for filter in range(self.numFilters):
            for width in range(WW):
                for height in range(HH):
                    feature_maps[filter,width,height]=np.sum(self.inputs[:,width:width+self.kernelSize,height:height+self.kernelSize]*self.weights[filter,:,:,:])+self.bias[filter]

        return feature_maps
    #  This function is for backward propagation of Conv2d layer
    def backward(self, dy):
        # initializing input shape- channels, width, height
        inputChannels, inputWidth, inputHeight = self.inputs.shape
        # intialising gradients with zeros
        dx = np.zeros(self.inputs.shape)
        dw = np.zeros(self.weights.shape)
        db = np.zeros(self.bias.shape)
        # extracting shapes of gradients
        dFilters, dWidth, dHeight = dy.shape
        # Using backward propagation to calculate gradients
        for filter in range(dFilters):
            for width in range(dWidth):
                for height in range(dHeight):
                    dw[filter,:,:,:]+=dy[filter,width,height]*self.inputs[:,width:width+self.kernelSize,height:height+self.kernelSize]
                    dx[:,width:width+self.kernelSize,height:height+self.kernelSize]+=dy[filter,width,height]*self.weights[filter,:,:,:]

        for filter in range(dFilters):
            db[filter] = np.sum(dy[filter, :, :])
        # Updating weights and bias using the gradients and learning rate
        self.weights -= self.learningRate * dw
        self.bias -= self.learningRate * db
        #print("Check this final..",self.inputs.shape,dx.shape)
        return dx
    # This function is to extract parameters of Conv2d layers
    def extract(self):
        return {self.layerName+'.weights':self.weights, self.layerName+'.bias':self.bias}
    # This function is to save parameters of conv2d layers
    def feed(self, weights, bias):
        self.weights = weights
        self.bias = bias
# This class is for MaxPooling layer
class Maxpooling2D:
    #  This function is to initialize current object's parameters
    def __init__(self, pool_size, stride, name):
        self.paddingool = pool_size
        self.stride = stride
        self.layerName = name
    # This function is for forward propagation of Maxpooling2d
    def forward(self, inputs):
        # Initializing shape of input data
        self.inputs = inputs
        inputChannels, inputWidth, inputHeight = inputs.shape
        # calculating new width and height after max pooling
        new_width = (inputWidth - self.paddingool)//self.stride + 1
        new_height = (inputHeight - self.paddingool)//self.stride + 1
        # initializing output values with zeros
        out = np.zeros((inputChannels, new_width, new_height))
        # updating output values with maxpool logic 
        for channel in range(inputChannels):
            for width in range(inputWidth//self.stride):
                for height in range(inputHeight//self.stride):
                    out[channel, width, height] = np.max(self.inputs[channel, width*self.stride:width*self.stride+self.paddingool, height*self.stride:height*self.stride+self.paddingool])
        return out
    # This function is for backward propagation of Maxpooling2d
    def backward(self, dy):
        # initializing shape of input values
        inputChannels, inputWidth, inputHeight = self.inputs.shape
        # initializing gradient values with zeros
        dx = np.zeros(self.inputs.shape)
        # Using maxpool logic we update gradients
        for channel in range(inputChannels):
            for width in range(0, inputWidth, self.paddingool):
                for height in range(0, inputHeight, self.paddingool):
                    st = np.argmax(self.inputs[channel,width:width+self.paddingool,height:height+self.paddingool])
                    (idx, idy) = np.unravel_index(st, (self.paddingool, self.paddingool))
                    dx[channel, width+idx, height+idy] = dy[channel, width//self.paddingool, height//self.paddingool]
        return dx

    def extract(self):
        return 
# This class is for dense layer
class FullyConnected:
    # This function is to initialize parameters
    def __init__(self, num_inputs, num_outputs, learning_rate, name):
        self.weights = 0.01*np.random.rand(num_inputs, num_outputs)
        self.bias = np.zeros((num_outputs, 1))
        self.learningRate = learning_rate
        self.layerName = name
    # Thisfunction is for forward propagation of FC layer
    def forward(self, inputs):
        self.inputs = inputs
        # Calculating output of FC layer
        return np.dot(self.inputs, self.weights) + self.bias.T
    
    # Thisfunction is for backward propagation of FC layer
    def backward(self, dy):
        # Initializing gradients
        if dy.shape[0] == self.inputs.shape[0]:
            dy = dy.T
        # calculating gradients of FC layer
        dw = dy.dot(self.inputs)
        db = np.sum(dy, axis=1, keepdims=True)
        dx = np.dot(dy.T, self.weights.T)
        # Updating weights and biases using gradients and learning rate
        self.weights -= self.learningRate * dw.T
        self.bias -= self.learningRate * db

        return dx

    # This function is to extract parameters of FC layer
    def extract(self):
        return {self.layerName+'.weights':self.weights, self.layerName+'.bias':self.bias}

    # This function is to save parameters of pool layers
    def feed(self, weights, bias):
        self.weights = weights
        self.bias = bias
# This class is to flatten before a Fully Connected layer
class Flatten:
    def __init__(self):
        pass
    # This method is for forward propagation of Flatten layer
    def forward(self, inputs):
        # modifying the shape of input data
        self.inputChannels, self.inputWidth, self.inputHeight = inputs.shape
        return inputs.reshape(1, self.inputChannels*self.inputWidth*self.inputHeight)
    # This method is for backward propagation of Flatten layer
    def backward(self, dy):
        # modifying the shape back to non-linear
        return dy.reshape(self.inputChannels, self.inputWidth, self.inputHeight)
    def extract(self):
        return
# This class is for ReLu Activation function
class ReLu:
    def __init__(self):
        pass
    # This method is forward propagation of ReLu activation function
    def forward(self, inputs):
        # according to relu logic we update inputs
        self.inputs = inputs
        ret = inputs.copy()
        ret[ret < 0] = 0
        return ret
    # This method is for backward propagation of ReLu activation function
    def backward(self, dy):
        dx = dy.copy()
        if(self.inputs.shape != dy.shape):
            ratio = self.inputs.shape[2]/dy.shape[2]
            dx = zoom(dx, (1, ratio, ratio))
        dx[self.inputs < 0] = 0
        return dx
    def extract(self):
        return
# This class is for softmax activation funtion which is used in output layer
class Softmax:
    def __init__(self):
        pass
    # This class is for forward propagation of softmax activation function
    def forward(self, inputs):
        # obtaining maximum value
        maxValue = np.max(inputs[0])
        #  calculating differences
        diff = np.exp(inputs[0]-maxValue)
        # calculating exponents
        exp = np.exp(diff.reshape(1,10,), dtype=np.float)
        self.out = exp/np.sum(exp)
        return self.out
    # thus method is for backward propagation 
    def backward(self, dy):
        # calculating gradients and updating values
        return self.out.T - dy.reshape(dy.shape[0],1)
    def extract(self):
        return
# Dropout is used to minimize overfitting. Dropout regularize the neural network by decreasing the coadaption between the neurons.
class Dropout():
    #Initializeing probability for dropout which is 0.5
    def __init__(self,probability=0.5):
        self.probability = probability
        self.parameters = []
        #In forward, each neuron will have a probability of 0.5 of being turned off.
    def forward(self,Input):
        self.mask = np.random.binomial(1,self.probability,size=Input.shape) / self.probability
        out = Input * self.mask
        return out.reshape(Input.shape)
    # We will backpropagate the neurons that were not turned off during the forward pass
    # as changing the output of the turned off neurons doesn’t change the actual output, 
    def backward(self,dY):
        dX = dY * self.mask
        return dX
    
    def extract(self):
        return
    