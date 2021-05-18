import numpy as np
import pickle
from LayersUtil import *

class VGG16Model:

    # This method initializes all the layers for the VGG16 network.
    def initializeVGG16Network(self):
        # defining learningRate, we are fine tuning the learning rate on multiple runs.
        lr = 0.5

        # initialize the layers below.
        self.layers = []

        # layer 1
        self.layers.append(
            Convolution2D(inputs_channel=1, num_filters=64, kernel_size=3, padding=1, stride=1, learning_rate=lr,
                          name='conv1'))
        self.layers.append(ReLu())

        # layer 2
        self.layers.append(
            Convolution2D(inputs_channel=64, num_filters=64, kernel_size=3, padding=1, stride=1, learning_rate=lr,
                          name='conv2'))
        self.layers.append(ReLu())
        self.layers.append(Maxpooling2D(pool_size=2, stride=2, name='maxpool1'))

        # layer 3
        self.layers.append(
            Convolution2D(inputs_channel=64, num_filters=128, kernel_size=3, padding=1, stride=1, learning_rate=lr,
                          name='conv3'))
        self.layers.append(ReLu())

        # layer 4
        self.layers.append(
            Convolution2D(inputs_channel=128, num_filters=128, kernel_size=3, padding=1, stride=1, learning_rate=lr,
                          name='conv4'))
        self.layers.append(ReLu())
        self.layers.append(Maxpooling2D(pool_size=2, stride=2, name='maxpool2'))

        # layer 5
        self.layers.append(
            Convolution2D(inputs_channel=128, num_filters=256, kernel_size=3, padding=1, stride=1, learning_rate=lr,
                          name='conv5'))
        self.layers.append(ReLu())

        # layer 6
        self.layers.append(
            Convolution2D(inputs_channel=256, num_filters=256, kernel_size=3, padding=1, stride=1, learning_rate=lr,
                          name='conv6'))
        self.layers.append(ReLu())

        # layer 7
        self.layers.append(
            Convolution2D(inputs_channel=256, num_filters=256, kernel_size=3, padding=1, stride=1, learning_rate=lr,
                          name='conv7'))
        self.layers.append(ReLu())
        self.layers.append(Maxpooling2D(pool_size=2, stride=2, name='maxpool3'))

        # layer 8
        self.layers.append(
            Convolution2D(inputs_channel=256, num_filters=512, kernel_size=3, padding=1, stride=1, learning_rate=lr,
                          name='conv8'))
        self.layers.append(ReLu())

        # layer 9
        self.layers.append(
            Convolution2D(inputs_channel=512, num_filters=512, kernel_size=3, padding=1, stride=1, learning_rate=lr,
                          name='conv9'))
        self.layers.append(ReLu())

        # layer 10
        self.layers.append(
            Convolution2D(inputs_channel=512, num_filters=512, kernel_size=3, padding=1, stride=1, learning_rate=lr,
                          name='conv10'))
        self.layers.append(ReLu())
        self.layers.append(Maxpooling2D(pool_size=2, stride=2, name='maxpool4'))

        # layer 11
        self.layers.append(
            Convolution2D(inputs_channel=512, num_filters=512, kernel_size=3, padding=1, stride=1, learning_rate=lr,
                          name='conv11'))
        self.layers.append(ReLu())

        # layer 12
        self.layers.append(
            Convolution2D(inputs_channel=512, num_filters=512, kernel_size=3, padding=1, stride=1, learning_rate=lr,
                          name='conv12'))
        self.layers.append(ReLu())

        # layer 13
        self.layers.append(
            Convolution2D(inputs_channel=512, num_filters=512, kernel_size=3, padding=1, stride=1, learning_rate=lr,
                          name='conv13'))
        self.layers.append(ReLu())
        self.layers.append(Maxpooling2D(pool_size=2, stride=2, name='maxpool5'))
        self.layers.append(Flatten())

        # layer 14
        self.layers.append(FullyConnected(num_inputs=2048, num_outputs=4096, learning_rate=lr, name='fc1'))
        self.layers.append(ReLu())

        # layer 15
        self.layers.append(FullyConnected(num_inputs=4096, num_outputs=4096, learning_rate=lr, name='fc2'))
        self.layers.append(ReLu())
        self.layers.append(Dropout())

        # layer 16
        self.layers.append(FullyConnected(num_inputs=4096, num_outputs=10, learning_rate=lr, name='fc3'))
        self.layers.append(Softmax())
        self.layers.append(Dropout())

        # initialize Total number of layers
        self.numOfLayers = len(self.layers)

    # evaluate the crossEntropy to calculate the loss values.
    def crossEntropy(self, inputs, labels):
        out_num = labels.shape[0]
        p = np.sum(labels.reshape(1, out_num) * inputs)
        loss = -np.log(p)
        return loss

    """
    This method runs the training on the model.
    we iteratively call the forward method on all the layers that are initiated in the network model, evaluate the loss
    finally back propagate all the values to identify the weights and biases.
    All the calculated weights are then updated into the outputWeights file that is used for the test runs."""

    def train(self, X_train, y_train, batchSize, epoch, outputWeights):
        totalMatch = 0
        # run for each batch
        for e in range(epoch):
            for batchIteration in range(0, X_train.shape[0], batchSize):
                # define batch data
                if batchIteration + batchSize < X_train.shape[0]:
                    data = X_train[batchIteration:batchIteration+batchSize]
                    label = y_train[batchIteration:batchIteration + batchSize]
                else:
                    data = X_train[batchIteration:X_train.shape[0]]
                    label = y_train[batchIteration:y_train.shape[0]]
                # initialize loss and batch accuracy to 0.
                loss = 0
                batchMatch = 0
                for eachElement in range(batchSize):
                    x = data[eachElement]
                    y = label[eachElement]

                    # forward pass on all the layers
                    # every layers forward method is implemented in the corresponding layer.
                    # propagate the output from 1 layer to the next layer.
                    for l in range(self.numOfLayers):
                        output = self.layers[l].forward(x)
                        x = output
                    # forward pass completes.    
                    
                    # evaluate loss 
                    loss += self.crossEntropy(output, y)
                    # loss evaluation complete.
                    
                    # prediction evaluation and update accuracy..
                    if np.argmax(output) == np.argmax(y):
                        batchMatch += 1
                        totalMatch += 1
                        
                    # backward propagation begin..
                    # partial derivative assignment begin..
                    dy = y
                    # run the layers in the reverse order and run back in the increments of -1 layer
                    # back propagate each layer's gradient to the previous layer.
                    for l in range(self.numOfLayers-1, -1, -1):
                        dout = self.layers[l].backward(dy)
                        dy = dout
                    # backward propagation for all the layers complete.    
                # repeat for all elements in the batch.
                
                #Batch level manipulations begin : 
                
                # evaluate loss for the batch here.
                loss /= batchSize
                
                # evaluate batch level accuracy here.
                batchAccuracy = float(batchMatch)/float(batchSize)
                cumulativeAccuracy = float(totalMatch)/float((batchIteration+batchSize)*(e+1))

                # print output to file..
                file1 = open("../outputConsole.txt", "a")  # append mode
                output_string='=== Epoch: {0:d}/{1:d} === Iter:{2:d} === Loss: {3:.2f} === TAcc: {4:.2f} ''==='.format(e+1,epoch,batchIteration+batchSize,loss,cumulativeAccuracy)
                file1.write(output_string)
                print(output_string)
                file1.write("\n")
                file1.close()

        # Finally Update the weights and biases to the outputFile.
        outputWeightsCalculated = []
        for i in range(self.numOfLayers):
            # call extract on each layer to extract weights on the specific layer and append to outputWeightsCalculated.
            outputWeightsCalculated.append(self.layers[i].extract())
        with open(outputWeights, 'wb') as handle:
            pickle.dump(outputWeightsCalculated, handle, protocol=pickle.HIGHEST_PROTOCOL)

    """This method tests the test data with calculated weights after the training.
        We load all the weights into corresponding layers and then call the forward pass on the layers."""
    def testAlongCalculatedWeights(self, X_test, y_test, inputWeights):
        # load all the calculated weights and biases.
        with open(inputWeights, 'rb') as handle:
            weightsAndBiases = pickle.load(handle, encoding='latin1')

        # load each layer with the respective weights and biases.
        self.layers[0].feed(weightsAndBiases[0]['conv1.weights'], weightsAndBiases[0]['conv1.bias'])
        self.layers[2].feed(weightsAndBiases[2]['conv2.weights'], weightsAndBiases[2]['conv2.bias'])
        self.layers[5].feed(weightsAndBiases[5]['conv3.weights'], weightsAndBiases[5]['conv3.bias'])
        self.layers[7].feed(weightsAndBiases[7]['conv4.weights'], weightsAndBiases[7]['conv4.bias'])
        self.layers[10].feed(weightsAndBiases[10]['conv5.weights'], weightsAndBiases[10]['conv5.bias'])
        self.layers[12].feed(weightsAndBiases[12]['conv6.weights'], weightsAndBiases[12]['conv6.bias'])
        self.layers[14].feed(weightsAndBiases[14]['conv7.weights'], weightsAndBiases[14]['conv7.bias'])
        self.layers[17].feed(weightsAndBiases[17]['conv8.weights'], weightsAndBiases[17]['conv8.bias'])
        self.layers[19].feed(weightsAndBiases[19]['conv9.weights'], weightsAndBiases[19]['conv9.bias'])
        self.layers[21].feed(weightsAndBiases[21]['conv10.weights'], weightsAndBiases[21]['conv10.bias'])
        self.layers[24].feed(weightsAndBiases[24]['conv11.weights'], weightsAndBiases[24]['conv11.bias'])
        self.layers[26].feed(weightsAndBiases[26]['conv12.weights'], weightsAndBiases[26]['conv12.bias'])
        self.layers[28].feed(weightsAndBiases[28]['conv13.weights'], weightsAndBiases[28]['conv13.bias'])
        self.layers[32].feed(weightsAndBiases[32]['fc1.weights'], weightsAndBiases[32]['fc1.bias'])
        self.layers[34].feed(weightsAndBiases[34]['fc2.weights'], weightsAndBiases[34]['fc2.bias'])
        self.layers[36].feed(weightsAndBiases[36]['fc3.weights'], weightsAndBiases[36]['fc3.bias'])

        # run the forward pass on the test data and on all layers and finally predict the label, match and calcualte the accuracy.
        totalAccuracy = 0

        for i in range(len(y_test)):
            x = X_test[i]
            y = y_test[i]
            # forward pass
            for l in range(self.numOfLayers):
                output = self.layers[l].forward(x)
                x = output
            # predict
            if np.argmax(output) == np.argmax(y):
                totalAccuracy += 1

        file1 = open("../outputConsole.txt", "a")  # append mode
        output_string='Test Acc:{0:.2f}'.format(float(totalAccuracy)/float(len(y_test)))
        file1.write(output_string)
        print(output_string)
        file1.write("\n")
        file1.close()
        # put the accuracy details into a file.
