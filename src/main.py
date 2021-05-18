import numpy as np
from Model import VGG16Model
from keras.datasets import mnist
from scipy.ndimage import zoom

# as per mnist dataset we have 10 classes (0 to 9)
num_classes = 10

train_images = []
test_images = []
# loading data from keras datasets
print("Loading data..")
(train_X, train_labels), (test_X, test_labels) = mnist.load_data();

""" The following is the feature engineering for padding the images with 0's
But we are preferring to use zoom for better results, hence commenting the below out.

for image in train_X:
    # we are padding to accommodate multiple max pooling to avoid non-integer shapes
    # hence, converting 28x28 images to (2^6)x64 images
    image = np.pad(image, (18, 18), 'constant', constant_values=(0, 0))
    train_images.append(image)

train_img = train_images[0:10]
for image in test_X:
    # we are padding to accommodate multiple max pooling to avoid non-integer shapes
    # hence, converting 28x28 images to (2^6)x64 images
    image = np.pad(image, (18, 18), 'constant', constant_values=(0, 0))
    test_images.append(image)

train_images = np.array(train_images)
test_images = np.array(test_images)
# Due to time restrictions we are only considering 10 images from each class of data
# total training images: 100
# total testing images: 10
for i in range(10):
    if i == 0: continue
    # initially each class has 6000 images in training dataset
    train_img = np.append(train_img, train_images[6000 * i: 6000 * i + 10], axis=0)

test_img = train_images[0:1]
for i in range(10):
    if i == 0: continue
    # initially each class has 1000 images in testing dataset
    test_img = np.append(test_img, train_images[6000 * i: 6000 * i + 1], axis=0)

test_images = test_img
train_images = train_img """

### Feature Engineering
### 1. Shuffling the dataset so that we have multiple classes of data in a batch.
### 2. Zooming into images to achieve 64 x 64 size

train_i = train_X[0:10]
train_l = train_labels[0:10]
# Due to time restrictions we are only considering 10 images from each class of data
# total training images: 100
# total testing images: 10

# 1. Selecting different class elements in every batch..

for j in range(10):
    for i in range(10):
        if i==0 and j==0: continue
        train_i = np.append(train_i, train_X[6000*i+j:6000*i+j+10], axis=0)
        train_l = np.append(train_l, train_labels[6000*i+j:6000*i+j+10], axis=0)

test_i=test_X[0:10]
test_l=test_labels[0:10]

for k in range(10):
    if k==0: continue
    test_i = np.append(test_i, test_X[1000*k:1000*k+10], axis=0)
    test_l = np.append(test_l, test_labels[1000*k:1000*k+10], axis=0)

# 2. Zooming..
# Convert the image to 64/64 to tackle the
ratio = 64/28
train_images = zoom(train_i, (1, ratio, ratio))
test_images = zoom(test_i, (1, ratio, ratio))
train_labels=train_l
test_labels=test_l

# 3. Shuffle all the data.
np.random.shuffle(train_images)

### Data preparation..
print('Preparing data......')
# 1. Normalizing data
train_images -= int(np.mean(train_images))
train_images //= int(np.std(train_images))
test_images -= int(np.mean(test_images))
test_images //= int(np.std(test_images))

# preparing labels from value to one hot encoded array
# 2. One hot encoding of labels
print(f"Training data shape: {train_images.shape}\nTesting data shape: {test_images.shape}")
training_data = train_images.reshape(1000, 1, 64, 64)
training_labels = np.eye(num_classes)[train_labels]
testing_data = test_images.reshape(100, 1, 64, 64)
testing_labels = np.eye(num_classes)[test_labels]

# initializing VGG16 network
model = VGG16Model()
model.initializeVGG16Network();

print('Training model......')
model.train(training_data, training_labels, 100, 10, 'MagicNumbersWB.pkl')
print("Training complete.. refer to outputConsole.txt for training output.")
print('Testing using calculated weights.....')
model.testAlongCalculatedWeights(testing_data, testing_labels, 'MagicNumbersWB.pkl')
print("Test complete.. refer to outputConsole.txt for test results")