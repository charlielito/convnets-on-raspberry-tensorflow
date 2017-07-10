print("Importing libraries...")
import argparse
import tensorflow as tf
from model import Model
from name import network_name, model_path
from tfinterface.supervised import SupervisedInputs
from PIL import Image
import numpy as np
import random, PIL, time

# seed: reproducible results
seed = 31
np.random.seed(seed=seed)
random.seed(seed)


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=False,
	help="relative or absolute path to the input image")
args = vars(ap.parse_args())

# obtain images
print("Loading image...")
if args["image"] == None: #if not argument was passed take sample image
    image_path = "test/stop.jpg"
else:
    image_path = args["image"]

im = Image.open(image_path)
size = 32
if ( im.size[0] != size or im.size[1] != size): #reshape it
    im = im.resize((size, size), PIL.Image.ANTIALIAS)

features_test = np.array(im)
#features_test, labels_test = next(dataset.test_set.random_batch_arrays_generator(1)) from the testset folder

print("Creating Convolutional Net ...")

graph = tf.Graph()
sess = tf.Session(graph=graph)

# inputs
inputs = SupervisedInputs(
    name = network_name + "_inputs",
    graph = graph,
    sess = sess,
    # tensors
    features = dict(shape = (None, size, size, 3)),
    labels = dict(shape = (None,), dtype = tf.uint8)
)

# create model template
template = Model(
    n_classes = 43,
    name = network_name,
    model_path = model_path,
    graph = graph,
    sess = sess,
    seed = seed,
)

#model
inputs = inputs()
model = template(inputs)

# restore
print("Restoring model...")
model.initialize(restore=True)

# test
print("Making inference! . . .")
t0 = time.time()
predictions = model.predict(features = [features_test])
predictions = np.argmax(predictions, axis=1)
t1 = time.time() - t0
print(predictions)
print("Elapsed time: {} seconds".format(t1))
