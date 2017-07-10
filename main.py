print("Importing libraries...")
import tensorflow as tf
from model import Model
from name import network_name, model_path
from tfinterface.supervised import SupervisedInputs
from PIL import Image
import numpy as np
import random, PIL, time

# seed: resultados repetibles
seed = 31
np.random.seed(seed=seed)
random.seed(seed)

# obtener imagenes
print("Loading image...")

im = Image.open("test/stop.jpg")
size = 32
if ( im.size[0] != size or im.size[1] != size): #reshape it
    im = im.resize((size, size), PIL.Image.ANTIALIAS)

features_test = np.array(im)

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
