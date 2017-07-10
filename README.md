# Running convents in a Raspberry 3

## Requirements
* Latest version of Raspbian Jessie Lite
* Python 2.7+
* Official Tensorflow 1.1.0 for Raspberry compatible from this [repository](https://github.com/samjabrahams/tensorflow-on-raspberry-pi). Short version, just run:

```
# For Python 2.7
wget https://github.com/samjabrahams/tensorflow-on-raspberry-pi/releases/download/v1.1.0/tensorflow-1.1.0-cp27-none-linux_armv7l.whl
sudo pip install tensorflow-1.1.0-cp27-none-linux_armv7l.whl
```

## Python Requirements
* Pillow
* numpy
* pandas
* dataget
* tfinterface
* juypter, matplotlib (optional for visualization)

`Pillow` needs some external libraries to run (install) properly. Install them before running pillow:

```
sudo apt-get install libjpeg8-dev zlib1g-dev libfreetype6-dev liblcms2-dev libwebp-dev libharfbuzz-dev libfribidi-dev tcl8.6-dev tk8.6-dev python-tk
```

Then you can run:

```
sudo pip install -r requirements.txt
```
**Note:** If you had memory problems (aka MemoryError) installing `matplotlib`, try it with
```
sudo pip --no-cache-dir install matplotlib
```

## Dataset
The model aims to solve the [German Traffic Signs](http://benchmark.ini.rub.de/?section=gtsrb&subsection=news) Dataset, which contains 50000 images of 43 different categories.

![alt text][s1] ![alt text][s2] ![alt text][s3] ![alt text][s4] ![alt text][s5] ![alt text][s6] ![alt text][s7] ![alt text][s8] ![alt text][s9]

To download the dataset I used this [library](https://github.com/cgarciae/dataget). Just type:
```
dataget get german-traffic-signs
```

### Model and training

A convolutional neural network was used with the following architecture:

* Inputs: 3 filters (32x32 RGB)
* Convolutional layer: 16 filters, kernel 5x5, padding 'same', ELU activation
* Convolutional layer: 32 filters, kernel 5x5, padding 'same', ELU activation
* Max Pool: kernel 2x2, stride 2
* Convolutional layer: 64 filters, kernel 3x3, padding 'same', ELU activation
* Max Pool: kernel 2x2, stride 2
* Convolutional layer: 64 filters, kernel 3x3, padding 'same', ELU activation
* Flatten vector
* Fully connected: 256 neurons, ELU activation, dropout = 0.15
* Fully connected: 128 neurons, ELU activation
* Dense layer for output: 43 neurons, Softmax activation

The model only has `1,156,747` parameters. That is approx. 13 Mb.

#### Training
The training of the model was accomplished in the cloud using [FloydHub](https://www.floydhub.com/). This [repo](https://github.com/charlielito/supervised-avanzado-german-traffic-signs/tree/red_pequena_prof) contains the procedure to train the model using tensorflow and [tfinterface](https://github.com/cgarciae/tfinterface) that helps building and training the model.

The accuracy of the model is **95,32%**


#### Runing the model and Making Inference
The model from the previous section was saved and located in the subfolder `models`. From there tensorflow reads the weights of the network.

```
python test.py
```

#### Time of inference
The time that takes to classify one image on the Raspberry pi 3 is approximately **0.05 seconds**, almost in real time, quite good! In comparison with an Asus Core i7, in the raspberry it run almost 10x slower.


[s1]: http://benchmark.ini.rub.de/Images/gtsrb/0.png "S"
[s2]: http://benchmark.ini.rub.de/Images/gtsrb/1.png "S"
[s3]: http://benchmark.ini.rub.de/Images/gtsrb/2.png "S"
[s4]: http://benchmark.ini.rub.de/Images/gtsrb/3.png "S"
[s5]: http://benchmark.ini.rub.de/Images/gtsrb/4.png "S"
[s6]: http://benchmark.ini.rub.de/Images/gtsrb/5.png "S"
[s7]: http://benchmark.ini.rub.de/Images/gtsrb/6.png "S"
[s8]: http://benchmark.ini.rub.de/Images/gtsrb/11.png "S"
[s9]: http://benchmark.ini.rub.de/Images/gtsrb/8.png "S"
