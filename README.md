So this is my first CNN! 

LN[1] We just get the packages we need to actually make the CNN. We also get numpy and sklearn model selection to split the data for the model we will be making 

Summary:
LN[2] We split the data into 3 parts TRAIN which is 70% and test/validation data which is 15% each 

Overview of what the code is:

The first line in this we get the data into var. The 2nd/3rd line of code I concatenate the data into X and Y variables(x - independent var Y - dependent what is the outcome?)
Then in line 4 of the code I split the training data with a temp variable 70:30 
Then in line 5 I split the 30% between 2 variables -test/validation- evenly 15% each.

Summary: 
LN[3] Data Augmentation 

Overview of what the code is: 

Data augmentation is important when it comes to increasing a models accuracy and decreasing its chance of overfitting to the data. In the first real line of code we rescale all the images of the CIFAR-10 dataset to 1/255 of their original size. Why? In order to help the neural network train more stably and ensuring that all features are on the same scale. Moving on to the 2nd line of code we flip the data horizontally. An example of this would be like "d" -> "b" I flipped it! For random rotation there's value of .2 which results in rotating in a random amount in the range of ±20 degrees. Random zoom with a factor of .1 will zoom in -10% / 10%. SO NOW you must be wondering what is the layers?? Well in a CNN you build it with multiple layers and this will be the first ones so we can make sure the model is good.

Summary: 
LN[4] Resnet architecture 

Overview of what the code is: 
First we define what it is-residual block- giving in parameters necessary for a CNN. Such as filters(how many neurons), reg(regulariaztion which helps with overfitting and ensures the model stays on course), dropout_rate(what % of the neurons is values is ignored), downsample(will be explained), and expansion(will be explained). 
The first part of the Residual block I want to talk about is the skip connection. This part is vital to our model! Skip connection is when the initial input value is added to the output of a layer or a block of layers. This is done because of the vanishing/exploding gradient problem(Vanishing - Emerges during backpropagation when the slopes/derivates of the activation functions become progressively smaller as we move backward through the layers of the neural network | Exploding - the exact opposite where the gradients during backpropagation become extremely large) 
So the first line of this module saves the original value in order to add on to the end value(skip connection BEFORE LAST ACTIVATION FUNCTION). The second line is activated when ever the input value of downsample is set to true by the user. We do this to decrease the size of the input value. Ex: we have an initial one of 32x32 we do downsampling and the value is decreased to 16x16. This allows for easier and faster training. The first layer of code - 1st conv2d to 1st ReLU - we start with a convolutional layer which extracts features from input data. Using a kernel(filter) which is a small matrices of learnable weights that are used to detect features in the input data(edges, textures, etc.). The output value of this is a feature map representing the respons of a specific filter across the input. Since there is typically multiple we stack them on top of each other which forms an output volume. Now we move on to the strides which is just how many pixels over does the kernel move over. For visualizations I recommend using this website( https://poloclub.github.io/cnn-explainer/ ) 

