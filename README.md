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

Overview: 

Definition and Parameters:

First we define what it is-residual block- giving in parameters necessary for a CNN. Such as filters(how many neurons), reg(regulariaztion which helps with overfitting and ensures the model stays on course), dropout_rate(what % of the neurons is values is ignored), downsample(will be explained), and expansion(will be explained).

Skip Connection

The first part of the Residual block I want to talk about is the skip connection. This part is vital to our model! Skip connection is when the initial input value is added to the output of a layer or a block of layers. This is done because of the vanishing/exploding gradient problem(Vanishing - Emerges during backpropagation when the slopes/derivates of the activation functions become progressively smaller as we move backward through the layers of the neural network | Exploding - the exact opposite where the gradients during backpropagation become extremely large)

So the first line of this module saves the original value in order to add on to the end value(skip connection BEFORE LAST ACTIVATION FUNCTION).

The second line is activated when ever the input value of downsample is set to true by the user. We do this to decrease the size of the input value. Ex: we have an initial one of 32x32 we do downsampling and the value is decreased to 16x16. This allows for easier and faster training.

First Layer Components:

  Convolutional Layer:

The first layer of code - 1st conv2d to 1st ReLU - we start with a convolutional layer which extracts features from input data. Using a kernel(filter it can be 1x1, 3x3, 5x5, 7x7, etc) which is a small matrices of learnable weights that are used to detect features in the input data(edges, textures, etc.). The output value of this is a feature map representing the respons of a specific filter across the input. Since there is typically multiple we stack them on top of each other which forms an output volume.

  Strides:

Now we move on to the strides which is just how many pixels over does the kernel move. So a stride of 1 would mean the kernal moves over 1 pixel to the right. For visualizations I recommend using this website( https://poloclub.github.io/cnn-explainer/ ).

  Padding:

Now for padding = 'same' adds extra rows and columns of pixels(typically 0s) around the borders of the input img/feature map. With this extra pixels he output(feature map) is the same as the dimensions as the input(only if stride = 1). So I know you must be wondering whats so important about this? Wouldn't it be more benefitial to downsize the image to make the CNN run faster? Well we do this because preservers spatial dimensions, handles edge effects( we don't want to lose important features of the edges), and simplifies network architecture(consistenty allows for easier designing of small and big network architectures).

  Kernel Initializer:

Kernal Initializer = 'he_normal' selects weights from nrmally distributed values with mean (μ)=0 and standard deviation (σ)= √2/√Fan-in. Its important to use an initializer as in most cases initial values for weights are random and the biases are given zero when orignally starting you NN. Which will lead to exploding/vanishing gradient! So using a kernel initalizer such as he_normal will solve that problem.

  Kernel Regularizer:

Now for the kernel_regularizer which is used to prevent overfitting. Overfitting is when your model tunes towards the training data and isn't able to fit to the validation/testing data. To lessen the issue kernel regularization adds a penalty term to the loss function that the network optimizes during training.

  Batch Normalization:

Batch Normalization helps stabilize and accelerating learning. Since its introduction to ML in 2015 it has become a staple. When applying batch normalization for a mini-batch(often powers of 2) it calculates the mean and variance of the activations. Then it normalizes the activations to have a mean of zero and a std of 1. This brings stability to the model and solves the problem of Internal Covariate Shift. So as we know as a NN learns the parameters are adjusted, these updates cause the distribution of the activations of each layer to change; therefore, the model is slower, instable, gradient problems, and has generalization issues(all solved mostly.. by batch normalization).

  ReLU Activation:

Now for the ReLU(Rectified Linear Unit) introducing non-linearity by outputting 0 if the input is negative and outputting the direct input if its positive. Allowing for the model to learn more complex and real-world data. Because if a model does not have non-linearity it is basically just linear regression no matter how complex you make the model. So by adding this you solve that problem and increase a variety of positive factors. This concludes the first layer! I won't be going into this much detail for the next ones. I am going to go more in depth with the architecture and its importance.

  Second Layer:

For this layers its similar to the last with a few key changes. The strides can either be 2 or 1. 1 doesn't affect the shape of the input but 2 will half it so lets say we have 32x32 then it will go to 16x16(as explained earlier).

  Final Step

Then now to if downsample / if somehow the shape doesn't match we can downsample the shortcut_x from earlier, which will flow into the next line of code in which we add f(x) + x = h(x) getting our final output value of the residual block.

Summary: LN[5] 
This is the actual model architeture if you didn't know this is based off resnet :)) 

  Regularizer:

I use the L2 regularizer at 1e-4 as anything above that impaired the models accuracy for me. Its main purpose is to decrease overfitting within my model by punishing large weights. 

  Inputs: 
  
We have to specify the input value necessary in order for the process to occur properly. And using the CIFAR-10 the initial shape is (32,32,3)

The next line we just apply the data augmentation from before. 

Now we make our first and only layer not defined by the residual_block function. Having 64 neurons and a kernal_size of (7,7) the largest one in the whole model! It downsizes the image with strides = 2 making. ( shape = 16x16x64 ) Then we apply batchnormalization and ReLU before moving on to our first residual block. 

  First residual layers??:

We have 2 both with 64 filters, as well as dropout to decrease the chances of overfitting to the data we have presented the model. In the first one of these 2 I applied downsampling making the new shape 8x8x64.

  Second residual layers: 

Downsampled in the first 128 layer(will be common in the rest) new shape --> 4x4x128
Increased the dropout rate by .05 --> new value is .15
Increased neurons from 64 --> 128 

  Third residual layers:

Downsampled in the first 256 layer new shape --> 2x2x256
Increased the dropout rate by .05 --> new value is .2
Increased neurons from 128 --> 256 

  Fourth Residual layers: 
  
Downsampled in the first 512 layer new shape --> 1x1x512
Increased the dropout rate by .05 --> new value is .25
Increased neurons from 256 --> 512

  GAP2D:

Reduced the spatial dimensions(hxW) of a feature map by calculating the average value of all elements within each feature map. It downsamples the shape and improves model efficiency and speed. This step is crucial for the dense layers to properly classify the images. 

  1st Dense Layer: 

Known as a full connected layer it receives input from all the neurons in the previous layer. Allowing the network to learn relationships between features extracted by the conv layers. 

  Dropout: 
APPLYING SO WE DON'T OVERFIT!!

  2nd Dense layer: 

So I know you're looking at the code and wondering why 10? It's so we have a neuron for each type of image in the dataset. I'll give an example lets say the img is a cat. And neuron 4 is the designated one for that. It'll be like neuron 4(cat) - 67% chance its a cat and then neuron 9(airplane) 23% chance its an airplane. We'll go with the one that has the highest probability. This is all possible by the softmax activation that we use in this layer turning it into probability. 





