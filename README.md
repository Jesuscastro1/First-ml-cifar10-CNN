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

Data augmentation is important when it comes to increasing a models accuracy and decreasing its chance of overfitting to the data. In the first real line of code we rescale all the images of the CIFAR-10 dataset to 1/255 of their original size. Why? Well we do this in order to lower the computational cost of each image and that many CNNs are designed to work with a fixed size. Moving on to the 2nd line of code we flip the data horizontally. An example of this would be like "/" -> "\" I flipped it! For random rotation there's value of .2 which results in rotating in a random amount in the range of [-20% * 360, 20% 360]. Random zoom with a factor of .1 will zoom in -10% / 10%. SO NOW you must be wondering what is the layers?? Well in a CNN you build it with multiple layers and this will be the first ones so we can make sure the model is good.

Summary: 
LN[4] Resnet architecture 

Overview of what the code is: 

