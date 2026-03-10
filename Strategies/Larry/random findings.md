# Random findings

<https://www.reddit.com/r/MachineLearning/comments/jlzv8n/d_what_is_optimal_number_of_hidden_unit_in/>

Performance of Neural Architectures is allways dependant on the dataset. So there is no way of knowing how the optimal architecture of any practical dataset will look like without trial and error.

In theory you can encode arbitrary many information in a floating point number so the theretical bottleneck is allways a single neuron. However in practice the neural network world runs in mostly in 32 bits or less and also more layers are generally advisiable.

With that being said. You can make some guesses.

I conducted some research on the sizing of layers and how the inference dynamics is distributed here: <https://arxiv.org/abs/2006.08679>. Bottomline for computer vision is, that the receptive field of your neural network should roughly match the input resolution at the bottleneck to gain maximum efficency. Otherwise the system will likely end up with ideling layers that do not add additional information an only act as pass-through layer.

The receptive field is a direct function of (among other things) the kernel sizes of convolutional layers and the position of the downsampling layers. Details can be found here: <https://medium.com/mlreview/a-guide-to-receptive-field-arithmetic-for-convolutional-neural-networks-e0f514068807>

You can also use saturation to figure out how much dimension your encoding needs (see linked paper). I use the delve-package for this to compute the saturation of the layers, roughly 40% in the hidden layers in most cases means good performance in my experience.

In the general case it's allways depndant on how complex your data is. Something like imagenet is very diverse and need a lot of dimension to encode. Something like MNIST can be compresse din 2 Dimensions. Basically it is mostly trial and error at this point.

EDIT: I was mostly referring to deep neural architectures with nonlinear activation functions. The world looks a little bit different for autoencoders with a single hidden layer and linear activation functions. In this case you will basically approximate a linear eigenfunction, which basically translates to something resembling a stochastical-iterative version of PCA. Because of this you could estimate the number of hidden units required by computing the how many eigendirections are required to explain the variance of a reasonably sized subset, like 99% or more. However in this case the question arises why even bother with the autoencoder, because you can get the straight dope from just applying PCA on the dataset without worrying about the architecture.
