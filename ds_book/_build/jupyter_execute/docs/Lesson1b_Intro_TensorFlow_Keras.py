#!/usr/bin/env python
# coding: utf-8

# # Introduction to TensorFlow and Keras

# ## Objectives

# The goal of this notebook is to teach some basics of the TensorFlow framework and the Keras API. 

# ### What is TensorFlow?
# 
# [TensorFlow](https://www.tensorflow.org/guide]) is an open-source framework developed in late 2015 by Google for building various machine learning and deep learning models. TensorFlow is free and open-source, thanks to the Apache Open Source license.
# 
# The main objective of using TensorFlow is to reduce the complexity of implementing computations on large numerical data sets. In practice, these large computations can manifest as training and inference with machine learning or deep learning models.
# 
# TensorFlow was designed to operate with multiple CPUs or GPUs, as well as a growing number of mobile operating systems. The framework includes wrappers in Python, C++, and Java.
# 
# #### How does it work?
# TensorFlow accepts inputs as a multi-dimensional array called a Tensor, which allows the programmer to create dataflow graphs and structures specifying how data travels through. The framework is designed to support creation of a flowchart of operations to be applied to input Tensors, which travel in one direction and out the other.
# 
# #### TensorFlow's structure
# There are three main components to TensorFlow's structure.
# 
# 1. preprocessing the data
# 2. building the model
# 3. training and estimating the model
# 
# The name Tensorflow derives from the way in which the framework receives input in the form of a multi-dimensional array, i.e. the tensors. These tensors travel sequentially through the specified flowchart of the operations, entering at one end and culminating as output at the other end. 
# 
# #### What are the key TensorFlow components?
# **Tensor**
# 
# A tensor is an n-dimensional vector or matrix. In theory, a tensor may represent any form of data. The values belonging to a tensor all share the same data type and often the same shape / dimensionality. A tensor can describe the input data and the output of a calculation. The main difference between a tensor and a conventional n-dimensional array is that tensors are immutable.
# 
# 
# **Graphs**
# 
# TensorFlow uses a graph framework. Graphs collect and summarize all of the calculations and offer several benefits:
# 
# 1. They are designed to work on CPUs or GPUs, as well as on mobile devices.
# 2. Graphs are portable which enables the computations to be saved for immediate or later usage. Otherwise stated, the graph can be frozen and run at a later time.
# 3. Graph calculations are executed by linking tensors together.
# 4. For each tensor, there is a node and an edge. The node carries out the mathematical process and produces endpoint outputs. The input/output connections are represented by the edges. 
# 5. All nodes are linked together, so the graph itself is a depiction of the operations and relationships that exist between the nodes.
# 
# In TensorFlow, all operations are carried out within a graph, which in effect is a series of computations that happen in order. Each individual operation is referred to as an op node. 
# 
# :::{figure-md} TFgraph-fig
# <img src="https://miro.medium.com/max/1838/1*aOYUa3hHKi9HlYnnFAipUQ.gif" width="650px">
# 
# TensorFlow graph example (from [https://medium.com/the-artificial-impostor/notes-understanding-tensorflow-part-1-5f0ebb253ad4](https://medium.com/the-artificial-impostor/notes-understanding-tensorflow-part-1-5f0ebb253ad4)).
# :::
# 
# 
# #### Why do so many people like TensorFlow?
# TensorFlow is intentionally user-friendly, with helpful plugins to visualize model training and a useful software debugging tool. As well, TensorFlow is highly scalable, with easy deployment on both CPUs and GPUs.

# ### What is Keras?
# 
# [Keras](https://keras.io/about/) is an API built on Python which reduces the cognitive load associated with programming models through human readability and simple and consistent structures.
# 
# Keras is what some might call a wrapper for TensorFlow. It is intended for rapid experimentation.
# 
# Tha main components of Keras include:
# 1. A models API, which enables one to construct a model with varying levels of complexity depending on use case. We will use the [Functional API](https://keras.io/guides/functional_api/) subclass.
# 2. A layers API, which allows one to define the tensor in/tensor out computation functions.
# 3. A callback API, which enables one to program specific actions to occur during training, such as log training metrics, visualize interim/internal states and statistics of the model during training, and perform early stopping when the model converges.
# 4. A data preprocessing API, which offers support for prepping raw data from disk to model ready Tensor format.
# 5. An optimizer API where all of the state of the art optimizers can be plugged in. Learning rate decay / scheduling can also be implemented as part of this API.
# 6. A metrics API which is used for assessing the performance of the model during training. A metric is the target to optimize during training, with specific metrics chosen for specific modeling objectives.
# 7. A loss API that informs the model quantitatively how much it should try to minimize during training by providing a measure of error. Similar to metrics, specific loss functions are selected for specific modeling objectives.
# 
# With the Functional API, our main workflow will follow the diagram below.
# 
# :::{figure-md} Keras-fig
# <img src="images/Keras_functional_API.jpg" width="650px">
# 
# Keras Functional API diagram (from [https://miro.com/app/board/o9J_lhnKhVE=/](hhttps://miro.com/app/board/o9J_lhnKhVE=/)).
# :::
