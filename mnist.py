#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().system('pip install tensorflow')
import tensorflow as tf


# In[10]:


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0


# In[23]:


model = tf.keras.models.Sequential([
tf.keras.layers.Flatten(input_shape=(28, 28)),
tf.keras.layers.Dense(128, activation = tf.nn.relu),
tf.keras.layers.Dense(128, activation = tf.nn.relu),
tf.keras.layers.Dropout(0.2),
tf.keras.layers.Dense(10, activation = tf.nn.sigmoid)])


# In[24]:


model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])


# In[25]:


model.fit(x_train, y_train, epochs=5)


# In[26]:


model.evaluate(x_test, y_test)


# In[ ]:




