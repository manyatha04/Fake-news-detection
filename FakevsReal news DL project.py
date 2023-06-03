#!/usr/bin/env python
# coding: utf-8

# <div style="text-align:center"><span style="color:darkyellow; font-size:3em;"> Fake News Detection using Deep Learning</span></div>

# In[2]:



#Importing the required Libraries
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
import tensorflow as tf 
import re 
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
import seaborn as sns 
plt.style.use('ggplot')
print("Tensorflow version " + tf.__version__)


# This code imports necessary libraries for data analysis and machine learning tasks, including Pandas, Matplotlib, NumPy, and TensorFlow. It also prints the version of TensorFlow that is currently installed.

# # Read Data

# In[3]:


fake_df = pd.read_csv('Fake.csv')
real_df = pd.read_csv('True.csv')


# This code reads data from two CSV files named 'Fake.csv' and 'True.csv' and creates two Pandas dataframes named fake_df and real_df, respectively.

# # Checking for null values

# In[4]:


fake_df.isnull().sum()
real_df.isnull().sum()


# These two lines of code count the number of missing values in each column of the fake_df and real_df Pandas dataframes.

# # Checking for unique values for subject.
# We want both data frames to have a similar distribution.

# In[5]:


fake_df.subject.unique()
real_df.subject.unique()


#  the code outputs a list of the unique values in the subject column of the fake_df and real_df dataframes.

# In[ ]:





# Drop the date from the dataset.
# 

# In[6]:


fake_df.drop(['date', 'subject'], axis=1, inplace=True)
real_df.drop(['date', 'subject'], axis=1, inplace=True)


# Fake news 0 , Real news 1
#  add a new column named class to the fake_df and real_df Pandas dataframes and assign a value of 0 and 1 to the column for the fake and real dataframes, respectively.

# In[7]:


fake_df['class'] = 0 
real_df['class'] = 1


# # Data visualization
# 

# In[8]:


plt.figure(figsize=(10, 5))
plt.bar('Fake News', len(fake_df), color='orange')
plt.bar('Real News', len(real_df), color='green')
plt.title('Distribution of Fake News and Real News', size=15)
plt.xlabel('News Type', size=15)
plt.ylabel('# of News Articles', size=15)


# In[9]:


# Plotting two samples from each class of the dataset
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
fig.subplots_adjust(hspace=0.3, wspace=0.3)
fake_samples = fake_df.sample(2)
real_samples = real_df.sample(2)
for i, sample in enumerate(fake_samples.values):
    axs[0, i].set_title('Fake News')
    axs[0, i].text(0.5, 0.5, sample[0])
for i, sample in enumerate(real_samples.values):
    axs[1, i].set_title('Real News')
    axs[1, i].text(0.5, 0.5, sample[0])
plt.show()


# In[10]:


#Histogram of the length of the news articles:

fake_lengths = fake_df['text'].apply(len)
real_lengths = real_df['text'].apply(len)
plt.hist([fake_lengths, real_lengths], bins=20, label=['Fake News', 'Real News'])
plt.xlabel('Article Length')
plt.ylabel('# of Articles')
plt.title('Length of News Articles')
plt.legend()


# In[11]:


print('Difference in news articles:',len(fake_df)-len(real_df))


# In[12]:


news_df = pd.concat([fake_df, real_df], ignore_index=True, sort=False)
news_df


# # Combining the title with the text,

# 

# In[13]:


news_df['text'] = news_df['title'] + news_df['text']
news_df.drop('title', axis=1, inplace=True)


# # Training and Test Split

# These lines of code split the news data into training and testing datasets for use in a machine learning model. The 'text' column is used as features, and the 'class' column is used as targets. The data is split into 'X_train', 'y_train', 'X_test', and 'y_test' using the 'train_test_split' function from Scikit-learn, with a test size of 0.2 and a random state of 18.

# In[14]:


features = news_df['text']
targets = news_df['class']

X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.20, random_state=18)


# # Normalizing Data
# lower case, get rid of extra spaces, and url links.

# In[15]:


def normalize(data):
    normalized = []
    for i in data:
        i = i.lower()
        # get rid of urls
        i = re.sub('https?://\S+|www\.\S+', '', i)
        # get rid of non words and extra spaces
        i = re.sub('\\W', ' ', i)
        i = re.sub('\n', '', i)
        i = re.sub(' +', ' ', i)
        i = re.sub('^ ', '', i)
        i = re.sub(' $', '', i)
        normalized.append(i)
    return normalized

X_train = normalize(X_train)
X_test = normalize(X_test)


# These lines of code create a 'Tokenizer' object from Keras, which is used to convert text data into numerical sequences for a machine learning model. The 'max_vocab' variable is set to 10000, which specifies the maximum number of words to be included in the vocabulary.
# 
# The 'fit_on_texts' method of the tokenizer is called on the training data 'X_train' to create the vocabulary index based on word frequency. This prepares the text data for use in a machine learning model by converting it into numerical sequences that can be processed by the model.
# 
# 
# 
# 
# 
# 
# 

# In[16]:


max_vocab = 10000
tokenizer = Tokenizer(num_words=max_vocab)
tokenizer.fit_on_texts(X_train)


# # Convert text to vectors (classifier takes only numerical data)

# tokenize the text data in 'X_train' and 'X_test' by converting each word in the text into a numerical index from the vocabulary created by the 'Tokenizer' object. The resulting sequences of indices are stored back into the 'X_train' and 'X_test' variables.

# In[17]:


# tokenize the text into vectors 
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)


# In[18]:


X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, padding='post', maxlen=256)
X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, padding='post', maxlen=256)


# In[19]:


print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)


# # Building the Model.

# In[ ]:





# In[21]:


from tensorflow.keras import regularizers

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(max_vocab, 32, input_length=256, embeddings_regularizer=regularizers.l2(0.001)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,  return_sequences=True, kernel_regularizer=regularizers.l2(0.001))),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16, kernel_regularizer=regularizers.l2(0.001))),
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, kernel_regularizer=regularizers.l2(0.001))
])


# In[22]:


model.summary()


# 
# 

# In[23]:


# We are going to use early stop, which stops when the validation loss no longer improve.
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])
    
history = model.fit(X_train, y_train, epochs=10,validation_split=0.1, batch_size=30, shuffle=True, callbacks=[early_stop])


# This code trains the previously defined Keras model using the fit() method with early stopping enabled to prevent overfitting. The EarlyStopping callback is used to monitor the validation loss and stop training if it does not improve for two consecutive epochs. The compile() method is used to configure the model for training with the binary cross-entropy loss function, the Adam optimizer with a learning rate of 1e-4, and the accuracy metric. The fit() method is used to train the model on the training data with a validation split of 0.1, a batch size of 30, and shuffling at each epoch. The callbacks parameter is used to specify the EarlyStopping callback, and the method returns a history object containing information about the training process.

# 
# # Visualize our training over time
# 
# 

# In[24]:


history_dict = history.history

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']
epochs = history.epoch

plt.figure(figsize=(12,9))
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss', size=20)
plt.xlabel('Epochs', size=20)
plt.ylabel('Loss', size=20)
plt.legend(prop={'size': 20})
plt.show()

plt.figure(figsize=(12,9))
plt.plot(epochs, acc, 'g', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy', size=20)
plt.xlabel('Epochs', size=20)
plt.ylabel('Accuracy', size=20)
plt.legend(prop={'size': 20})
plt.ylim((0.5,1))
plt.show()


# This code plots the training and validation loss and accuracy for the trained Keras model. It extracts the loss, accuracy, validation loss, and validation accuracy values for each epoch from the history.history dictionary and creates two plots using Matplotlib to visualize them.

# # Evaluate the testing set

# In[25]:


model.evaluate(X_test, y_test)


# In[26]:


pred = model.predict(X_test)

binary_predictions = []

for i in pred:
    if i >= 0.5:
        binary_predictions.append(1)
    else:
        binary_predictions.append(0)


# In[27]:


print('Accuracy on testing set:', accuracy_score(binary_predictions, y_test))
print('Precision on testing set:', precision_score(binary_predictions, y_test))
print('Recall on testing set:', recall_score(binary_predictions, y_test))


# # Confusion matrix

# In[28]:



matrix = confusion_matrix(binary_predictions, y_test, normalize='all')
plt.figure(figsize=(16, 9))
ax= plt.subplot()
sns.heatmap(matrix, annot=True, ax = ax)

# labels, title and ticks
ax.set_xlabel('Predicted Labels', size=20)
ax.set_ylabel('True Labels', size=20)
ax.set_title('Confusion Matrix', size=20) 
ax.xaxis.set_ticklabels([0,1], size=15)
ax.yaxis.set_ticklabels([0,1], size=15)


# #  Model with increased network depth

# In[57]:


model_1 = tf.keras.Sequential([
tf.keras.layers.Embedding(max_vocab, 32),
tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),
tf.keras.layers.Dropout(0.2),
tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
tf.keras.layers.Dropout(0.2),
tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
tf.keras.layers.Dropout(0.2),
tf.keras.layers.Dense(64, activation='relu'),
tf.keras.layers.Dropout(0.2),
tf.keras.layers.Dense(1)
])


# In[58]:


model_1.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
optimizer=tf.keras.optimizers.Adam(1e-4),
metrics=['accuracy'])


# In[65]:


history_1 = model_1.fit(X_train, y_train, epochs=10,validation_split=0.1, batch_size=30, shuffle=True, callbacks=[early_stop])


# In[68]:


model_1.evaluate(X_test, y_test)


# In[69]:


pred_1 = model_1.predict(X_test)
binary_predictions_1 = [1 if i >= 0.5 else 0 for i in pred_1]


# In[70]:


print('Accuracy on testing set:', accuracy_score(binary_predictions_1, y_test))
print('Precision on testing set:', precision_score(binary_predictions_1, y_test))
print('Recall on testing set:', recall_score(binary_predictions_1, y_test))


# In[71]:


matrix_1 = confusion_matrix(binary_predictions_1, y_test, normalize='all')
plt.figure(figsize=(16, 9))
ax= plt.subplot()
sns.heatmap(matrix_1, annot=True, ax = ax)


# # Model without regularization

# In[80]:


model_2 = tf.keras.Sequential([
tf.keras.layers.Embedding(max_vocab, 32),
tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16)),
tf.keras.layers.Dense(64, activation='relu'),
tf.keras.layers.Dense(1)
])


# **Model compilation**

# In[81]:


model_2.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
optimizer=tf.keras.optimizers.Adam(1e-4),
metrics=['accuracy'])


# **Model Training**

# In[82]:


history_2 = model_2.fit(X_train, y_train, epochs=10,validation_split=0.1, batch_size=30, shuffle=True, callbacks=[early_stop])


# In[ ]:





# In[83]:


#Visualize the training progress

history_dict_2 = history_2.history

acc_2 = history_dict_2['accuracy']
val_acc_2 = history_dict_2['val_accuracy']
loss_2 = history_dict_2['loss']
val_loss_2 = history_dict_2['val_loss']
epochs_2 = history_2.epoch


# In[84]:


plt.figure(figsize=(12,9))
plt.plot(epochs_2, loss_2, 'r', label='Training loss')
plt.plot(epochs_2, val_loss_2, 'b', label='Validation loss')
plt.title('Training and validation loss', size=20)
plt.xlabel('Epochs', size=20)
plt.ylabel('Loss', size=20)
plt.legend(prop={'size': 20})
plt.show()


# In[85]:


plt.figure(figsize=(12,9))
plt.plot(epochs_2, acc_2, 'g', label='Training acc')
plt.plot(epochs_2, val_acc_2, 'b', label='Validation acc')
plt.title('Training and validation accuracy', size=20)
plt.xlabel('Epochs', size=20)
plt.ylabel('Accuracy', size=20)
plt.legend(prop={'size': 20})
plt.ylim((0.5,1))
plt.show()


# In[86]:


#Model Evaluation

model_2.evaluate(X_test, y_test)


# In[88]:


pred_2 = model_2.predict(X_test)

binary_predictions_2 = []

for i in pred_2:
    if i >= 0.5:
        binary_predictions_2.append(1)
    else:
        binary_predictions_2.append(0)


# In[89]:


print('Accuracy on testing set:', accuracy_score(binary_predictions_2, y_test))
print('Precision on testing set:', precision_score(binary_predictions_2, y_test))
print('Recall on testing set:', recall_score(binary_predictions_2, y_test))


# In[90]:


#Confusion Matrix

matrix_2 = confusion_matrix(binary_predictions_2, y_test, normalize='all')
plt.figure(figsize=(16, 9))
ax= plt.subplot()
sns.heatmap(matrix_2, annot=True, ax = ax)


# The three models have different architectures, with varying numbers of layers, units, and regularization techniques. Here's a comparison between them:
# 
# The first model has an embedding layer, two bidirectional LSTM layers, two dense layers with ReLU activation and dropout, and no regularization. It is a relatively simple model, with a moderate number of units in each layer, and no explicit regularization. It may be prone to overfitting, especially for larger datasets.
# 
# The second model has an embedding layer, three bidirectional LSTM layers with dropout, two dense layers with ReLU activation and dropout, and L2 regularization. It is a more complex model, with larger number of units in each layer, and explicit dropout and L2 regularization added to combat overfitting. This model is suitable for larger datasets or problems that require a more complex architecture.
# 
# The third model has an embedding layer, two bidirectional LSTM layers, one dense layer with ReLU activation, and L2 regularization. It is a simpler model compared to the second model, with fewer layers and units, but still has L2 regularization to help prevent overfitting. This model is suitable for smaller datasets or problems that do not require a very complex architecture.

# #                                                 THANK YOU 
