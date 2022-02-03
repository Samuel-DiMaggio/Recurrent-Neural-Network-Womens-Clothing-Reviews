# Recurrent Neural Network (RNN) - Womens Clothing Reviews

## Overview

Working with a Women’s Clothing E-Commerce dataset revolving around the reviews written by customers. I will be performing text classification while building a Recurrent Neural Network (RNN). This dataset contains the following columns:

* Clothing ID
* Age
* Title
* Review Text
* Rating
* Recommended IND
* Positive Feedback Count
* Division Name
* Department Name
* Class Name

### Loading the dataset

```python
import pandas as pd
from sklearn import preprocessing
import re
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('Womens Clothing E-Commerce Reviews.csv')
df.head()
```
![image](https://user-images.githubusercontent.com/47721595/152287187-6f96a3c2-aa5d-46ba-b28e-64ebb644ee2c.png)

```python
df.dtypes
```
![image](https://user-images.githubusercontent.com/47721595/152287236-d226cc49-d852-4783-8352-8bb1c06fa69c.png)

```python
df.isnull().sum(axis = 0)
```
![image](https://user-images.githubusercontent.com/47721595/152287276-46fca9e7-0fd5-49b6-aa64-3caf30b1e17b.png)

```python
df.dropna(inplace = True)
df.isnull().sum()
```
![image](https://user-images.githubusercontent.com/47721595/152287319-edb6e4c5-9d89-451c-80df-06439931ae5c.png)

### Concatenating the Title, Review Text, Division Name, Department Name, and Class Name as a new feature of Reviews.

```python
df['Reviews'] = df['Title'] + ' ' + df['Review Text'] + ' ' + df['Division Name'] + ' ' + df['Department Name'] + ' ' + df['Class Name']

df.head()
```
![image](https://user-images.githubusercontent.com/47721595/152287539-93da5aca-a535-481d-9b61-f704361bd26e.png)

### Data clean up using the new feature of Reviews using regular expressions.

```python
# Remove all special characters, punctuation and spaces
df['Reviews'] = df['Reviews'].apply(lambda x: re.sub(r'[^A-Za-z0-9]+',' ',x))
# Replace special characters,<br />,  in the file
df['Reviews'] = df['Reviews'].apply(lambda x: re.sub(r"<br />", " ", x))
# Remove length <=2
df['Reviews'] = df['Reviews'].apply(lambda x: re.sub(r'\b[a-zA-Z]{1,2}\b', '', x))

df.head()
```
![image](https://user-images.githubusercontent.com/47721595/152287661-dbfed309-70ad-4d3f-a0ab-5a8453079ab2.png)

### Building a RNN model to forecast the Recommended IND based on Reviews using TensorFlow.

```python
X = df['Reviews'].values
y = df['Recommended IND'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)

print(f'X_train size  ={X_train.shape}; X_test size  ={X_test.shape}')
```
![image](https://user-images.githubusercontent.com/47721595/152287774-a8e9d0a2-a776-4746-916e-df2681a9a650.png)

```python
VOCAB_SIZE = 1000
encoder = tf.keras.layers.experimental.preprocessing.TextVectorization(
    max_tokens=VOCAB_SIZE)
encoder.adapt(X_train)

vocab = np.array(encoder.get_vocabulary())
vocab[:20]
```
![image](https://user-images.githubusercontent.com/47721595/152287832-3bf37811-4e1b-483c-b6b8-54613ac6a117.png)

```python
model = tf.keras.Sequential([encoder,
                             tf.keras.layers.Embedding(input_dim=len(encoder.get_vocabulary()),
                                                       output_dim=64,
                                                       mask_zero=True),
                             tf.keras.layers.GRU(128, return_sequences=True),
                             tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128)),
                             tf.keras.layers.Dense(64, activation='relu'),
                             tf.keras.layers.Dense(1)])
 
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])

%%time
history = model.fit(x=X_train,y=y_train,batch_size= 32,epochs=5,
          validation_data=(X_test,y_test), verbose= 1)
```
![image](https://user-images.githubusercontent.com/47721595/152287960-2ee29f4c-6aa4-4954-9ca2-063167a46bdf.png)

### Evaluate the model and check the model fit. 

```python
train_history = pd.DataFrame(history.history)
train_history['epoch'] = history.epoch
sns.lineplot(x='epoch', y ='loss', data =train_history)
sns.lineplot(x='epoch', y ='val_loss', data =train_history)
plt.legend(labels=['train_loss', 'val_loss'])
```
![image](https://user-images.githubusercontent.com/47721595/152288047-fe8950cb-e62c-4890-abb7-81858488f874.png)

By looking at the train and validation loss above, I found that it may underfit the data due to the following reasons.

+ The training loss steadily decrease with a negative slope
+ The validation loss decrease with a small negative slope.

```python
sns.lineplot(x='epoch', y ='accuracy', data =train_history)
sns.lineplot(x='epoch', y ='val_accuracy', data =train_history)
plt.legend(labels=['train_accuracy', 'val_accuracy'])
```
![image](https://user-images.githubusercontent.com/47721595/152288115-a50ed0c0-449c-4869-a932-e3d3ca6cfe7e.png)

```python
y_pred = (model.predict(X_test)> 0.5).astype(int)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)
```
![image](https://user-images.githubusercontent.com/47721595/152288169-22e6ff39-1747-4b02-97ed-18eeda5b0a15.png)

```python
from sklearn.metrics import classification_report
label_names = ['negative', 'positive']
print(classification_report(y_test, y_pred, target_names=label_names))
```
![image](https://user-images.githubusercontent.com/47721595/152288212-fb36bd96-6d5a-4273-a658-10163950fc90.png)

### Conclusion
The accuracy and f1-score is larger than 90%. This is a decent model and I would recomend this based off of this score. 


