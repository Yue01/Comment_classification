
"""
Reason why choosing CNN+ LSTM:
1. We used well-pre-trained embedding from Stanford based on large quantity of language materials.
2. CNN has good performance in analyzing sentences after all the words have been transformed into vectors. The partial
properties of a sentence can be well extracted by CNN.
3. By analyzing the length of each word, the sentence has 150 words in average, which seems to be really long. As a result,
the traditional neural network has really serious gradient vanishing problem. That's why we use bi-direction LSTM, which
gather the information from before and after each word.
LSTM has update gate,forget gate and output gate, which has very good performance for memorizing long-short term sentences.
In order to avoid gradient exploding, dropout are also added.
"""


import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, Bidirectional, LSTM, Embedding, Reshape, Multiply, Activation
from keras.layers import Dropout, SpatialDropout1D
from keras.layers import Lambda,  TimeDistributed
from keras.models import Model
from keras import backend as be


train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
classes = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# Unlike Logreg or svm approach, we should tokenize the words before splitting the data set.


def clean_txt(txt):
    stopwords = ['about', 'above', 'across', 'after', 'afterwards', 'doesn', 'does', 'hello', 'again', 'against', 'all',
                 'almost', 'alone', 'along', 'already', 'also', 'although', 'always', 'among', 'amongst', 'amoungst',
                 'amount', 'and', 'another', 'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are',
                 'around', 'back', 'became', 'because', 'become', 'becomes', 'becoming', 'been', 'before', 'beforehand',
                 'behind', 'being', 'below', 'beside', 'besides', 'between', 'beyond', 'bill', 'both', 'bottom', 'but',
                 'call', 'can', 'cannot', 'cant', 'con', 'could', 'couldnt', 'cry', 'describe', 'detail', 'done',
                 'down', 'due', 'during', 'each', 'eight', 'either', 'eleven', 'else', 'elsewhere', 'empty', 'enough',
                 'etc', 'even', 'ever', 'every', 'everyone', 'everything', 'everywhere', 'except', 'few', 'fifteen',
                 'fify', 'fill', 'find', 'fire', 'first', 'five', 'for', 'former', 'formerly', 'forty', 'found', 'four',
                 'from', 'front', 'full', 'further', 'get', 'give', 'had', 'has', 'hasnt', 'have', 'hence', 'her',
                 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him', 'himself', 'his', 'how',
                 'however', 'hundred', 'inc', 'indeed', 'interest', 'into', 'its', 'itself', 'keep', 'last', 'latter',
                 'latterly', 'least', 'less', 'ltd', 'made', 'many', 'may', 'meanwhile', 'might', 'mill', 'mine',
                 'more', 'moreover', 'most', 'mostly', 'move', 'much', 'must', 'myself', 'name', 'namely', 'neither',
                 'never', 'nevertheless', 'next', 'nine', 'nobody', 'none', 'noone', 'nor', 'not', 'nothing', 'now',
                 'nowhere', 'off', 'often', 'once', 'one', 'only', 'onto', 'other', 'others', 'otherwise', 'our',
                 'ours', 'ourselves', 'out', 'over', 'own', 'part', 'per', 'perhaps', 'please', 'put', 'rather', 'same',
                 'see', 'seem', 'seemed', 'seeming', 'seems', 'serious', 'several', 'she', 'should', 'show', 'side',
                 'since', 'sincere', 'six', 'sixty', 'some', 'somehow', 'someone', 'something', 'sometime', 'sometimes',
                 'somewhere', 'still', 'such', 'system', 'take', 'ten', 'than', 'that', 'the', 'their', 'them',
                 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein', 'thereupon',
                 'these', 'they', 'thickv', 'thin', 'third', 'this', 'those', 'though', 'three', 'through',
                 'throughout', 'thru', 'thus', 'together', 'too', 'top', 'toward', 'towards', 'twelve', 'twenty', 'two',
                 'under', 'until', 'upon', 'very', 'via', 'was', 'well', 'were', 'what', 'whatever', 'when', 'whence',
                 'whenever', 'where', 'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether',
                 'which', 'while', 'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'will', 'with',
                 'within', 'without', 'would', 'yet', 'you', 'your', 'yours', 'yourself', 'yourselves', 'the', 'com']
    txt = re.sub(r"\n", " ", txt)
    txt = re.sub("[\<\[].*?[\>\]]", " ", txt)
    txt = txt.lower()
    txt = re.sub(r"\b\w{1,2}\b", " ", txt)
    txt = re.sub(r"[^a-z ]", " ", txt)
    txt = " ".join([x for x in txt.split() if x not in stopwords])
    return txt


def load_data():
    df_1 = pd.read_csv("data/train.csv")
    df_1['comment_text'] = df_1.comment_text.apply(lambda x: clean_txt(x))
    df_2 = pd.read_csv("data/test.csv")
    df_2['comment_text'] = df_2.comment_text.apply(lambda x: clean_txt(x))

    print("Training set size:{}".format(df_1.shape[0]))
    print("Test set size:{}".format(df_2.shape[0]))
    # Separate text and labels
    train_x = df_1.iloc[:, 1]
    train_y = df_1.iloc[:, 2:]
    test_x = df_2.iloc[:, 1]
    return train_x, train_y, test_x


train_x, train_y, test_x = load_data()
print("Data has been loaded")


# Read embedding matrix using a dictionary
# A line of embedding is like this: [word 1,2,3,4,5....]
#                                         <-   300d   ->

embedding_index = {}
with open("data/glove.840B.300d.txt", encoding='utf8') as f:
    for line in f:
        values = line.rstrip().rsplit(' ')
        word = values[0]
        weights = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = weights
print("Embedding has been loaded")

"""
Tokenizer parameters:
1. num_words: We have around 15600 sentences. As much of each sentences overlaps, 
   I choose 5000 as the maximum number of words we would consider.
2. filer: we have implemented word cleaning before, so I left blank here.
3. lower: I have added 'lower()' in the clean text function. However, there are still upper 
   class words left (and I didn't figure out why, maybe because of my computer). So I had to 
   turn this on.
"""

# Words -> tokenizer
tokenizer = Tokenizer(num_words=5000, lower=True)
tokenizer.fit_on_texts(list(train_x)+list(test_x))
# Words in a sentence -> [1,2,3,6,7]...
train_seq = tokenizer.texts_to_sequences(train_x)
test_seq = tokenizer.texts_to_sequences(test_x)

# Now we gonna build our model. The first thing we need to configure is the input form.
sentence_length=[]
for each in train_seq:
    sentence_length.append(len(each))
plt.hist(sentence_length,bins=np.arange(10, 310, 10))


# As can be seen from the graph, most of the sentences have length less than 150.
# So we will limit the long sentences and pad short sentences into length of 150.
train_X = pad_sequences(train_seq, 150)
test_X = pad_sequences(test_seq, 150)


# Now transform each of the sentence into vector
word_index = tokenizer.word_index
fre_words = min(5000, len(word_index))
embedding_matrix = np.zeros((fre_words, 300))

for word, i in word_index.items():
    if i >= fre_words:
        break
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

x_train, x_val, y_train, y_val = train_test_split(train_X, train_y, test_size=0.2, random_state=123)


# model building
inp = Input(shape=(150,))
x = Embedding(5000, 300, weights=[embedding_matrix])(inp)
# dropout系数可调
x = SpatialDropout1D(0.35)(x)
h = Bidirectional(LSTM(128, return_sequences=True, dropout=0.15, recurrent_dropout=0.15))(x)

# 这里我自己也有疑点，这个是有个人写的教程我拿来用的，TimeDistributed具体作用不太清楚
# 后面的网格结构应该没什么要改的，效果还行
u = TimeDistributed(Dense(128, activation='relu', use_bias=False))(h)
alpha = TimeDistributed(Dense(1, activation='relu'))(u)
x = Reshape((150,))(alpha)
x = Activation('softmax')(x)
x = Reshape((150, 1))(x)
x = Multiply()([h, x])
x = Lambda(lambda x: be.sum(x, axis=1))(x)
x = Dense(64, activation="relu")(x)
x = Dropout(0.2)(x)
# 输出层是6，总共有6类的分类

x = Dense(6, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)

# 优化方法： adam， sgd， RMS_prop
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

batch_size = 128
model.fit(x_train, y_train, batch_size=batch_size, epochs=10, validation_data=(x_val, y_val))
