
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split

# The stopwords are collected from the internet. According to the material, some words like 'hello' are added.
"""
* Remove stopwords
* Letters to lower case
* Remove numbers and special symbols or other special tags
"""
def cleantxt(txt):
    stopwords = ['about', 'above', 'across', 'after', 'afterwards', 'doesn', 'does', 'hello', 'again', 'against', 'all', 'almost', 'alone', 'along', 'already', 'also', 'although', 'always', 'among', 'amongst', 'amoungst', 'amount', 'and', 'another', 'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are', 'around', 'back', 'became', 'because', 'become', 'becomes', 'becoming', 'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides', 'between', 'beyond', 'bill', 'both', 'bottom', 'but', 'call', 'can', 'cannot', 'cant', 'con', 'could', 'couldnt', 'cry', 'describe', 'detail', 'done', 'down', 'due', 'during', 'each', 'eight', 'either', 'eleven', 'else', 'elsewhere', 'empty', 'enough', 'etc', 'even', 'ever', 'every', 'everyone', 'everything', 'everywhere', 'except', 'few', 'fifteen', 'fify', 'fill', 'find', 'fire', 'first', 'five', 'for', 'former', 'formerly', 'forty', 'found', 'four', 'from', 'front', 'full', 'further', 'get', 'give', 'had', 'has', 'hasnt', 'have', 'hence', 'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'however', 'hundred', 'inc', 'indeed', 'interest', 'into', 'its', 'itself', 'keep', 'last', 'latter', 'latterly', 'least', 'less', 'ltd', 'made', 'many', 'may', 'meanwhile', 'might', 'mill', 'mine', 'more', 'moreover', 'most', 'mostly', 'move', 'much', 'must', 'myself', 'name', 'namely', 'neither', 'never', 'nevertheless', 'next', 'nine', 'nobody', 'none', 'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'off', 'often', 'once', 'one', 'only', 'onto', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'part', 'per', 'perhaps', 'please', 'put', 'rather', 'same', 'see', 'seem', 'seemed', 'seeming', 'seems', 'serious', 'several', 'she', 'should', 'show', 'side', 'since', 'sincere', 'six', 'sixty', 'some', 'somehow', 'someone', 'something', 'sometime', 'sometimes', 'somewhere', 'still', 'such', 'system', 'take', 'ten', 'than', 'that', 'the', 'their', 'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein', 'thereupon', 'these', 'they', 'thickv', 'thin', 'third', 'this', 'those', 'though', 'three', 'through', 'throughout', 'thru', 'thus', 'together', 'too', 'top', 'toward', 'towards', 'twelve', 'twenty', 'two', 'under', 'until', 'upon', 'very', 'via', 'was', 'well', 'were', 'what', 'whatever', 'when', 'whence', 'whenever', 'where', 'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while', 'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'will', 'with', 'within', 'without', 'would', 'yet', 'you', 'your', 'yours', 'yourself', 'yourselves', 'the', 'com']
    txt = re.sub(r"\n", " ", txt)
    txt = re.sub("[\<\[].*?[\>\]]", " ", txt)
    txt = txt.lower()
    txt = re.sub(r"\b\w{1,2}\b", " ",txt)
    txt = re.sub(r"[^a-z ]", " ", txt)
    txt = " ".join([x for x in txt.split() if x not in stopwords])
    return txt

def load_data():
    df = pd.read_csv("data/train.csv")
    df['comment_text'] = df.comment_text.apply(lambda x : cleantxt(x))
    
    #seperate text and labels
    X = df.iloc[:,1]
    y = df.iloc[:,2:]
    
    # split the data into train,validation and test by 0.6,0.3 and 0.1 with reandom seed 1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.25, random_state=1)

    return X_train, X_val, X_test, y_train, y_val, y_test


X_train, X_val, X_test, y_train, y_val, y_test = load_data()


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
# This fit_transform is a batch normalization operation, which would fasten the speed of convergence and deal with NaN values here.
# The following transform is to apply the same μ and θ to the validation set and test set.
train_vec = vectorizer.fit_transform(X_train)
val_vec = vectorizer.transform(X_val)
test_vec = vectorizer.transform(X_test)

# Make one-on-one prediction for each tag
col = ['toxic', 'severe_toxic', 'obscene', 'threat','insult', 'identity_hate']
pred_train = np.zeros((X_train.shape[0],len(col)))
pred_test = np.zeros((X_test.shape[0],len(col)))
pred_val = np.zeros((X_val.shape[0],len(col)))

from sklearn.linear_model import LogisticRegression
# According to the official tutorial, the LogReg will use liblinear solver,1.0 regularization strength and ovr (binary) mode 
# for we currently do not need the possibility distribution.
LogR = LogisticRegression(solver='lbfgs')
for i,x in enumerate(col):
    LogR.fit(train_vec, y_train[x])
    pred_train[:,i] = LogR.predict_proba(train_vec)[:,1]
    pred_val[:,i] = LogR.predict_proba(val_vec)[:,1]

# Use Roc_auc to evaluate the results
from sklearn import metrics
for i,x in enumerate(col):
    print("-------")
    print("Label: ",x)
    print("ROC_AUC_Score for training set: ",metrics.roc_auc_score(y_train[x], pred_train[:,i]))
    print("ROC_AUC_Score for validation set: ",metrics.roc_auc_score(y_val[x], pred_val[:,i]))

vect2 = TfidfVectorizer(decode_error='ignore',stop_words='english')
train_vec = vect2.fit_transform(X_train)
val_vec = vect2.transform(X_val)
test_vec = vect2.transform(X_test)

from sklearn import svm
import matplotlib.pyplot as plt   # use matplotlib for plotting with inline plots
%matplotlib inline # if you use ipython notebook, add this line

C= np.linspace(1,21,11)
score =np.zeros(C.shape[0])
for i,c in enumerate(C):
    clf = svm.SVC(C=c,decision_function_shape='ovo')
    clf.fit(train_vec, y_train.severe_toxic)
    y_pred= clf.predict(train_vec)
    score[i]=metrics.accuracy_score(y_train.severe_toxic,y_pred)
plt.plot(C, score, marker='^',color = "red")

