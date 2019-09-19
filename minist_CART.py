
# import the related functions from module of sklearn and matplotlib
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# load the data and check the type of the given stored data
digits = load_digits()
print(digits)

# check the keys from the dict type data sets
print(digits.keys())

# data exploration
data = digits.data
target = digits.target
images = digits.images

print('Shape of data sets for data, target and images: {}, {}, {}'.format(data.shape, target.shape, images.shape))

# take a example for the first image
print('The first 8 x 8 image:\n', images[0])
print('The first 8 x 8 gray imgae:')
plt.gray()
plt.imshow(images[0])
plt.show()

# seperate the train and test data sets
train_x, test_x, train_y, test_y = train_test_split(data, target, test_size=0.25)

# create the decision tree classifier with gini impurity
clf = DecisionTreeClassifier(criterion = 'gini')
clf.fit(train_x, train_y)
predict_y = clf.predict(test_x)
print('The accuracy score is {}'.format(accuracy_score(predict_y, test_y)))
