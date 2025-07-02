import numpy as np 
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
from tensorflow import keras


# iris =load_iris()
# x=iris.data[:,(2,3)]
# y=(iris.target==0).astype(np.int64)

# per_clf=Perceptron()
# per_clf.fit(x,y)
# y_pred = per_clf.predict([[2, 0.5]])
# print(y_pred)


fashion_mnist= keras.datasets.fashion_mnist
(X_train_full,y_train_full),(X_test,y_test)=fashion_mnist.load_data()
print(X_train_full.shape)
x_valid,x_train=X_train_full[:5000]/255.0,X_train_full[5000:]/255.0
y_valid,y_train=y_train_full[:5000]/255.0,y_train_full[5000:]/255.0
class_names=["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
 "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


print(class_names[y_test[9]])