from tensorflow import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense

(x_train,y_train),(x_test,y_test)=mnist.load_data()

x_train=x_train.reshape(-1,784).astype('float32')/255.0
x_test=x_test.reshape(-1,784).astype('float32')/255.0

y_train=keras.utils.to_categorical(y_train,10)
y_test=keras.utils.to_categorical(y_test,10)

model=Sequential()
model.add(Dense(128,input_shape=(784,),activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(10,activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['acc'])
#print(model.summary())
model.fit(x_train,y_train,epochs=20,validation_split=0.1)
loss, accuracy=model.evaluate(x_test,y_test)
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")