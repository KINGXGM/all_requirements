from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import rmsprop_v2

'''
训练数据要经过归一化
例:
原数据:x1=[0,100,20,31051,……]
归一化后:x2[i]=(min(x1)+x1[i])/(max(x1)-min(x1)) i=0,1,2,3,……
数据格式:
训练数据              值
X_train              Y_train
0.2 0.4 0.1 0.3      0
0.1 0.5 0.9 0.4      1
……
'''
#训练数据集
X_train=[]
Y_train=[]
import random
#随机生成一个训练集
for i in range(10):
  X_train.append([random.random(),random.random(),random.random(),random.random()])
  Y_train.append(random.randint(0,3))

#输入层
i_nodes=(4,1)

#隐藏层1
h1_nodes=9

#隐藏层2
h2_nodes=20

#输出层
o_nodes=3

#学习率
l_rate=0.01

model = Sequential()

#输入层
model.add(Dense(units=i_nodes,input_shape=(i_nodes,)))

#隐藏层1
model.add(Dense(units=h1_nodes))

#隐藏层2
model.add(Dense(units=h2_nodes))

#输出层
model.add(Dense(units=o_nodes,activation="softmax"))

#设置学习率与梯度下降方式
model.compile(loss="categorical_crossentropy",optimizer=rmsprop_v2.RMSprop(learning_rate=l_rate))

#输出神经网络信息
model.summary()

#训练次数
t = 100
#批文件大小
b=128
#开始训练
model.fit(X_train,Y_train,batch_size=b,epochs=t)
file_name="model"
model.save(file_name)
from time import sleep
while True:
  #随机生成一个测试数据
  test_input=[random.random(),random.random(),random.random(),random.random()]
  #根据测试数据生成预测值
  test_output=model.predict(test_data)
  #输出
  print(test_input)
  print(test_output)
  #停顿一秒
  sleep(1)
