import os
import math
import matplotlib.pyplot as plt
import numpy as np

mseList = {"brdf":[], "nee":[], "mis":[]}

with open("mse_brdf", "r") as f:
    line = f.readline()
    while(line != ""):
        mseList["brdf"].append(math.log(float(line)))
        line = f.readline()

with open("mse_nee", "r") as f:
    line = f.readline()
    while(line != ""):
        mseList["nee"].append(math.log(float(line)))
        line = f.readline()

with open("mse_mis", "r") as f:
    line = f.readline()
    while(line != ""):
        mseList["mis"].append(math.log(float(line)))
        line = f.readline()


len_min = min(min(len(mseList["brdf"]), len(mseList["mis"])), len(mseList["nee"]))

lim = min(1024, len_min)
x_list = []
x = 1
while(x < lim):
    x_list.append(x)
    x *= 2



plt.figure()
plt.title("mse")

l1 = plt.plot(np.array(mseList["brdf"][:lim])[x_list], label='brdf')
l1 = plt.plot(np.array(mseList["mis"][:lim])[x_list], label='mis')
l1 = plt.plot(np.array(mseList["nee"][:lim])[x_list], label='nee')

plt.ylabel("ln(mse)")
plt.xlabel("log2(frames)")
# plt.xticks(ticks=[1,2,3,4,5], labels=['1GHz','2GHz','3GHz','4GHz','5GHz'])

#添加图例
plt.legend(loc='best') 

#handles加labels=可以换数据名称，注意要加逗号“l1,=plt……”
#loc为图例的位置，best会自己找数据少的地方放，也可以为'right','upper right','lower left'

# plt.show()
plt.savefig('mse.png')
