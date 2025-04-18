import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
mpl.rcParams["font.sans-serif"]=["SimHei"]
mpl.rcParams["axes.unicode_minus"]=False

# some simple data
x = np.arange(4)
y000 = [0.1,0.2,0.3,0.4]
y001 = [0.4,0.3,0.2,1]
y010 = [0.5,0.4,0.3,0.2]
y011 = [0.2,0.3,0.4,0.5]
y100 = [0.5,0.6,0.7,0.8]
y101 = [0.4,0.3,0.2,0.2]
y110 = [0.7,0.5,0.4,0.6]
y111 = [0.5,0.2,0.3,0.4]

bar_width = 0.15
tick_label=["A","B","C","D"]
# create bar
plt.bar(x,y000,bar_width,align="center",color="c",label="班级 A",alpha=0.5)
plt.bar(x,y001,bar_width,align="center",bottom=y000,color="b",label="班级 B",alpha=0.5)
plt.bar(x+bar_width,y010,bar_width,align="center",color="#66c2a5",label="班级 C",alpha=0.5)
plt.bar(x+bar_width,y011,bar_width,align="center",bottom=y010,color="#8da0cb",label="班级 D",alpha=0.5)
plt.bar(x-bar_width,y100,bar_width,align="center",color="#2580a5",label="班级 E",alpha=0.5)
plt.bar(x-bar_width,y101,bar_width,align="center",bottom=y100,color="#8d44ea",label="班级 F",alpha=0.5)
plt.bar(x+2*bar_width,y110,bar_width,align="center",color="#58c7bc",label="班级 G",alpha=0.5)
plt.bar(x+2*bar_width,y111,bar_width,align="center",bottom=y110,color="#95da3c",label="班级 H",alpha=0.5)

# set x,y_axis label
plt.xlabel("测试难度")
plt.ylabel("试卷份数")
plt.xticks(x+bar_width/2,tick_label)
plt.legend()
plt.show()

