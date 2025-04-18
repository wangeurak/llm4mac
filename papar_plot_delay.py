#import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
#mpl.rcParams["font.sans-serif"]=["SimHei"]
#mpl.rcParams["axes.unicode_minus"]=False

#mpl.rcParams["font.sans-serif"]=["SimHei"]
#mpl.rcParams["axes.unicode_minus"]=False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['legend.title_fontsize'] = 14

data=np.loadtxt("E:\search\data\excel\\0313delay.txt")
# some simple data
x = np.arange(5)
NO = data[:,5]
LEACH = data[:,2]
LEACH_MF = data[:,3]-LEACH
JDE = data[:,0]
JDE_MF = data[:,1]-JDE
MDAC = data[:,4]

plt.figure(figsize=(8, 6))
bar_width = 0.15
tick_label=["40","60","80","100","120"]
# create bar
plt.bar(x,NO,bar_width,align="center",label="no adaptive clustering",color='#8ECFC9',hatch='\\') #no
plt.bar(x+bar_width,JDE,bar_width,align="center",label="LEACH-Ckmeans",color='#FFBE7A',hatch='--') #JDE
plt.bar(x+2*bar_width,LEACH,bar_width,align="center",label="JDECDR",color='#FA7F6F',hatch='..') #leach
#plt.bar(x+bar_width,LEACH_MF,bar_width,align="center",bottom=LEACH,label="LEACH_MF",color='gold') #leach
#plt.bar(x+2*bar_width,JDE_MF,bar_width,align="center",bottom=JDE,label="JDE_MF",color='dodgerblue') #JDE
plt.bar(x+3*bar_width,MDAC,bar_width,align="center",label="MDAC",color='#82B0D2',hatch='//') #MDAC

# set x,y_axis label
plt.xlabel("The number of UAVs")
plt.ylabel("Communication Delay(s)")
plt.xticks(x+bar_width*1.5,tick_label)
plt.legend()
plt.show()

#plt.savefig("E:\search\data\excel\\delay.png")

