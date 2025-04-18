import matplotlib.pyplot as plt
import numpy as np
import xlrd
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = ['KaiTi']
mpl.rcParams["axes.unicode_minus"]=False
plt.rcParams['font.size'] = 18

data_path = "./hop_stat.xlsx"
# data_path = "./hops_stat.xlsx"

# sheet_name0 = "drop_rate_sheet"
sheet_name0 = "drop_rate_sheet"
data_excel = xlrd.open_workbook(data_path)
table = data_excel.sheet_by_name(sheet_name=sheet_name0)


x = np.arange(4)
y_base_h = table.row_values(0, start_colx=1, end_colx=None)
y_base_l = table.row_values(1, start_colx=1, end_colx=None)
y_iql_h = table.row_values(2, start_colx=1, end_colx=None)
y_iql_l = table.row_values(3, start_colx=1, end_colx=None)
y_mf_dqn_h = table.row_values(4, start_colx=1, end_colx=None)
y_mf_dqn_l = table.row_values(5, start_colx=1, end_colx=None)

plt.figure(figsize=(8, 6))
bar_width = 0.2
tick_label = ["10", "20", "30", "40"]

plt.bar(x,y_base_h,bar_width,align="center",label="base high",color='#FF8C00')
# plt.bar(x,y_base_l,bar_width,align="center",bottom=y_base_h,label="base low",color='#F4A460')
plt.bar(x+bar_width,y_iql_h,bar_width,align="center",label="iql high",color='#228B22')
# plt.bar(x+bar_width,y_iql_l,bar_width,align="center",bottom=y_iql_h,label="iql low",color='#7FFF00')
plt.bar(x+2*bar_width,y_mf_dqn_h,bar_width,align="center",label="mf-dqn high",color='#4169E1')
# plt.bar(x+2*bar_width,y_mf_dqn_l,bar_width,align="center",bottom=y_mf_dqn_h,label="mf-dqn low",color='#87CEFA')

plt.xlabel("网络节点数")
plt.ylabel("业务时延")
plt.xticks(x+bar_width,tick_label)
plt.legend()
plt.show()
