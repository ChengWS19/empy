import empymod
import numpy as np
import matplotlib.pyplot as plt
# 按行读取model.txt文件
file_path = "model.txt"  # 替换为实际的文件路径
with open(file_path, 'r') as file:
    lines = file.readlines()
data = []
for line in lines:
    line = line.strip()
    if line:
        values = line.split(',')
        data.append(values)
# 查看data的数据类型和大小
# print(data,len(data))
# 将model转换为numpy数组
model = np.array([np.array(row, dtype=np.float64) for row in data[:-2]], dtype=object)
src = model[0]
rec = model[1]
depth = model[2]
res_list = model[3]
srcpts = model[4]
srcpts = int(srcpts)
time_range = model[5]
strength = model[6]
signal = model[7]
flag = data[-2][0]
outprefix = data[-1][0]
time = np.logspace(time_range[0], time_range[1], int(time_range[2]))
verb = 3
if flag == 'E':
    mrec = False
    # 计算电场响应
    EMfield = empymod.bipole(src, rec, depth, res_list, freqtime=time, mrec=mrec, srcpts=srcpts, signal=signal, verb=verb, strength=strength)
elif flag == 'B':
    mrec = True
    # 计算磁场响应
    mu = 4 * np.pi * 1e-7
    EMfield = mu * empymod.bipole(src, rec, depth, res_list, freqtime=time, mrec=mrec, srcpts=srcpts, signal=signal, verb=verb, strength=strength)
else:
    print('flag error!set flag = E')
    mrec = False
    # 计算电场响应
    EMfield = empymod.bipole(src, rec, depth, res_list, freqtime=time, mrec=mrec, srcpts=srcpts, signal=signal, verb=verb, strength=strength)
# 输出时间和结果到指定目录文件
EMfield_out = np.column_stack((time, EMfield))
# 保存文件名为outprefix.txt
np.savetxt( outprefix + '.dat', EMfield_out, fmt='%.6e', delimiter=',')