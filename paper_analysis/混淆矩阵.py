#confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
# classes = ['A','B','C','D','E']
# confusion_matrix = np.array([(9,1,3,4,0),(2,13,1,3,4),(1,4,10,0,13),(3,1,1,17,0),(0,0,0,1,14)],dtype=np.float64)


# 标签
classes=['background','Corn leaves','Corn leaf Spot']

# 标签的个数
classNamber=3 #表情的数量

# 在标签中的矩阵
# confusion_matrix = np.array([
#     (94279918,3949631,247424),
#     (1667212,197910770,4787428),
#     (302896,3391907,31411054),
#     ],dtype=np.float64)
confusion_matrix = np.array([
    (96444912,1916501,115551),
    (823891,200716885,2824634),
    (81652,2613333,32410872),
    ],dtype=np.float64)
# 计算每个元素的平方和开方作为归一化因子
# norm_factor = np.sqrt(np.sum(confusion_matrix ** 2, axis=0))
# # 归一化矩阵
# confusion_matrix = confusion_matrix / norm_factor
# # 计算每一列的和
# column_sum = np.sum(confusion_matrix, axis=0)
# # 将每一列除以对应列的和
# confusion_matrix = confusion_matrix / column_sum
# confusion_matrix = np.round(confusion_matrix, 3)

plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)  #按照像素显示出矩阵
plt.title('confusion_matrix')
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=-45)
plt.yticks(tick_marks, classes)

thresh = confusion_matrix.max() / 2.
#iters = [[i,j] for i in range(len(classes)) for j in range((classes))]
#ij配对，遍历矩阵迭代器
iters = np.reshape([[[i,j] for j in range(classNamber)] for i in range(classNamber)],(confusion_matrix.size,2))
for i, j in iters:
    plt.text(j, i, format(confusion_matrix[i, j]),va='center',ha='center')   #显示对应的数字

plt.ylabel('Real label')
plt.xlabel('Prediction')
plt.tight_layout()
plt.savefig("混淆矩阵.pdf",dpi=120)
plt.show()


