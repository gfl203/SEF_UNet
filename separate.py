import os

# 文件夹筛选器
formFilter = lambda form: \
    lambda f: True if f[-len(form):] == form else False

# 只对文件名称中有mark的进行操作
markFilter = lambda mark: \
    lambda f: True if mark in f else False

# 输入文件夹，返回文件夹下所有dType类型数据
getAllFiles = lambda path, dType: \
    list(filter(formFilter(dType), os.listdir(path)))  # 筛选文件


# 新建文件夹
def mkDir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def splitFolder(path=r'C:\Users\ASUS\Desktop\output', mark="groundtruth"):
    if path[-1] != '\\':
        path += "\\"
    files = getAllFiles(path, "png")
    files = list(filter(markFilter(mark), files))
    for f in files:
        try:
            folder = f.split(mark)[0]
            newName = f[len(folder) + 1:]
            mkDir(path + folder)
            os.rename(path + f, path + folder + '\\' + newName)
        except:
            print(f)


if __name__ == "__main__":
    splitFolder()