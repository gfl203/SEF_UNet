import os.path

def rename(img_folder,num):
    for img_name in os.listdir(img_folder):  # os.listdir()： 列出路径下所有的文件
        #os.path.join() 拼接文件路径
        src = os.path.join(img_folder, img_name)   #src：要修改的目录名
        dst = os.path.join(img_folder, 'IMG'+ str(num) + '.jpg') #dst： 修改后的目录名      注意此处str(num)将num转化为字符串,继而拼接
        num= num+1
        os.rename(src, dst) #用dst替代src


def main():
    img_folder0 = 'C:/Users/ASUS/Desktop/5' #图片的文件夹路径    直接放你的文件夹路径即可
    num=1
    rename(img_folder0,num)

if __name__=="__main__":
    main()

