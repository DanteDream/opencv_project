# import tensorflow as tf
# import numpy as np
# import os
# import cv2
# img_Width=208
# img_height=208
#
# def get_files(file_dir):
#     #file_dir:文件夹路径
#     #return 乱序后的图片和标签
#
#     cats=[]
#     label_cats=[]
#     dogs=[]
#     label_dogs=[]
#
#     a=[]
#     label_a=[]
#     b = []
#     label_b = []
#     c = []
#     label_c = []
#     d = []
#     label_d = []
#     g = []
#     label_g = []
#     h = []
#     label_h= []
#     i = []
#     label_i = []
#     l = []
#     label_l = []
#     v = []
#     label_v = []
#     y = []
#     label_y = []
#
#     #载入数据路径并写入标签
#
#     for file in os.listdir(file_dir): #Return a list containing the names of the files in the directory.
#         name=file.split(sep='.')
#         if name[0]=='a':
#             # cats.append(file_dir+file)
#             # label_cats.append(0)
#             a.append(file_dir + file)
#             label_a.append(0)
#         elif name[0]=='b':
#             # dogs.append(file_dir+file)
#             # label_dogs.append(1)
#             b.append(file_dir + file)
#             label_b.append(1)
#         elif name[0] == 'c':
#             c.append(file_dir + file)
#             label_c.append(2)
#         elif name[0] == 'd':
#             d.append(file_dir + file)
#             label_d.append(3)
#         elif name[0] == 'g':
#             g.append(file_dir + file)
#             label_g.append(4)
#         elif name[0] == 'h':
#             h.append(file_dir + file)
#             label_h.append(5)
#         elif name[0] == 'i':
#             i.append(file_dir + file)
#             label_i.append(6)
#         elif name[0] == 'l':
#             l.append(file_dir + file)
#             label_l.append(7)
#         elif name[0] == 'v':
#             v.append(file_dir + file)
#             label_v.append(8)
#         elif name[0] == 'y':
#             y.append(file_dir + file)
#             label_y.append(9)
#
#         #print('There are %d cats\nThere are %d dogs'% (len(cats),len(dogs)))
#
#     #打乱文件顺序
#
#     # image_list=np.hstack((cats,dogs)) #将cat和dog水平方式连接
#     image_list = np.hstack((a,b,c,d,g,h,i,l,v,y))
#     # label_list=np.hstack((label_cats,label_dogs))
#     label_list = np.hstack((label_a, label_b,label_c,label_d,label_g, label_h,label_i,label_l,label_v,label_y))
#     temp=np.array([image_list,label_list])#将image_list和label_list再次合并
#     temp=temp.transpose() #转置 将2X25000变成25000x2
#     np.random.shuffle(temp)#随机打乱temp
#
#     image_list=list(temp[:,0])#提取第1列向量
#     label_list=list(temp[:,1])#提取第二列向量
#     label_list=[int(float(i)) for i in label_list]
#
#     return image_list,label_list
#
#
# #生成相同大小的批次
# def get_batch(image,label,image_W,image_H,batch_size,capacity):
#     #batch_size:每个batch有多少张图片
#     #capacity:队列容量
#     #return：图像和标签的batch
#
#     #将python.list类型转换成tf能够识别的格式
#     image=tf.cast(image,tf.string)
#     label=tf.cast(label,tf.int32)
#
#     #生成队列
#     input_queue=tf.train.slice_input_producer([image,label])
#     #建立一个队列，将图片和标签放进一个队列中
#
#     image_contents=tf.read_file(input_queue[0])#获取图片内容
#     label=input_queue[1]#获取标签内容
#     image=tf.image.decode_jpeg(image_contents,channels=3)#对图片进行解码
#
#     #图片统一大小
#
#     #官方方法
#     image=tf.image.resize_image_with_crop_or_pad(image,image_W,image_H)
#     #裁剪图片从中间往四周裁剪,最后只剩中间区域
#
#     #另外一种
#     # image=tf.image.resize_images(image,[image_H,image_W],#把图片进行缩放
#     #                              method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)#插值法
#     #image=tf.cast(image,tf.float32)
#     image=tf.image.per_image_standardization(image)#标准化数据
#     image_batch,label_batch=tf.train.batch([image,label],
#                                            batch_size=batch_size
#                                            ,num_threads=64
#                                            ,capacity=capacity)
#
#     #多余行
#     label_batch=tf.reshape(label_batch,[batch_size])
#     image_batch = tf.cast(image_batch, tf.float32)
#
#     return image_batch,label_batch
#
#
import tensorflow as tf
import numpy as np
import os

img_Width=208
img_height=208

def get_files(file_dir):
    #file_dir:文件夹路径
    #return 乱序后的图片和标签
    a = []
    b = []
    c = []
    d = []
    g = []
    h = []
    i = []
    l = []
    v = []
    y = []
    label_a = []
    label_b = []
    label_c = []
    label_d = []
    label_g = []
    label_h = []
    label_i = []
    label_l = []
    label_v = []
    label_y = []
    dogs=[]
    label_dogs=[]

    #载入数据路径并写入标签

    for file in os.listdir(file_dir): #Return a list containing the names of the files in the directory.
        name=file.split(sep='.')
        if name[0]=='a':
            a.append(file_dir+file)
            label_a.append(0)
        if name[0]=='b':
            b.append(file_dir+file)
            label_b.append(1)
        if name[0]=='c':
            c.append(file_dir+file)
            label_c.append(2)
        if name[0]=='d':
            d.append(file_dir+file)
            label_d.append(3)
        if name[0]=='g':
            g.append(file_dir+file)
            label_g.append(4)
        if name[0]=='h':
            h.append(file_dir+file)
            label_h.append(5)
        if name[0]=='i':
            i.append(file_dir+file)
            label_i.append(6)
        if name[0]=='l':
            l.append(file_dir+file)
            label_l.append(7)
        if name[0]=='v':
            v.append(file_dir+file)
            label_v.append(8)
        if name[0]=='y':
            y.append(file_dir+file)
            label_y.append(9)

        #print(r'There are %d a\nThere are %d b'% (len(a),len(b)))

    #打乱文件顺序

    image_list=np.hstack((a,b,c,d,g,h,i,l,v,y)) #将cat和dog水平方式连接
    label_list=np.hstack((label_a,label_b,label_c,label_d,label_g,label_h,label_i,label_l,label_v,label_y,))
    temp=np.array([image_list,label_list])#将image_list和label_list再次合并
    temp=temp.transpose() #转置 将2X25000变成25000x2
    np.random.shuffle(temp)#随机打乱temp

    image_list=list(temp[:,0])#提取第1列向量
    label_list=list(temp[:,1])#提取第二列向量
    label_list=[int(float(i)) for i in label_list]



    return image_list,label_list

def get_batch(image,label,image_W,image_H,batch_size,capacity):
    #batch_size:每个batch有多少张图片
    #capacity:队列容量
    #return：图像和标签的batch

    #将python.list类型转换成tf能够识别的格式
    image=tf.cast(image,tf.string)
    label=tf.cast(label,tf.int32)

    #生成队列
    input_queue=tf.train.slice_input_producer([image,label])
    #建立一个队列，将图片和标签放进一个队列中

    image_contents=tf.read_file(input_queue[0])#获取图片内容
    label=input_queue[1]#获取标签内容
    image=tf.image.decode_jpeg(image_contents,channels=3)#对图片进行解码

    #图片统一大小

    #官方方法
    image=tf.image.resize_image_with_crop_or_pad(image,image_W,image_H)
    #裁剪图片从中间往四周裁剪,最后只剩中间区域

    #另外一种
    # image=tf.image.resize_images(image,[image_H,image_W],#把图片进行缩放
    #                              method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)#插值法
    #image=tf.cast(image,tf.float32)
    image=tf.image.per_image_standardization(image)#标准化数据
    image_batch,label_batch=tf.train.batch([image,label],
                                           batch_size=batch_size
                                           ,num_threads=64
                                           ,capacity=capacity)

    #多余行
    label_batch=tf.reshape(label_batch,[batch_size])
    image_batch = tf.cast(image_batch, tf.float32)

    return image_batch,label_batch