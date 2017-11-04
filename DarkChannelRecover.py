""" a module for a dark channel based algorithm which remove haze on picture """

__author__ = 'Ray'

import numpy as np
import cv2


# 用于排序时存储原来像素点位置的数据结构
class Node(object):
    def __init__(self, x, y, value):
        self.x = x
        self.y = y
        self.value = value

    def print_info(self):
        print('%s:%s:%s' % (self.x, self.y, self.value))


# 获取最小值矩阵
# 获取每个像素BGR三个通道的最小值
def get_min_channel(img):
    # 输入检查
    if len(img.shape) == 3 and img.shape[2] == 3:
        pass
    else:
        print("bad image shape, input must be color image")
        return None

    img_gray = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            local_min = 255
            for k in range(0, 3):
                if img.item((i, j, k)) < local_min:
                    local_min = img.item((i, j, k))
            img_gray[i, j] = local_min

    return img_gray


# 获取暗通道
def get_dark_channel(img, block_size=3):
    # 输入检查
    if len(img.shape) == 2:
        pass
    else:
        print("bad image shape, input image must be two demensions")
        return None

    # blockSize检查
    if block_size % 2 == 0 or block_size < 3:
        print('blockSize is not odd or too small')
        return None

    # 计算addSize
    add_size = int((block_size - 1) / 2)

    new_height = img.shape[0] + block_size - 1
    new_width = img.shape[1] + block_size - 1

    # 中间结果
    img_middle = np.zeros((new_height, new_width))
    img_middle[:, :] = 255

    img_middle[add_size:new_height - add_size, add_size:new_width - add_size] = img

    img_dark = np.zeros((img.shape[0], img.shape[1]), np.uint8)

    for i in range(add_size, new_height - add_size):
        for j in range(add_size, new_width - add_size):
            local_min = 255
            for k in range(i - add_size, i + add_size + 1):
                for l in range(j - add_size, j + add_size + 1):
                    if img_middle.item((k, l)) < local_min:
                        local_min = img_middle.item((k, l))
            img_dark[i - add_size, j - add_size] = local_min

    return img_dark


# 获取全局大气光强度
def get_atomspheric_light(dark_channel, img, mean_mode=False, percent=0.001):
    size = dark_channel.shape[0] * dark_channel.shape[1]
    height = dark_channel.shape[0]
    width = dark_channel.shape[1]

    nodes = []

    # 用一个链表结构(list)存储数据
    for i in range(0, height):
        for j in range(0, width):
            one_node = Node(i, j, dark_channel[i, j])
            nodes.append(one_node)

    # 排序
    nodes = sorted(nodes, key=lambda node: node.value, reverse=True)

    atmospheric_light = 0

    # 原图像像素过少时，只考虑第一个像素点
    if int(percent * size) == 0:
        for i in range(0, 3):
            if img[nodes[0].x, nodes[0].y, i] > atmospheric_light:
                atmospheric_light = img[nodes[0].x, nodes[0].y, i]

        return atmospheric_light

    # 开启均值模式
    if mean_mode:
        sum = 0
        for i in range(0, int(percent * size)):
            for j in range(0, 3):
                sum = sum + img[nodes[i].x, nodes[i].y, j]

        atmospheric_light = int(sum / (int(percent * size) * 3))
        return atmospheric_light

    # 获取暗通道前0.1%(percent)的位置的像素点在原图像中的最高亮度值
    for i in range(0, int(percent * size)):
        for j in range(0, 3):
            if img[nodes[i].x, nodes[i].y, j] > atmospheric_light:
                atmospheric_light = img[nodes[i].x, nodes[i].y, j]

    return atmospheric_light


# 恢复原图像
# Omega 去雾比例 参数
# t0 最小透射率值
def get_recover_scene(img, omega=0.95, t0=0.1, block_size=15, mean_mode=False, percent=0.001):
    img_gray = get_min_channel(img)
    img_dark = get_dark_channel(img_gray, block_size=block_size)
    atmospheric_light = get_atomspheric_light(img_dark, img, mean_mode=mean_mode, percent=percent)

    img_dark = np.float64(img_dark)
    transmission = 1 - omega * img_dark / atmospheric_light

    # 防止出现t小于0的情况
    # 对t限制最小值为0.1
    for i in range(0, transmission.shape[0]):
        for j in range(0, transmission.shape[1]):
            if transmission[i, j] < t0:
                transmission[i, j] = t0

    scene_radiance = np.zeros(img.shape)

    for i in range(0, 3):
        img = np.float64(img)
        scene_radiance[:, :, i] = (img[:, :, i] - atmospheric_light) / transmission + atmospheric_light

        # 限制透射率 在0～255
        for j in range(0, scene_radiance.shape[0]):
            for k in range(0, scene_radiance.shape[1]):
                if scene_radiance[j, k, i] > 255:
                    scene_radiance[j, k, i] = 255
                if scene_radiance[j, k, i] < 0:
                    scene_radiance[j, k, i] = 0

    scene_radiance = np.uint8(scene_radiance)

    return scene_radiance


# 调用示例
def sample():
    img = cv2.imread('tiananmen1.bmp', cv2.IMREAD_COLOR)
    scene_radiance = get_recover_scene(img)

    cv2.imshow('original', img)
    cv2.imshow('test', scene_radiance)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


sample()
