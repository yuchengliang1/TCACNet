import numpy as np
import pywt


# 定义函数计算每个通道的WPSer值
def compute_wpser(x, layers=5):
    """
    计算给定EEG信号x的WPSer值
    :param x: 输入的EEG信号
    :param layers: 分解层数（默认5）
    :return: WPSer值
    """
    signal_len = len(x)

    # 初始化y，大小为 layers x (2^layers)，每层每个节点的信号
    y = []  # 使用列表动态存储每层信号

    # 第一层（原始信号）
    y.append([x])  # 第一层包含原始信号

    # 小波包分解（每层的低频和高频信号）
    for layer in range(1, layers + 1):
        current_layer = []
        for node in range(2 ** layer):
            # 小波变换，将信号分为低频和高频部分
            signal = y[layer - 1][node // 2]
            cA, cD = pywt.dwt(signal, 'db1')
            if node % 2 == 0:
                current_layer.append(cA)  # 偶数节点为低通
            else:
                current_layer.append(cD)  # 奇数节点为高通
        y.append(current_layer)  # 存储当前层的信号

    # 计算每个子节点的能量和总能量
    E_total = 0
    for layer in range(layers):
        for node in range(2 ** layer):
            E_total += np.sum(y[layer][node] ** 2)  # 计算每个节点的能量

    # 计算每个节点的能量比例
    wpser = 0
    for layer in range(layers):
        for node in range(2 ** layer):
            P = np.sum(y[layer][node] ** 2) / E_total
            if 8 <= node < 30:  # 判断频率范围 8-30 Hz
                wpser += P

    return wpser


# 主程序
def calculate_wpser_for_epochs(all_epochs_file):
    """
    计算每个通道的WPSer，并将其添加到原始数据的时间维度最后
    :param all_epochs_file: 包含脑电片段数据的文件路径
    """
    # 加载脑电数据
    all_epochs = np.load(all_epochs_file)

    # 获取原始数据的形状
    n_epochs, n_channels, n_time_points = all_epochs.shape

    # 创建一个新的数组来存放包含WPSer的结果
    wpser_epochs = np.zeros((n_epochs, n_channels, n_time_points + 1))

    # 遍历每个通道，计算WPSer值
    for ch in range(n_channels):
        for epoch in range(n_epochs):
            signal = all_epochs[epoch, ch, :]  # 获取该通道的某个片段信号
            wpser_value = compute_wpser(signal)  # 计算该通道的WPSer
            wpser_epochs[epoch, ch, :-1] = signal  # 将原始信号放回
            wpser_epochs[epoch, ch, -1] = wpser_value  # 将WPSer值放在时间维度的最后一位

    # 保存新的数据（包括WPSer）
    np.save('all_epochs_with_wpser.npy', wpser_epochs)
    print(f"保存新的脑电数据，数据形状: {wpser_epochs.shape}")


# 调用函数计算WPSer并保存结果
calculate_wpser_for_epochs('D:/pycharm project/motorimage/output_epochs/all_epochs.npy')  # 替换为你实际的文件路径
