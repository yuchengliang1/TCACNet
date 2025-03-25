import numpy as np
import pandas as pd
import os

def npy_to_csv(npy_file_path, csv_file_path=None):
    """
    将.npy文件转换为CSV格式
    
    参数:
    npy_file_path: .npy文件的路径
    csv_file_path: 输出CSV文件的路径，如果为None，则使用与npy文件相同的名称但扩展名为.csv
    """
    # 加载.npy文件
    data = np.load(npy_file_path)
    
    # 如果没有指定CSV文件路径，则使用相同的文件名但扩展名为.csv
    if csv_file_path is None:
        base_path = os.path.splitext(npy_file_path)[0]
        csv_file_path = base_path + '.csv'
    
    # 根据数据的维度选择适当的保存方法
    if data.ndim == 1:
        # 一维数组
        pd.DataFrame(data).to_csv(csv_file_path, index=False, header=False)
    elif data.ndim == 2:
        # 二维数组
        pd.DataFrame(data).to_csv(csv_file_path, index=False, header=False)
    else:
        # 处理多维数组
        # 对于三维或更高维的数组，需要重塑或分别保存每个维度
        shape = data.shape
        print(f"原始数据形状: {shape}")
        
        # 例如，对于三维数组，可以将其保存为多个CSV文件
        if data.ndim == 3:
            # 假设数据形状为[epochs, channels, timepoints]
            for i in range(shape[0]):
                epoch_csv_path = f"{os.path.splitext(csv_file_path)[0]}_epoch_{i}.csv"
                pd.DataFrame(data[i]).to_csv(epoch_csv_path, index=False, header=False)
            print(f"已将{shape[0]}个epochs保存为单独的CSV文件")
        else:
            # 对于更高维的数组，可以先将其重塑为2D，然后保存
            # 注意：这可能会丢失多维结构信息
            reshaped_data = data.reshape(shape[0], -1)
            pd.DataFrame(reshaped_data).to_csv(csv_file_path, index=False, header=False)
            print(f"已将多维数组重塑为2D并保存为CSV，形状: {reshaped_data.shape}")
    
    print(f"已将.npy文件保存为CSV: {csv_file_path}")

# 示例用法
if __name__ == "__main__":
    npy_file = "x0.npy"  # 替换为你的.npy文件路径
    npy_to_csv(npy_file)