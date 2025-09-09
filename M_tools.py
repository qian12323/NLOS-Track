import numpy as np
import os
import csv
import json

import similaritymeasures

def calculate_metrics(model_output, label):
    """
    计算模型输出和标签之间的 RMS、PCM、area 和 DTW 并打印结果。

    参数:
    model_output (torch.Tensor): 模型的输出，形状为 (B, T, 2)
    label (torch.Tensor): 标签，形状为 (B, T, 2)
    """
    rms = 0.0
    pcm = 0.0
    area = 0.0
    dtw = 0.0
    B, T, _ = model_output.shape
    label = label[:,-T:]
    route = model_output.detach().cpu().numpy()
    label = label.cpu().numpy()

    for b in range(B):
        # 计算 RMS
        rms += np.sqrt(np.mean((route[b] - label[b])**2))
        # 计算 PCM
        pcm += similaritymeasures.pcm(route[b], label[b])
        # 计算 area
        area += similaritymeasures.area_between_two_curves(route[b], label[b])
        # 计算 DTW
        dtw += similaritymeasures.dtw(route[b], label[b])[0]

    # 返回总和
    return  rms, pcm, area, dtw
 

from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

def M_vis_route(route, prediction, kind="train", epoch="temp", date=None):
    T = prediction.shape[0]
    route = route[-T:]

    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.1])
    ax1 = fig.add_subplot(gs[0])  # 修改点：重命名子图变量
    ax2 = fig.add_subplot(gs[1])
    cax = fig.add_subplot(gs[2])
    
    # 统一使用plasma色阶
    cmap = plt.cm.plasma
    
    # 绘制route轨迹
    segments_route = np.stack([route[:-1], route[1:]], axis=1)
    norm_route = plt.Normalize(0, len(route))  # 新增标准化处理
    lc_route = LineCollection(segments_route, cmap=cmap, norm=norm_route, linewidth=2)
    lc_route.set_array(np.arange(len(route)-1))
    ax1.add_collection(lc_route)  # 修改点：使用ax1
    
    # 绘制prediction轨迹
    segments_pred = np.stack([prediction[:-1], prediction[1:]], axis=1)
    norm_pred = plt.Normalize(0, len(prediction))  # 新增标准化处理
    lc_pred = LineCollection(segments_pred, cmap=cmap, norm=norm_pred, linewidth=2)
    lc_pred.set_array(np.arange(len(prediction)-1))
    ax2.add_collection(lc_pred)  # 修改点：使用ax2

    # 修复循环逻辑为独立配置
    for ax, data in [(ax1, route), (ax2, prediction)]:  # 修改点：直接指定ax和数据对
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_aspect('equal')
        ax.axis('off')
        min_coord = data.min() * 0.9
        max_coord = data.max() * 1.1
        ax.set_xlim(min_coord, max_coord)
        ax.set_ylim(min_coord, max_coord)
    
    # 添加统一色阶
    norm = plt.Normalize(0, T)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=norm)
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label('Time Steps', rotation=270, labelpad=20)
    
    # 调整布局间距
    plt.subplots_adjust(wspace=0.15)
    
    # 保持原有的保存逻辑
    plt.tight_layout()
    if date is not None:
        path = f"result/{date}/photo/{kind}"
        os.makedirs(path, exist_ok=True)
        save_path = f"{path}/{epoch}.png"
    else: 
        save_path = f"result/{epoch}.png"
    
    # 保存为透明PNG（与route_visualizer参数一致）
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, transparent=True, dpi=300)
    plt.close()



def record_data_to_csv(epoch,loss, rms, pcm,area,dtw, file_path='training_data.csv'):
    with open(file_path, 'a', newline='') as csvfile:
        fieldnames = ['epoch','T','Loss' ,'RMS', 'PCM', 'Area', "DTW"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if csvfile.tell() == 0:
            writer.writeheader()
        writer.writerow({'epoch': epoch, 'Loss':loss, 'RMS': rms, 'PCM': pcm, 'Area': area, "DTW": dtw})


def record_data_to_json(epoch, loss, rms, pcm, area, dtw, file_path='training_data.json'):
    try:
        # 尝试读取现有的 JSON 文件内容
        with open(file_path, 'r') as jsonfile:
            data = json.load(jsonfile)
    except FileNotFoundError:
        # 如果文件不存在，创建一个空列表用于存储数据
        data = []
    
    # 创建当前 epoch 的数据字典
    new_entry = {
        'epoch': epoch,
        'Loss': loss,
        'RMS': rms,
        'PCM': pcm,
        'Area': area,
        'DTW': dtw
    }
    
    # 将当前 epoch 的数据追加到数据列表中
    data.append(new_entry)
    
    # 将更新后的数据列表写回到 JSON 文件中
    with open(file_path, 'w') as jsonfile:
        json.dump(data, jsonfile, indent=4)


# from scipy.ndimage import gaussian_filter1d
# def gaussian_filter(data, sigma):
#     """
#     高斯滤波函数
#     :param data: 输入数据，形状为 (B, T, 2)
#     :param sigma: 高斯核的标准差
#     :return: 滤波后的数据，形状为 (B, T, 2)
#     """
#     B, T, _ = data.shape
#     filtered_data = np.zeros_like(data)
#     for b in range(B):
#         for dim in range(2):
#             # 对每个批次和每个维度分别进行滤波
#             filtered_data[b, :, dim] = gaussian_filter1d(data[b, :, dim], sigma)
#     return filtered_data

def moving_average_filter(data, window_size):
    """
    移动平均滤波函数
    :param data: 输入数据，形状为 (T, 2)
    :param window_size: 窗口大小
    :return: 滤波后的数据，形状为 (T, 2)
    """
    T, _ = data.shape
    filtered_data = np.zeros_like(data)
    for dim in range(2):
        # 对每个维度分别进行滤波
        filtered_data[:, dim] = np.convolve(data[:, dim], np.ones(window_size)/window_size, mode='same')
    return filtered_data

