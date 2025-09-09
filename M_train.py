from data.dataset import TrackingDataset
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from M_tools import calculate_metrics
from M_tools import M_vis_route, record_data_to_csv

import datetime
import os
now = datetime.datetime.now()
month = str(now.month)
day = str(now.day)
hour = str(now.hour)
minute = str(now.minute)
date = month + "-" + day + "-" + hour + ":" + minute
tmp_path = f"result/{date}"
if not os.path.exists(tmp_path):
    os.makedirs(tmp_path)

dataset = TrackingDataset(data_type="real-shot",dataset_root="dataset")

print(f"dataset len:", len(dataset))

# hyperparameters
batch_size = 3
epochs = 60
use_warmup = False
warmup_frames = 8
warmup_frames = 0 if not use_warmup else warmup_frames

##device
device = torch.device("cuda" if torch.cuda.is_available else "cpu")
# device = torch.device("cpu")


## data loader
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

## load model
from M_PACnet import Base_Net
model = Base_Net(use_warmup = use_warmup, warmup_frames = warmup_frames)
model.to(device)

## optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

## 训练指标和验证指标
train_samples = 0
train_rms = 0.0
train_pcm = 0.0
train_area = 0.0
train_dtw = 0.0

for epoch in tqdm(range(epochs)):

    ## 训练
    model.train()
    runnig_loss = 0.0
    epoch_samples=0
    for batch_idx, input in enumerate(train_loader):

        video = input[0].to(device).to(torch.float32)
        route = input[1].to(device).to(torch.float32)
        map_size = input[2].to(device)
        batch_size = video.shape[0]

        # 梯度清零
        optimizer.zero_grad()

        # forward
        predictions = model(video, route)

        # loss
        loss = model.compute_loss(predictions, route)

        loss.backward()
        optimizer.step()
        runnig_loss += loss.item()

        ## 计算指标
        rms, pcm, area, dtw = calculate_metrics(predictions, route)
        train_samples += len(route)
        epoch_samples += len(route)
        train_rms += rms
        train_area += area
        train_pcm += pcm
        train_dtw += dtw

        ## 可视化
        if batch_idx == 0:
            M_vis_route(route[0].cpu().numpy(), predictions[0].cpu().detach().numpy(),"train",epoch, date)

        epoch_loss = runnig_loss / epoch_samples
        print("******************Tain********************")
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
        # 记录指标
        avg_rms = train_rms / train_samples
        avg_area = train_area / train_samples
        avg_pcm = train_pcm / train_samples
        avg_dtw = train_dtw / train_samples
        record_data_to_csv(epoch, epoch_loss, avg_rms, avg_pcm, avg_area, avg_dtw, file_path=f"result/{date}/training_data.csv")
        print(f'Epoch {epoch + 1}/{epochs}, RMS: {avg_rms:.4f}, Area: {avg_area:.4f}, PCM: {avg_pcm:.4f}, DTW: {avg_dtw:.4f}')
        ## 保存静态参数
        torch.save(model.state_dict(), f"result/{date}/epoch{epoch}_model_state_dict.pth")



