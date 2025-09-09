import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from einops import rearrange

class TimeAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        
    def forward(self, x):
        # x: (B,T,D)
        q = self.query(x[:, 0])  # 用初始帧生成query
        k = self.key(x)          # (B,T,D)
        
        attn = torch.einsum('bd,btd->bt', q, k)  # (B,T)
        attn = torch.softmax(attn, dim=1)
        return attn.unsqueeze(-1) * x  # 加权特征

class PAC_Net(nn.Module):
    def __init__(self, pretrained=True, rnn_hdim=128, use_warmup=False, warmup_frames=32):
        super().__init__()
        self.rnn_hdim = rnn_hdim
        self.use_warmup = use_warmup
        self.warmup_frames = warmup_frames if use_warmup else 0

        # 主干网络初始化
        backbone_weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.c_encoder = self._build_backbone(resnet18, backbone_weights, rnn_hdim)
        self.p_encoder = self._build_backbone(resnet18, backbone_weights, rnn_hdim)

        # self.new_encoder = self._build_backbone(resnet18, backbone_weights, rnn_hdim)

        # RNN单元
        self.c_cell = nn.GRUCell(rnn_hdim, rnn_hdim)
        self.p_cell = nn.GRUCell(rnn_hdim, rnn_hdim)

        # self.new_cell = nn.GRUCell(rnn_hdim, rnn_hdim)

        # 预测头
        self.decoder = nn.Sequential(
            nn.Linear(rnn_hdim, rnn_hdim//2),
            nn.ReLU(),
            nn.Linear(rnn_hdim//2, 2),
            # nn.Tanh()
            nn.Sigmoid()
        )
        # self.ta = TimeAttention(rnn_hdim)

        # 初始化warmup模块
        if self.use_warmup:
            self.warmup_c_encoder = self._build_backbone(resnet18, backbone_weights, rnn_hdim)
            self.warmup_p_encoder = self._build_backbone(resnet18, backbone_weights, rnn_hdim)
            # self.warmup_c_encoder = self.c_encoder
            # self.warmup_p_encoder = self.p_encoder
            self.warmup_c_cell = nn.GRUCell(rnn_hdim, rnn_hdim)
            self.warmup_p_cell = nn.GRUCell(rnn_hdim, rnn_hdim)

        self._init_weights()

    def forward(self, I, x_gt):
        # I: (B, C, T, H, W), x_gt: (B, T, 2)
        B, T_total = x_gt.shape[:2]
        delta_I = torch.diff(I, dim=2)  # (B, C, T-1, H, W)

        # dd_T = torch.diff(delta_I, dim=2)  # (B, C, T-2, H, W)

        if self.use_warmup and self.warmup_frames >0:
            warm_up_I, I = I[:,:,:self.warmup_frames], I[:,:,self.warmup_frames:]
            warm_up_delta_I, delta_I = delta_I[:,:,:self.warmup_frames], delta_I[:,:,self.warmup_frames:]
            # hv = self._handle_warmup(warm_up_I, warm_up_delta_I, B, self.warmup_frames)
            hv =  self._warm_up(warm_up_I, warm_up_delta_I)
            T = I.size(2)
        else:
            T = I.size(2)
            hv = None 
        

        fx = self._extract_features(self.c_encoder, I, B)    # (T, B, D)
        fv = self._extract_features(self.p_encoder, delta_I, B)

        # fnew = self._extract_features(self.new_encoder, dd_T, B)    # (T, B, D)
        
        # T = T_total - self.warmup_frames
        # 时序处理
        Hx = torch.zeros(B, T, self.rnn_hdim, device=I.device)
        for t in range(T):
            hv = self.c_cell(fx[t], hv)  # Calibrate
            Hx[:, t] = hv
            # if t < T-2:
                # hv = self.new_cell(fnew[t], hv)  # new
            if t < T-1:
                hv = self.p_cell(fv[t], hv)  # Propagate

        # Hx = self.ta(Hx)

        # 预测结果
        x_pred = self.decoder(Hx)
        return x_pred


    def _warm_up(self, I, delta_I):
        B, T = I.size(0), I.size(2)
        fx = self.warmup_c_encoder(rearrange(I.float(), 'b c t h w -> (b t) c h w'))
        fx = rearrange(fx, '(b t) d -> t b d', b=B)

        fv = self.warmup_p_encoder(rearrange(delta_I, 'b c t h w -> (b t) c h w'))
        fv = rearrange(fv, '(b t) d -> t b d', b=B)

        hv_t = torch.zeros(B, self.rnn_hdim, device=I.device)
        for t in range(T):
            hx_t = self.warmup_c_cell(input=fx[t], hx=hv_t)
            hv_t = self.warmup_p_cell(input=fv[t], hx=hx_t)

        return hv_t

    def _extract_features(self, encoder, x, batch_size):
        """特征提取与重组"""
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        features = encoder(x)
        return rearrange(features, '(b t) d -> t b d', b=batch_size)

    def _build_backbone(self, model_fn, weights, output_dim):
        """构建并修改ResNet主干网络"""
        model = model_fn(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, output_dim)
        return model

    def _init_weights(self):
        """初始化全连接层参数"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def compute_loss(self, pred, gt, v_loss_weight=0.1):
        """计算联合损失"""

        # 位置损失
        pos_loss = nn.MSELoss()(pred, gt[:, self.warmup_frames:])
        
        # 速度损失
        v_pred = torch.diff(pred, dim=1)
        v_gt = torch.diff(gt[:, self.warmup_frames:], dim=1)
        vel_loss = nn.MSELoss()(v_pred, v_gt)
        
        return pos_loss + v_loss_weight * vel_loss


class P_Net(nn.Module):
    def __init__(self, pretrained=True, rnn_hdim=128, use_warmup=False, warmup_frames=32):
        super().__init__()
        self.rnn_hdim = rnn_hdim
        self.use_warmup = use_warmup
        self.warmup_frames = warmup_frames if use_warmup else 0
        # 主干网络初始化
        backbone_weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.p_encoder = self._build_backbone(resnet18, backbone_weights, rnn_hdim)
        # RNN单元
        self.p_cell = nn.GRUCell(rnn_hdim, rnn_hdim)
        # 预测头
        self.decoder = nn.Sequential(
            nn.Linear(rnn_hdim, rnn_hdim//2),
            nn.ReLU(),
            nn.Linear(rnn_hdim//2, 2),
            # nn.Tanh()
            nn.Sigmoid()
        )
        # 初始化warmup模块
        if self.use_warmup:
            self.warmup_p_encoder = self._build_backbone(resnet18, backbone_weights, rnn_hdim)
            self.warmup_p_cell = nn.GRUCell(rnn_hdim, rnn_hdim)
        self._init_weights()

    def forward(self, I, x_gt):
        # I: (B, C, T, H, W), x_gt: (B, T, 2)
        B, T_total = x_gt.shape[:2]
        delta_I = torch.diff(I, dim=2)  # (B, C, T-1, H, W) 
        if self.use_warmup and self.warmup_frames >0:
            warm_up_I, I = I[:,:,:self.warmup_frames], I[:,:,self.warmup_frames:]
            warm_up_delta_I, delta_I = delta_I[:,:,:self.warmup_frames], delta_I[:,:,self.warmup_frames:]
            hv =  self._warm_up(warm_up_I, warm_up_delta_I)
            T = I.size(2)
        else:
            T = I.size(2)
            hv = None

        fv = self._extract_features(self.p_encoder, delta_I, B)    # (T, B, D)
        # 时序处理
        Hx = torch.zeros(B, T, self.rnn_hdim, device=I.device)
        for t in range(T):
            hv = self.p_cell(fv[t], hv)  # Propagate
            Hx[:, t] = hv
        # 预测结果
        x_pred = self.decoder(Hx)
        return x_pred
    def _warm_up(self, I, delta_I):
        B, T = I.size(0), I.size(2)
        fx = self.warmup_p_encoder(rearrange(I.float(), 'b c t h w -> (b t) c h w'))
        fx = rearrange(fx, '(b t) d -> t b d', b=B)
        fv = self.warmup_p_encoder(rearrange(delta_I, 'b c t h w -> (b t) c h w'))
        fv = rearrange(fv, '(b t) d -> t b d', b=B)
        hv_t = torch.zeros(B, self.rnn_hdim, device=I.device)
        for t in range(T):
            hv_t = self.warmup_p_cell(input=fv[t], hx=hv_t)
        return hv_t
    def _extract_features(self, encoder, x, batch_size):
        """特征提取与重组"""
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        features = encoder(x)
        return rearrange(features, '(b t) d -> t b d', b=batch_size)
    def _build_backbone(self, model_fn, weights, output_dim):
        """构建并修改ResNet主干网络"""
        model = model_fn(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, output_dim)
        return model
    def _init_weights(self):
        """初始化全连接层参数"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    def compute_loss(self, pred, gt, v_loss_weight=0.1):
        """计算联合损失"""
        # 位置损失
        pos_loss = nn.MSELoss()(pred, gt[:, self.warmup_frames:])
        # 速度损失
        v_pred = torch.diff(pred, dim=1)
        v_gt = torch.diff(gt[:, self.warmup_frames:], dim=1)
        vel_loss = nn.MSELoss()(v_pred, v_gt)
        return pos_loss + v_loss_weight * vel_loss

class C_Net(nn.Module):
    def __init__(self, pretrained=True, rnn_hdim=128, use_warmup=False, warmup_frames=32):
        super().__init__()
        self.rnn_hdim = rnn_hdim
        self.use_warmup = use_warmup
        self.warmup_frames = warmup_frames if use_warmup else 0
        # 主干网络初始化
        backbone_weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.c_encoder = self._build_backbone(resnet18, backbone_weights, rnn_hdim)
        # RNN单元
        self.c_cell = nn.GRUCell(rnn_hdim, rnn_hdim)
        # 预测头
        self.decoder = nn.Sequential(
            nn.Linear(rnn_hdim, rnn_hdim//2),
            nn.ReLU(),
            nn.Linear(rnn_hdim//2, 2),
            # nn.Tanh()
            nn.Sigmoid()
        )
        # 初始化warmup模块
        if self.use_warmup:
            self.warmup_c_encoder = self._build_backbone(resnet18, backbone_weights, rnn_hdim)
            self.warmup_c_cell = nn.GRUCell(rnn_hdim, rnn_hdim)
        self._init_weights()

    def forward(self, I, x_gt):
        # I: (B, C, T, H, W), x_gt: (B, T, 2)
        B, T_total = x_gt.shape[:2]
        if self.use_warmup and self.warmup_frames >0:
            warm_up_I, I = I[:,:,:self.warmup_frames], I[:,:,self.warmup_frames:]
            hv =  self._warm_up(warm_up_I)
            T = I.size(2)
        else:
            T = I.size(2)
            hv = None

        fx = self._extract_features(self.c_encoder, I, B)    # (T, B, D)
        # 时序处理
        Hx = torch.zeros(B, T, self.rnn_hdim, device=I.device)
        for t in range(T):
            hv = self.c_cell(fx[t], hv)  # Calibrate
            Hx[:, t] = hv
        # 预测结果
        x_pred = self.decoder(Hx)
        return x_pred
    def _warm_up(self, I):
        B, T = I.size(0), I.size(2)
        fx = self.warmup_c_encoder(rearrange(I.float(), 'b c t h w -> (b t) c h w'))
        fx = rearrange(fx, '(b t) d -> t b d', b=B)
        hv_t = torch.zeros(B, self.rnn_hdim, device=I.device)
        for t in range(T):
            hv_t = self.warmup_c_cell(input=fx[t], hx=hv_t)
        return hv_t
    def _extract_features(self, encoder, x, batch_size):
        """特征提取与重组"""
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        features = encoder(x)
        return rearrange(features, '(b t) d -> t b d', b=batch_size)
    def _build_backbone(self, model_fn, weights, output_dim):
        """构建并修改ResNet主干网络"""
        model = model_fn(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, output_dim)
        return model
    def _init_weights(self):
        """初始化全连接层参数"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    def compute_loss(self, pred, gt, v_loss_weight=0.1):
        """计算联合损失"""
        # 位置损失
        pos_loss = nn.MSELoss()(pred, gt[:, self.warmup_frames:])
        # 速度损失
        v_pred = torch.diff(pred, dim=1)
        v_gt = torch.diff(gt[:, self.warmup_frames:], dim=1)
        vel_loss = nn.MSELoss()(v_pred, v_gt)
        return pos_loss + v_loss_weight * vel_loss

class Base_Net(nn.Module):
    def __init__(self, pretrained=True, rnn_hdim=128, use_warmup=False, warmup_frames=32):
        super().__init__()
        self.rnn_hdim = rnn_hdim
        self.use_warmup = use_warmup
        self.warmup_frames = warmup_frames if use_warmup else 0
        # 主干网络初始化
        backbone_weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.base_encoder = self._build_backbone(resnet18, backbone_weights, rnn_hdim)
        # RNN单元
        self.base_cell1 = nn.GRUCell(rnn_hdim, rnn_hdim)
        self.base_cell2 = nn.GRUCell(rnn_hdim, rnn_hdim)
        # 预测头
        self.decoder = nn.Sequential(
            nn.Linear(rnn_hdim, rnn_hdim//2),
            nn.ReLU(),
            nn.Linear(rnn_hdim//2, 2),
            nn.Sigmoid()
        )
        # 初始化warmup模块
        if self.use_warmup:
            self.warmup_base_encoder = self._build_backbone(resnet18, backbone_weights, rnn_hdim)
            self.warmup_base_cell = nn.GRUCell(rnn_hdim, rnn_hdim)
        self._init_weights()

    def forward(self, I, x_gt):
        # I: (B, C, T, H, W), x_gt: (B, T, 2)
        B, T_total = x_gt.shape[:2]
        if self.use_warmup and self.warmup_frames >0:
            warm_up_I, I = I[:,:,:self.warmup_frames], I[:,:,self.warmup_frames:]
            hv =  self._warm_up(warm_up_I)
            T = I.size(2)
        else:
            T = I.size(2)
            hv = None

        fx = self._extract_features(self.base_encoder, I, B)    # (T, B, D)
        # 时序处理
        Hx = torch.zeros(B, T, self.rnn_hdim, device=I.device)
        for t in range(T):
            hv_temp = self.base_cell1(fx[t], hv)  # Calibrate
            hv = self.base_cell2(hv_temp, hv)
            Hx[:, t] = hv
        # 预测结果
        x_pred = self.decoder(Hx)
        return x_pred
    def _warm_up(self, I):
        B, T = I.size(0), I.size(2)
        fx = self.warmup_base_encoder(rearrange(I.float(), 'b c t h w -> (b t) c h w'))
        fx = rearrange(fx, '(b t) d -> t b d', b=B)
        hv_t = torch.zeros(B, self.rnn_hdim, device=I.device)
        for t in range(T):
            hv_t = self.warmup_base_cell(input=fx[t], hx=hv_t)
        return hv_t
    def _extract_features(self, encoder, x, batch_size):
        """特征提取与重组"""
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        features = encoder(x)
        return rearrange(features, '(b t) d -> t b d', b=batch_size)
    def _build_backbone(self, model_fn, weights, output_dim):
        """构建并修改ResNet主干网络"""
        model = model_fn(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, output_dim)
        return model
    def _init_weights(self):
        """初始化全连接层参数"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    def compute_loss(self, pred, gt, v_loss_weight=0.1):
        """计算联合损失"""
        # 位置损失
        pos_loss = nn.MSELoss()(pred, gt[:, self.warmup_frames:])
        # 速度损失
        v_pred = torch.diff(pred, dim=1)
        v_gt = torch.diff(gt[:, self.warmup_frames:], dim=1)
        vel_loss = nn.MSELoss()(v_pred, v_gt)
        return pos_loss + v_loss_weight * vel_loss
