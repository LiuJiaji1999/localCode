import torch
import torch.nn.functional as F

def compute_mmd_loss(source_feat, target_feat, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    计算源域和目标域特定层特征的最大均值差异（MMD）。
    使用多个尺度的高斯核来衡量两个分布之间的差异。
    
    参数:
        source_feat (torch.Tensor): 源域特征，形状 (batch, channels, height, width)
        target_feat (torch.Tensor): 目标域特征，形状 (batch, channels, height, width)
        kernel_mul (float): 高斯核带宽因子，默认值为 2.0
        kernel_num (int): 使用的高斯核数量，默认值为 5
        fix_sigma (float): 如果不为 None，则固定使用该 sigma，否则根据数据自适应计算 sigma
        
    返回:
        float: MMD 损失值（标量）
    """
    # 获取 batch 数，并将特征展平为 (batch, -1)
    b, c, h, w = source_feat.shape
    source_flat = source_feat.view(b, -1)
    target_flat = target_feat.view(b, -1)
    
    def gaussian_kernel(x, y, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        """
        计算 x 和 y 之间所有样本两两之间的高斯核值，并求和返回核矩阵。
        """
        n = x.size(0)
        total = torch.cat([x, y], dim=0)  # shape: (2*n, features)
        total0 = total.unsqueeze(0).expand(total.size(0), total.size(0), total.size(1))
        total1 = total.unsqueeze(1).expand(total.size(0), total.size(0), total.size(1))
        # 计算两两欧氏距离平方
        L2_distance = ((total0 - total1) ** 2).sum(2)
        
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n**2 - n)
        
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)
    
    # 计算高斯核矩阵，核矩阵形状为 (2*b, 2*b)
    kernels = gaussian_kernel(source_flat, target_flat, kernel_mul, kernel_num, fix_sigma)
    print(kernels)
    # 分别取出源域、目标域和跨域部分
    XX = kernels[:b, :b]       # 源域内核
    YY = kernels[b:, b:]       # 目标域内核
    XY = kernels[:b, b:]       # 跨域核
    YX = kernels[b:, :b]       # 跨域核（与 XY 等价）
    
    print(XX)
    # 根据 MMD 定义计算均值差异
    loss = torch.mean(XX + YY - XY - YX)
    return loss.item()

# 示例：
if __name__ == "__main__":
    # 随机生成示例特征：batch=4, channels=64, height=16, width=16
    source_feat = torch.randn(4, 64, 16, 16)
    target_feat = torch.randn(4, 64, 16, 16)
    mmd = compute_mmd_loss(source_feat, target_feat)
    print("MMD Loss:", mmd)
