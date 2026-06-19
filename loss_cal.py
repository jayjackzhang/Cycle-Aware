



m=15
pred_embed = pcls[:, :m].float()  # (batch, 15)
class_num = (self.codeValueList[tcls[i] - 1]).long() # 2190编号转到2^15编号,codeValueList与生成数据的matlab代码中的变量一致
n = class_num.shape[0]
# 转成15位二进制
bits = ((class_num.unsqueeze(1) >> torch.arange(15 - 1, -1, -1,device=class_num.device)) & 1).float()  # (n,15)
# 生成所有循环位移
index = (torch.arange(m, device=bits.device).unsqueeze(0)+ torch.arange(m, device=bits.device).unsqueeze(1)) % m
target_result = (bits.unsqueeze(1).expand(-1, m, -1).gather(2, index.unsqueeze(0).expand(n, -1, -1)).reshape(n * m, m))
predict_result = (pred_embed.unsqueeze(1).expand(n, m, m).reshape(n * m, m))
# BCE loss
losses = self.cross(predict_result,target_result)  # (n*m,15) self.cross是nn.BCEWithLogitsLoss函数
losses_sum = losses.sum(dim=1)
# 每个样本15个位移中取最小
losses_reshape = losses_sum.view(n, m)
min_losses, min_idx = torch.min(losses_reshape,dim=1)
loss_embed = min_losses.mean()
lcls += loss_embed