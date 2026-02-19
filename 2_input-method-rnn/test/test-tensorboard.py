from torch.utils.tensorboard import SummaryWriter

# 创建写入器，指定日志保存目录
writer = SummaryWriter(log_dir="./logs")
for step in range(100):
    writer.add_scalar("scaler/y=x", step, step)
    writer.add_scalar("scaler/y=x^2", step ** 2, step)

writer.close()
