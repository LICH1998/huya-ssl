from tensorboardX import SummaryWriter
##writer 是写入器
writer = SummaryWriter("log")
for i in range(100):
    #global_step 深度学习可以看作迭代次数
    writer.add_scalar("a", i, global_step = i)
    writer.add_scalar("b", i ** 2, global_step = i)
writer.close()

##之后进入log文件夹  用tensorboard --logdir ./ 命令可以查看可视化图形