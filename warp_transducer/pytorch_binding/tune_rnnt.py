import torch
import torch.nn.functional as F
from warprnnt_pytorch import DelayTLoss, RNNTLoss
import time
device=0
import os,psutil

process = psutil.Process(os.getpid())


def main(src_len=1000, tgt_len=100, voc=30000, bsz=1):
    
    cuda= True
    rnnt_loss=DelayTLoss(blank=0, delay_scale=1., reduction='mean')
    rnnt2= RNNTLoss()
    torch.manual_seed(20)
    label= (torch.rand(bsz,tgt_len)*voc).int().cuda(device)    
    acts= torch.randn(bsz,src_len,tgt_len+1, voc).cuda(device)
    label= label.clamp(0, voc-1)
    acts.requires_grad=True
    acts2= acts.detach()
    acts2.requires_grad= True
    #print(f"acts cost {acts.numel()*4} bytes")
    #print(torch.cuda.memory_summary(device=device))
    #print('Used Memory:',process.memory_info().rss / 1024 / 1024,'MB')
    start= time.time()
    #acts=F.log_softmax(acts, dim=-1)
    act_lengths= label.new(bsz).fill_(src_len)
    label_lengths= label.new(bsz).fill_(tgt_len)
    """ loss= rnnt2(acts, label,act_lengths, label_lengths)
    loss= loss*0.1
    loss.backward()
    print("loss={loss.item()}") """
    
    loss, loss_rnnt, loss_delay= rnnt_loss(acts2, label, act_lengths, label_lengths)
    print(f"loss={loss.item()}, loss_rnnt={loss_rnnt.item()}, loss_delay={loss_delay.item()}") 
    print(torch.cuda.memory_summary(device=device))
    loss= loss*0.1
    loss.backward()
    print(torch.cuda.memory_summary(device=device))
    #diff= acts.grad - acts2.grad
    
    #import pdb;pdb.set_trace()
    
    grad= acts2.grad
    isbad= torch.abs(grad)>10
    print(f" max={grad.max().item()}, min={grad.min().item()}, bad={isbad.sum().item()}")
    end=time.time()
    print(f"src_len={src_len}, tgt_len={tgt_len},voc={voc},bsz={bsz}, cost={end-start} secs")

if __name__ == "__main__":
    # (32, 70, 30000,1) 780M,0.027s
    #(32, 70, 30000,10) 7.8G 0.218s
    #(32, 70, 30000,5) 3.9G 0.093s
    #(32, 70, 30000,5)cpu segment fault
    #cli 5 80 70 30000 6.6G, 0.33s
    #cli 10,32,70,30000 5.3G 0.276s
    #cli 5 80 70 30000 cpu 5.9g mem, 2.38s
    #python 5 80 70 30000 cpu 13g mem, 3.17s, 去除logsoftmax 9g
    #(32, 70, 30000,5) 0.42s, 2.6G
    #(20, 70, 30000, 10) 0.07s, 3.3G
    #(20, 70, 30000, 10) release  0.087s, 3.3G
    # 看上去做文本任务必需拆输出,隐层不需要拆借,softmax投影需要拆
    #(20,70,30000,30)12g不足
    # should compact with split trick
    #[2, 32, 117, 256], slen [27,27], tlen [105,91]
    #[4, 25, 83, 256]
    main(40, 100, 3000,20) 
