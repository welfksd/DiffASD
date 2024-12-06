import torch
import os
from unet import *


from dataset import DCASE2022Dataset
from test_gmmanno import cal_metric, build_model
from loss import get_loss_usingtime
from tqdm import tqdm




from torch.optim.lr_scheduler import _LRScheduler

class GradualWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, multiplier, warm_epoch, after_scheduler = None, last_epoch = None):
        self.multiplier = multiplier
        self.total_epoch = warm_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        self.last_epoch = last_epoch
        self.base_lrs = None
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
    def state_dict(self):
        warmdict = {key:value for key, value in self.__dict__.items() if (key != 'optimizer' and key != 'after_scheduler')}
        cosdict = {key:value for key, value in self.after_scheduler.__dict__.items() if key != 'optimizer'}
        return {'warmup':warmdict, 'afterscheduler':cosdict}
    def load_state_dict(self, state_dict: dict):
        self.after_scheduler.__dict__.update(state_dict['afterscheduler'])
        self.__dict__.update(state_dict['warmup'])

    def step(self, epoch=None, metrics=None):
        if self.finished and self.after_scheduler:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.total_epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)


def train(config):
    start_epoch = -1
    model = build_model(config)
    print("Num params: ", sum(p.numel() for p in model.parameters()))
    model = torch.nn.DataParallel(model)  
    ema = copy.deepcopy(model)
    model.to(config.model.device)
    ema.to(config.model.device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.model.learning_rate, weight_decay=1e-4
    )
    
    cosineScheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer = optimizer,
                    T_max = config.model.epochs,
                    eta_min = 0,
                    last_epoch = -1
                )
    warmupScheduler = GradualWarmupScheduler(
                    optimizer = optimizer,
                    multiplier = 2.5,
                    warm_epoch = config.model.epochs // 10,
                    after_scheduler = cosineScheduler,
                    last_epoch = 0
                )
    if config.model.resume:
        checkpoint = torch.load(os.path.join(os.getcwd(), config.model.checkpoint_dir, str(config.model.load_chp)))
        start_epoch = checkpoint['epoch']
        
        model.load_state_dict(checkpoint['model'])
        ema.load_state_dict(checkpoint['ema'])
        model.to(config.model.device)
        ema.to(config.model.device)
    
        optimizer.load_state_dict(checkpoint['optimizer'])
        
        warmupScheduler.load_state_dict(checkpoint['lr_scheduler'])
        
    model.train()
    
    

    if not os.path.exists('checkpoints'):
        os.mkdir('checkpoints')
    if not os.path.exists(config.model.checkpoint_dir):
        os.mkdir(config.model.checkpoint_dir)

    ckpt_save_dir = os.path.join(os.getcwd(), config.model.checkpoint_dir)
    train_dataset = DCASE2022Dataset(config.train_dirs, ckpt_save_dir, is_train=True, need_convert_mel=False)
    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.model.num_workers,
        drop_last=True,
        pin_memory=True,
    )
    
    print("begin training")
    max_metric = 0.65
    for epoch in range(start_epoch + 1, config.model.epochs):  
        mean_loss = []
        # for batch in tqdm(trainloader, desc=f'Epoch {epoch}'):  
        for batch in trainloader:
            optimizer.zero_grad()
            t = torch.randint(0, config.model.time_steps, (batch[0].shape[0],), device=config.model.device).long()  # 先获取一个步数
            class_condition = batch[1].to(config.model.device)
            loss = get_loss_usingtime(model, batch[0], t, class_condition, config)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            
            update_ema_params(ema, model)
            mean_loss.append(loss.data.cpu())
                
        mean_loss = np.array(mean_loss).mean()

        print(f"Epoch {epoch} | Loss: {mean_loss.item()}")
        warmupScheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Learning Rate: {current_lr}')

        if epoch % config.model.save_model_epoch == 0:
            ckpt = {
                'model': model.state_dict(),
                'ema': ema.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': warmupScheduler.state_dict(),
                'epoch': epoch
            }
            torch.save(ckpt, os.path.join(ckpt_save_dir, str(epoch)))
            metric_all = cal_metric(model, config, ckpt_save_dir)

    print("end training")