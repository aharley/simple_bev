import torch
import os, pathlib
import numpy as np

def save(ckpt_dir, optimizer, model, global_step, scheduler=None, model_ema=None, keep_latest=5, model_name='model'):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    prev_ckpts = list(pathlib.Path(ckpt_dir).glob('%s-*' % model_name))
    prev_ckpts.sort(key=lambda p: p.stat().st_mtime,reverse=True)
    if len(prev_ckpts) > keep_latest-1:
        for f in prev_ckpts[keep_latest-1:]:
            f.unlink()
    model_path = '%s/%s-%09d.pth' % (ckpt_dir, model_name, global_step)
    
    ckpt = {'optimizer_state_dict': optimizer.state_dict()}
    ckpt['model_state_dict'] = model.state_dict()
    if scheduler is not None:
        ckpt['scheduler_state_dict'] = scheduler.state_dict()
    if model_ema is not None:
        ckpt['ema_model_state_dict'] = model_ema.state_dict()
    torch.save(ckpt, model_path)
    print("saved a checkpoint: %s" % (model_path))


def load(ckpt_dir, model, optimizer=None, scheduler=None, model_ema=None, step=0, model_name='model', ignore_load=None, device_ids=[0]):
    print('reading ckpt from %s' % ckpt_dir)
    device = 'cuda:%d' % device_ids[0]
    
    if not os.path.exists(ckpt_dir):
        print('...there is no full checkpoint here!')
        print('-- note this function no longer appends "saved_checkpoints/" before the ckpt_dir --')
    else:
        ckpt_names = os.listdir(ckpt_dir)
        steps = [int((i.split('-')[1]).split('.')[0]) for i in ckpt_names]
        if len(ckpt_names) > 0:
            if step==0:
                step = max(steps)
            model_name = '%s-%09d.pth' % (model_name, step)
            path = os.path.join(ckpt_dir, model_name)
            print('...found checkpoint %s'%(path))

            if ignore_load is not None:
                
                print('ignoring', ignore_load)

                checkpoint = torch.load(path, map_location=device)['model_state_dict']

                model_dict = model.state_dict()

                # 1. filter out ignored keys
                pretrained_dict = {k: v for k, v in checkpoint.items()}
                for ign in ignore_load:
                    pretrained_dict = {k: v for k, v in pretrained_dict.items() if not ign in k}
                    
                # 2. overwrite entries in the existing state dict
                model_dict.update(pretrained_dict)
                # 3. load the new state dict
                model.load_state_dict(model_dict, strict=False)
                                                                                                             
            else:
                checkpoint = torch.load(path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if scheduler is not None:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if model_ema is not None:
                model_ema.load_state_dict(checkpoint['ema_model_state_dict']) 
        else:
            print('...there is no full checkpoint here!')
    return step
