import torch
import os, pathlib
# import hyperparams as hyp
import numpy as np

def save_ensemble(ckpt_dir, optimizer, models, models_ema, global_step, keep_latest=10):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    prev_ckpts = list(pathlib.Path(ckpt_dir).glob('model-*'))
    prev_ckpts.sort(key=lambda p: p.stat().st_mtime,reverse=True)
    if len(prev_ckpts) > keep_latest-1:
        for f in prev_ckpts[keep_latest-1:]:
            f.unlink()
    model_path = '%s/model-%09d.pth' % (ckpt_dir, global_step)
    
    ckpt = {'optimizer_state_dict': optimizer.state_dict()}
    for i in range(len(models)):
        ckpt['model_state_dict_{}'.format(i)] = models[i].state_dict()
        ckpt['ema_model_state_dict_{}'.format(i)] = models_ema[i].state_dict()
    torch.save(ckpt, model_path)
    print("saved a checkpoint: %s" % (model_path))

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

def load_ensemble(ckpt_dir, optimizer, models, models_ema):
    print('reading ckpt from %s' % ckpt_dir)
    checkpoint_dir = os.path.join('checkpoints/', ckpt_dir)
    step = 0
    if not os.path.exists(checkpoint_dir):
        print('...there is no full checkpoint here!')
    else:
        ckpt_names = os.listdir(checkpoint_dir)
        steps = [int((i.split('-')[1]).split('.')[0]) for i in ckpt_names]
        if len(ckpt_names) > 0:
            step = max(steps)
            model_name = 'model-%09d.pth' % (step)
            path = os.path.join(checkpoint_dir, model_name)
            print('...found checkpoint %s'%(path))

            checkpoint = torch.load(path)

            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            for i, (model, model_ema) in enumerate(zip(models, models_ema)):
                model.load_state_dict(checkpoint['model_state_dict_{}'.format(i)])
                model_ema.load_state_dict(checkpoint['ema_model_state_dict_{}'.format(i)])
        else:
            print('...there is no full checkpoint here!')
    return step


def load(ckpt_dir, model, optimizer=None, scheduler=None, model_ema=None, step=0, model_name='model', ignore_load=None):
    print('reading ckpt from %s' % ckpt_dir)
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

                checkpoint = torch.load(path)['model_state_dict']
                # model.load_state_dict(checkpoint['model_state_dict'], strict=False)

                # model_state = model.state_dict()
                # for name, param in checkpoint.items():
                #     if name not in model_state:
                #         continue
                #     if isinstance(param, torch.nn.Parameter):
                #         # backwards compatibility for serialized parameters
                #         param = param.data
                #         model_state[name].copy_(param)
                # model.load_state_dict(model_state)

                model_dict = model.state_dict()

                # 1. filter out ignored keys
                pretrained_dict = {k: v for k, v in checkpoint.items()}
                for ign in ignore_load:
                    pretrained_dict = {k: v for k, v in pretrained_dict.items() if not ign in k}
                    
                # pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
                # 2. overwrite entries in the existing state dict
                model_dict.update(pretrained_dict)
                # 3. load the new state dict
                model.load_state_dict(model_dict, strict=False)
                                                                                                             
                
                # checkpoint = torch.load(path)['model_state_dict']
                # # model.load_state_dict(dict([(n, p) for n, p in checkpoint['model_state_dict'].items()]), strict=False)

                # # # keys_vin=torch.load('',map_location=device)
                
                # # new_state_dict = {k:v if v.size()==current_model[k].size()  else  current_model[k] for k,v in zip(current_model.keys(), checkpoint['model_state_dict'].values())}
                # # # model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                # # model.load_state_dict(new_state_dict, strict=False)

                # current_model_dict = model.state_dict()

                # for k,v in zip(current_model_dict.keys(), checkpoint.values()):
                #     # print('k', k)
                #     if not (v.size()==current_model_dict[k].size()):
                #         print('shape mismatch for', k, v.size(), current_model_dict[k].size())
                #     # print('v', v)
                #     # v.size()
                #     # k:v if v.size()==current_model_dict[k].size()  else  current_model_dict[k] 
                
                # new_state_dict={k:v if v.size()==current_model_dict[k].size()  else  current_model_dict[k] for k,v in zip(current_model_dict.keys(), checkpoint.values())}
                # model.load_state_dict(new_state_dict, strict=False)

                # # def get_attr(obj, names):
                #     if len(names) == 1:
                #         getattr(obj, names[0])
                #     else:
                #         get_attr(getattr(obj, names[0]), names[1:])
                        
                # def set_attr(obj, names, val):
                #     if len(names) == 1:
                #         setattr(obj, names[0], val)
                #     else:
                #         set_attr(getattr(obj, names[0]), names[1:], val)
                        
                # for key, dict_param in checkpoint.items():
                #     submod_names = key.split(".")
                #     curr_param = get_attr(mod, submod_names)
                #     new_param = your_processing(curr_param, dict_param)
                #     # Here you can either replace the existing one
                #     set_attr(mod, subdmod_names, new_param)
                #     # Or re-use it (as done in load_checkpoint) but the sizes have to match!
                #     with torch.no_grad():
                #         curr_param.copy_(new_param)

            else:
                checkpoint = torch.load(path)
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
