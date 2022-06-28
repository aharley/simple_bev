import numpy as np

def merge_rt(r, t):
    # r is 3 x 3
    # t is 3 or maybe 3 x 1
    t = np.reshape(t, [3, 1])
    rt = np.concatenate((r,t), axis=1)
    # rt is 3 x 4
    br = np.reshape(np.array([0,0,0,1], np.float32), [1, 4])
    # br is 1 x 4
    rt = np.concatenate((rt, br), axis=0)
    # rt is 4 x 4
    return rt

def merge_lrt(l, rt):
    # l is 3
    # rt is 4 x 4
    # merges these into a 19 vector
    D = len(l)
    assert(D==3)
    E, F = list(rt.shape)
    assert(E==4 and F==4)
    rt = rt.reshape(16)
    lrt = np.concatenate([l, rt], axis=0)
    return lrt
