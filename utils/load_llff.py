import numpy as np
import os, imageio
import torch

def _minify(basedir, subdir=None, factors=[], resolutions=[]):
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, f'{subdir}_{r}')
        if not os.path.exists(imgdir):
            print("needtoload")
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, f'{subdir}_{r[1]}x{r[0]}')
        if not os.path.exists(imgdir):
            print("needtoload")
            needtoload = True
    if not needtoload:
        return
    
    from shutil import copy
    from subprocess import check_output
    
    imgdir = basedir if subdir is None else os.path.join(basedir, subdir)
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
    imgdir_orig = imgdir
    
    wd = os.getcwd()
    if subdir is None:
        subdir = 'images'

    for r in factors + resolutions:
        if isinstance(r, int):
            name = '{}_{}'.format(subdir, r)
            resizearg = '{}%'.format(100./r)
        else:
            name = '{}_{}x{}'.format(subdir, r[1], r[0])
            resizearg = '{}x{}'.format(r[1], r[0])
        imgdir = os.path.join(basedir, name)
        if os.path.exists(imgdir):
            continue
            
        print('Minifying', r, imgdir)
        
        os.makedirs(imgdir)
        check_output('cp {}/* {}'.format(imgdir_orig, imgdir), shell=True)
        
        ext = imgs[0].split('.')[-1]
        args = ' '.join(['mogrify', '-resize', resizearg, '-format', 'png', '*.{}'.format(ext)])
        print(args)
        os.chdir(imgdir)
        check_output(args, shell=True)
        os.chdir(wd)
        
        if ext != 'png':
            check_output('rm {}/*.{}'.format(imgdir, ext), shell=True)
            print('Removed duplicates')
        print('Done')
 
def _load_data(basedir, poses, images, factor=None, width=None, height=None, load_imgs=True, subdir=None, n_imgs=-1):
    fileposes = 'poses_bounds.npy' if subdir is None else f'poses_bounds_{subdir}.npy'
    print(f"Fileposes:  {basedir}/{fileposes}")
    poses_arr = np.load(os.path.join(basedir, fileposes))
    
    if n_imgs == -1:
        n_imgs = poses_arr.shape[0]
    elif n_imgs == 0:
        return torch.zeros((3,)), torch.zeros(3, 4, 0)
    
    pos = poses_arr[:n_imgs, :-2].reshape([-1, 3, 5])
    hwf = pos[0, :3, -1]
    poses[:] = pos[:, :3, :4]
    poses = poses.transpose([1,2,0])
    
    img0 = [os.path.join(basedir, subdir, f) for f in sorted(os.listdir(os.path.join(basedir, subdir))) \
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]
    
    sh = imageio.imread(img0).shape
    
    sfx = ''
    
    if factor is not None:
        sfx = '_{}'.format(factor)
        _minify(basedir, subdir=subdir, factors=[factor])
        factor = factor
    elif height is not None:
        factor = sh[0] / float(height)
        width = int(sh[1] / factor)
        _minify(basedir, subdir=subdir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    elif width is not None:
        factor = sh[1] / float(width)
        height = int(sh[0] / factor)
        _minify(basedir, subdir=subdir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    else:
        factor = 1
    
    if subdir is None:
        subdir = 'images'
    print(f"\tfrom {subdir + sfx}")
    imgdir = os.path.join(basedir, subdir + sfx)
    
    if not os.path.exists(imgdir):
        print( imgdir, 'does not exist, returning' )
        return hwf, torch.zeros(3, 4, 0)
    
    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][:n_imgs] #changed to[-n_imgs:] from [:n_imgs]
    print("img files", imgfiles)
    if poses.shape[-1] != len(imgfiles):
        print( 'Mismatch between imgs {} and poses {} !!!!'.format(len(imgfiles), poses.shape[-1]) )
        return hwf, torch.zeros(3, 4, 0)
    
    # Focal length refactor
    hwf *= (1./factor)

    if not load_imgs:
        return hwf, torch.zeros(3, 4, 0)
    
    def imread(f):
        if f.endswith('png'):
            return imageio.imread(f, ignoregamma=True)
        else:
            return imageio.imread(f)
    
    imgs = [imread(f)[...,:3]/255. for f in imgfiles]
    images[:] = np.stack(imgs, -1)  
    
    print('\tLoaded image data', images.shape, hwf)
    return hwf, poses

def load_llff_data(basedir, poses, images, factor=8, subdir=None, i=0, i_n=1, n_imgs=-1):
    images = np.moveaxis(images, 0, -1)
    hwf, poses = _load_data(basedir, poses, images, factor=factor, subdir=subdir, n_imgs=n_imgs) # factor=8 downsamples original imgs by 8x
    
    print('\tLoaded', basedir, subdir)
    
    # Correct rotation matrix ordering and move variable dim to axis 0
    poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    images = np.moveaxis(images, -1, 0).astype(np.float32)
    
    #images = images.astype(np.float32)
    #poses = poses.astype(np.float32)
    #hwf = poses[0, :3, -1]
    #poses = poses[:, :3, :4]

    return hwf, poses


