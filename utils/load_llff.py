import numpy as np
import os, imageio
import utils.utils as utils

def _minify(basedir, subdir=None, factors=[], resolutions=[]):
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, 'images_{}'.format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
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
            
        print('Minifying', r, basedir+name)
        
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
 

def _load_data(basedir, poses, images, factor=None, width=None, height=None, load_imgs=True, subdir=None):
    fileposes = 'poses_bounds.npy' if subdir is None else f'poses_bounds_{subdir}.npy'
    
    poses_arr = np.load(os.path.join(basedir, fileposes))
    pos = poses_arr[:, :-2].reshape([-1, 3, 5])
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

    imgdir = os.path.join(basedir, subdir + sfx)
    
    if not os.path.exists(imgdir):
        print( imgdir, 'does not exist, returning' )
        return
    
    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    if poses.shape[-1] != len(imgfiles):
        print( 'Mismatch between imgs {} and poses {} !!!!'.format(len(imgfiles), poses.shape[-1]) )
        return
    
    # Focal length refactor
    hwf *= (1./factor)

    if not load_imgs:
        return hwf
    
    def imread(f):
        if f.endswith('png'):
            return imageio.imread(f, ignoregamma=True)
        else:
            return imageio.imread(f)
    
    imgs = [imread(f)[...,:3]/255. for f in imgfiles]
    images[:] = np.stack(imgs, -1)  
    
    print('Loaded image data', images.shape, poses[:,-1,0])
    print("_load data poses shape", poses.shape)
    return hwf, poses

def load_llff_data(basedir, poses, images, factor=8, subdir=None, i=0, i_n=1):
    images = np.moveaxis(images, 0, -1)
    hwf, poses = _load_data(basedir, poses, images, factor=factor, subdir=subdir) # factor=8 downsamples original imgs by 8x
    
    print("_load llf data poses shape", poses.shape)
    print('Loaded', basedir, subdir)
    
    # Correct rotation matrix ordering and move variable dim to axis 0
    poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    images = np.moveaxis(images, -1, 0).astype(np.float32)
    
    #images = images.astype(np.float32)
    #poses = poses.astype(np.float32)
    #hwf = poses[0, :3, -1]
    #poses = poses[:, :3, :4]

    return hwf, poses


