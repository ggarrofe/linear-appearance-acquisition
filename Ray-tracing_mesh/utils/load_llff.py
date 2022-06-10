import numpy as np
import os, imageio


########## Slightly modified version of LLFF data loading code 
##########  see https://github.com/Fyusion/LLFF for original

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
            
        
'''def _load_data(basedir, factor=None, width=None, height=None, load_imgs=True, filename='poses_bounds.npy'):
    
    poses_arr = np.load(os.path.join(basedir, filename))
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1,2,0])
    bds = poses_arr[:, -2:].transpose([1,0])
    
    img0 = [os.path.join(basedir, 'images', f) for f in sorted(os.listdir(os.path.join(basedir, 'images'))) \
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]
    sh = imageio.imread(img0).shape
    
    sfx = ''
    
    if factor is not None:
        sfx = '_{}'.format(factor)
        _minify(basedir, factors=[factor])
        factor = factor
    elif height is not None:
        factor = sh[0] / float(height)
        width = int(sh[1] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    elif width is not None:
        factor = sh[1] / float(width)
        height = int(sh[0] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    else:
        factor = 1
    
    imgdir = os.path.join(basedir, 'images' + sfx)
    if not os.path.exists(imgdir):
        print( imgdir, 'does not exist, returning' )
        return
    
    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    if poses.shape[-1] != len(imgfiles):
        print( 'Mismatch between imgs {} and poses {} !!!!'.format(len(imgfiles), poses.shape[-1]) )
        return
    
    # Focal length refactor
    sh = imageio.imread(imgfiles[0]).shape
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
    poses[2, 4, :] = poses[2, 4, :] * 1./factor

    if not load_imgs:
        return poses, bds
    
    def imread(f):
        if f.endswith('png'):
            return imageio.imread(f, ignoregamma=True)
        else:
            return imageio.imread(f)
        
    imgs = imgs = [imread(f)[...,:3]/255. for f in imgfiles]
    imgs = np.stack(imgs, -1)  
    
    print('Loaded image data', imgs.shape, poses[:,-1,0])
    return poses, bds, imgs'''

def normalize(x):
    return x / np.linalg.norm(x)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def ptstocam(pts, c2w):
    tt = np.matmul(c2w[:3,:3].T, (pts-c2w[:3,3])[...,np.newaxis])[...,0]
    return tt

def poses_avg(poses):

    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)
    
    return c2w


    

# def load_llff_data(basedir, factor=8, recenter=True, bd_factor=.75, spherify=False, path_zflat=False, test_path=None):
#     poses, bds, imgs = _load_data(basedir, factor=factor) # factor=8 downsamples original imgs by 8x
#     print('Loaded', basedir, bds.min(), bds.max())
# 
#     if test_path is not None:
#         custom_pose_nerf, _, _ = _load_data(test_path, factor=factor, filename="custom_poses_llff.npy")
#         poses = np.append(poses, custom_pose_nerf, axis=2)
#     
#     # Correct rotation matrix ordering and move variable dim to axis 0
#     poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
#     poses = np.moveaxis(poses, -1, 0).astype(np.float32)
#     imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)
#     images = imgs
#     bds = np.moveaxis(bds, -1, 0).astype(np.float32)
# 
#     custom_pose_nerf = poses[-1, ...]
#     poses = poses[:100, ...]
#     
#     # Rescale if bd_factor is provided
#     sc = 1. if bd_factor is None else 1./(bds.min() * bd_factor)
#     poses[:,:3,3] *= sc
#     bds *= sc
#     
#     '''if recenter:
#         poses = recenter_poses(poses)'''
#         
#     '''if spherify:
#         poses, render_poses, bds = spherify_poses(poses, bds)
# 
#     else:
#         
#         c2w = poses_avg(poses)
#         print('recentered', c2w.shape)
#         print(c2w[:3,:4])
# 
#         ## Get spiral
#         # Get average pose
#         up = normalize(poses[:, :3, 1].sum(0))
# 
#         # Find a reasonable "focus depth" for this dataset
#         close_depth, inf_depth = bds.min()*.9, bds.max()*5.
#         dt = .75
#         mean_dz = 1./(((1.-dt)/close_depth + dt/inf_depth))
#         focal = mean_dz
# 
#         # Get radii for spiral path
#         shrink_factor = .8
#         zdelta = close_depth * .2
#         tt = poses[:,:3,3] # ptstocam(poses[:3,3,:].T, c2w).T
#         rads = np.percentile(np.abs(tt), 90, 0)
#         c2w_path = c2w
#         N_views = 120
#         N_rots = 2
#         if path_zflat:
#             zloc = np.percentile(tt, 10, 0)[2]
#             zloc = -close_depth * .1
#             c2w_path[:3,3] = c2w_path[:3,3] + zloc * c2w_path[:3,2]
#             rads[2] = 0.
#             N_rots = 1
#             N_views/=2
# 
#         # Generate poses for spiral path
#         render_poses = render_path_spiral(c2w_path, up, rads, focal, zdelta, zrate=.5, rots=N_rots, N=N_views)'''
#         
#         
#     render_poses = np.array(render_poses).astype(np.float32)
#     render_poses = None
# 
#     c2w = poses_avg(poses)
#     print('Data:')
#     print(poses.shape, images.shape, bds.shape)
#     
#     dists = np.sum(np.square(c2w[:3,3] - poses[:,:3,3]), -1)
#     i_test = np.argmin(dists)
#     print('HOLDOUT view is', i_test)
#     
#     images = images.astype(np.float32)
#     poses = poses.astype(np.float32)
# 
#     return images, poses, bds, render_poses, i_test, custom_pose_nerf

def _load_data(basedir, factor=None, width=None, height=None, load_imgs=True, subdir=None):
    fileposes = 'poses_bounds.npy' if subdir is None else f'poses_bounds_{subdir}.npy'
    poses_arr = np.load(os.path.join(basedir, fileposes))
    
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1,2,0])
    bds = poses_arr[:, -2:].transpose([1,0])
    
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
    sh = imageio.imread(imgfiles[0]).shape
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
    poses[2, 4, :] = poses[2, 4, :] * 1./factor

    if not load_imgs:
        return poses, bds
    
    def imread(f):
        if f.endswith('png'):
            return imageio.imread(f, ignoregamma=True)
        else:
            return imageio.imread(f)
        
    imgs = imgs = [imread(f)[...,:3]/255. for f in imgfiles]
    imgs = np.stack(imgs, -1)  
    
    print('Loaded image data', imgs.shape, poses[:,-1,0])
    return poses, bds, imgs

def load_llff_data(basedir, factor=8, subdir=None):
    poses, bds, imgs = _load_data(basedir, factor=factor, subdir=subdir) # factor=8 downsamples original imgs by 8x
    print('Loaded', basedir, bds.min(), bds.max())
    
    # Correct rotation matrix ordering and move variable dim to axis 0
    poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)
    images = imgs
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)
    
    # Rescale if bd_factor is provided
    '''sc = 1. if bd_factor is None else 1./(bds.min() * bd_factor)
    poses[:,:3,3] *= sc
    bds *= sc'''

    '''c2w = poses_avg(poses)
    print('Data:')
    print(poses.shape, images.shape, bds.shape)
    
    dists = np.sum(np.square(c2w[:3,3] - poses[:,:3,3]), -1)
    i_test = np.argmin(dists)
    print('HOLDOUT view is', i_test)'''
    
    images = images.astype(np.float32)
    poses = poses.astype(np.float32)

    return images, poses, bds


