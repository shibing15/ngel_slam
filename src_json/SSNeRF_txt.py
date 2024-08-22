import os
from pickle import TRUE
import time
import yaml
import random
import copy

import numpy as np
import torch
from torch.autograd import Variable
from colorama import Fore, Style
from scipy.spatial.transform import Rotation as R

from wisp.models.grids import OctreeGrid
from wisp.core import Rays

from src.common import get_samples, crop_pc, get_rays_all, get_pc_from_depth, getc2wError, count_parameters, getModelSize
from src.models.decoder import OCCDecoder, RGBDecoder
from src.models.renderer import Renderer
from src.utils.visualizer import Visualizer
from src.utils.datasets import get_dataset


LOSS_TYPE = ['depth_occ', 'rgb']

class SSNeRFTXT():
    """
    Main class
    """
    def __init__(self, cfg, args):
        self.cfg = cfg
        self.args = args
        self.device = cfg['device']
        self.verbose = cfg['verbose']
        torch.cuda.set_device(self.device)
        self.dilate = cfg['dilate']
        

        # param of octree grid
        self.feature_dim = cfg['models']['octree']['feature_dim']
        self.base_lod = cfg['models']['octree']['base_lod'] 
        self.num_lods = cfg['models']['octree']['num_lods'] 
        self.interpolation_type = cfg['models']['octree']['interpolation_type']
        self.multiscale_type = cfg['models']['octree']['multiscale_type']
        self.feature_std = cfg['models']['octree']['feature_std']
        self.feature_bias = cfg['models']['octree']['feature_bias']

        # param of decoders
        self.occ_dim = cfg['models']['occ_dim']
        self.rgb_dim = cfg['models']['color_dim']
        self.rgb_start = cfg['models']['color_start']

        # mapping config
        self.frustum_feature_selection = cfg['mapping']['frustum_feature_selection']
        self.pix_per_frame = cfg['mapping']['ray_sample'] # for training
        self.pix_per_keyframe = cfg['mapping']['ray_sample_keyframe_select'] # for keyframe selection
        self.keyframe_num = cfg['mapping']['keyframe_num']
        self.iters = cfg['mapping']['iters']
        self.init_iters = cfg['mapping']['init_iters']

        # learning rate
        self.grid_lr = cfg['mapping']['grid_lr']
        self.occ_lr = cfg['mapping']['occ_lr']
        self.rgb_lr = cfg['mapping']['rgb_lr']

        # visualize frequency
        self.mesh_freq = cfg['mapping']['mesh_freq']
        self.vis_freq = cfg['mapping']['vis_freq']
        self.ckpt_freq = cfg['mapping']['ckpt_freq'] 

        # mesh config
        self.mesh_reso = cfg['meshing']['reso']
        self.clean_mesh = cfg['meshing']['clean']


        self.lamudas = {}
        for t in LOSS_TYPE:
            self.lamudas[t] = cfg['mapping']['lamudas'][t]

        # sparse volume config
        self.map_length = cfg['map_length']
        self.overlap_threshold = cfg['overlap_threshold']

        # data loading and saving
        if args.input_folder is None:
            self.input_folder = cfg['data']['input_folder']
        else:
            self.input_folder = args.input_folder
        self.output_folder = os.path.join(cfg['data']['output'] + '_ORB_cl', f'{self.base_lod}_{self.num_lods}_{self.map_length}_{self.overlap_threshold}')
        os.makedirs(self.output_folder, exist_ok=True)
        print(f'Output folder: {self.output_folder}')
        
        self.ckpt_output_folder = os.path.join(self.output_folder, 'ckpt')
        os.makedirs(self.ckpt_output_folder, exist_ok=True)


        # camera intrinsic
        self.H, self.W, self.fx, self.fy, self.cx, self.cy = cfg['cam']['H'], cfg['cam'][
            'W'], cfg['cam']['fx'], cfg['cam']['fy'], cfg['cam']['cx'], cfg['cam']['cy']
        self.update_cam()

        # volume bound
        self.bbox = cfg['mapping']['bound']
        origin = (np.array(self.bbox)[:,1] + np.array(self.bbox)[:,0]) / 2
        scale = np.ceil((np.array(self.bbox)[:,1] - np.array(self.bbox)[:,0]).max()/2)
        self.origin = torch.tensor(origin, dtype=torch.float32).to(self.device)
        self.scale = torch.tensor(scale, dtype=torch.float32).to(self.device)

        # init dataloader
        self.frame_reader = get_dataset(cfg, args, 1, self.device)
        self.n_img = len(self.frame_reader)

        # freeze configs
        with open(os.path.join(self.output_folder,'config.yaml'), 'w') as file:
            file.write(yaml.dump(self.cfg, allow_unicode=True))

        # init children models container, dict keys are anchor frames 
        self.anchor_frames = []
        self.grids = {}
        self.occ_decoders = {}
        self.rgb_decoders = {}
        self.keyframe_list = []
        self.keyframe_list_anchor = {}

        self.keyframe_info = {}
        self.keyframe_img = {}

        # some local param
        self.first_init = True
        self.final = False
        self.i = 0 # iter

        with open(os.path.join(self.input_folder, 'keyframes', 'map_info.yaml'), 'r') as f:
            data = yaml.full_load(f)
        
        self.is_loop_close = False
        self.loop_closes = data['loop_close']
        if len(self.loop_closes) > 0:
            self.is_loop_close = True
        self.loop_close_detect = False
        self.first_init_frame = data['init_frame']

        self.renderer = Renderer(cfg, args, self)
        self.visualizer = Visualizer(cfg, args, self)
        self.log_info = []

        self.new_kf_in = False
        self.active_kf_c2w = {'idx': [], 'old': [], 'new': []}
        

        

    def update_cam(self):
        """
        Update the camera intrinsics according to pre-processing config, 
        such as resize or edge crop.
        """
        # resize the input images to crop_size (variable name used in lietorch)
        if 'crop_size' in self.cfg['cam']:
            crop_size = self.cfg['cam']['crop_size']
            sx = crop_size[1] / self.W
            sy = crop_size[0] / self.H
            self.fx = sx*self.fx
            self.fy = sy*self.fy
            self.cx = sx*self.cx
            self.cy = sy*self.cy
            self.W = crop_size[1]
            self.H = crop_size[0]

        # croping will change H, W, cx, cy, so need to change here]
        if self.cfg['cam']['crop_edge'] > 0:
            self.H -= self.cfg['cam']['crop_edge']*2
            self.W -= self.cfg['cam']['crop_edge']*2
            self.cx -= self.cfg['cam']['crop_edge']
            self.cy -= self.cfg['cam']['crop_edge']


    def init_local_volume_decoders(self):
        ret = {}
        
        if self.first_init:
            ret['occ'] = OCCDecoder(self.cfg['models']["occ_decoder"], occ_feat_dim=self.occ_dim).to(self.device)
        ret['rgb'] = RGBDecoder(self.cfg['models']["rgb_decoder"], self.rgb_dim).to(self.device) 

        return ret

    def save_models(self, idx):
        print('Model saving ...')
        models_path = os.path.join(self.ckpt_output_folder, f'models_{idx}.pth')
        models = {'anchors': self.anchor_frames, 'kf_anchor': self.keyframe_list_anchor, 'grids': self.grids, 'occ_decoders': self.occ_decoders, 'rgb_decoders': self.rgb_decoders}
        torch.save(models, models_path)
        print(f'Models saved in {models_path}.')


    def get_rays(self, gt_color, gt_depth, gt_c2w, pixs_per_image = None):
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy

        
        if pixs_per_image is None:
            batch_rays_o, batch_rays_d = get_rays_all(
                        H, W, fx, fy, cx, cy, gt_c2w, self.device)

            batch_rays_o =  batch_rays_o.reshape(-1, 3)
            batch_rays_d =  batch_rays_d.reshape(-1, 3)
            batch_gt_depth = gt_depth.reshape(-1)
            batch_gt_color = gt_color.reshape(-1, 3)

        
        else:
            batch_rays_o, batch_rays_d, indices = get_samples(
                0, H, 0, W, pixs_per_image, H, W, fx, fy, cx, cy, gt_c2w, self.device)
        
            batch_gt_depth = gt_depth.reshape(-1)[indices]
            batch_gt_color = gt_color.reshape(-1, 3)[indices]

            

        batch_rays_o = (batch_rays_o - self.origin) / self.scale

        return batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color
    
    
    def update_grid(self, pixs_per_image, grid, decoders, optimizer, std_c2w, selected_keyframes):
        '''
        Train model
        '''

        # Init this epoch
        loss_dict_total = {'loss': 0., 'unc': 0.}
        for t in LOSS_TYPE:
            loss_dict_total[t] = 0.
        
        
        num_kfs = len(selected_keyframes)

        pixs_per_kf = pixs_per_image // (num_kfs + 1)



        if self.frustum_feature_selection:
            for i in range(grid.num_lods):
                feat = grid.features[i]
                feat = feat.to(self.device)
                feat[self.feat_mask[i]] = self.feat_grad[i]
                grid.features[i] = feat


        batch_rays_o_list, batch_rays_d_list, batch_gt_color_list, batch_gt_depth_list = [], [], [], []


        # batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color = self.get_rays(gt_color, gt_depth, gt_c2w, pixs_per_image = pixs_per_kf)

        # batch_rays_o_list.append(batch_rays_o)
        # batch_rays_d_list.append(batch_rays_d)
        # batch_gt_color_list.append(batch_gt_color)
        # batch_gt_depth_list.append(batch_gt_depth)

        # get rays from keyframes
        for idx in selected_keyframes:
            gt_color_i, gt_depth_i = self.keyframe_img[idx]['color'], self.keyframe_img[idx]['depth']
            gt_c2w_i = std_c2w @ self.keyframe_info[idx]['c2w']

            batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color = self.get_rays(gt_color_i, gt_depth_i, gt_c2w_i, pixs_per_image = pixs_per_kf)

            
            batch_rays_o_list.append(batch_rays_o)
            batch_rays_d_list.append(batch_rays_d)
            batch_gt_color_list.append(batch_gt_color)
            batch_gt_depth_list.append(batch_gt_depth)

        batch_rays_o = torch.cat(batch_rays_o_list)
        batch_rays_d = torch.cat(batch_rays_d_list)
        batch_gt_depth = torch.cat(batch_gt_depth_list)
        batch_gt_color = torch.cat(batch_gt_color_list)


        batch_rays = Rays(batch_rays_o, batch_rays_d)

        loss_dict = self.renderer.get_hit(batch_rays, batch_gt_color, batch_gt_depth, grid, decoders)

        loss = 0.
        for k, v in loss_dict.items():
            loss += self.lamudas[k] * loss_dict[k]
            loss_dict_total[k] += loss_dict[k].cpu().item()

        loss.backward(retain_graph = True)
        optimizer.step()
        optimizer.zero_grad()


        if self.frustum_feature_selection: 
            for i in range(grid.num_lods):
                feat = grid.features[i]
                feat = feat.detach()
                feat[self.feat_mask[i]] = self.feat_grad[i].clone().detach()
                grid.features[i] = feat
        

        loss_dict_total['loss']+= loss.cpu().item()
        # uncertainty = unc_dict['occ'].mean().cpu().item()
        # loss_dict_total['unc'] += uncertainty

        if self.verbose:
            edesc = f'Iter {self.i} train info: '
            for k, v in loss_dict_total.items():
                loss_dict_total[k] = v
                if v > 0:
                    edesc += k + '=' + "{:.5f} ".format(v)
            
            print(edesc) 

        return loss_dict_total

    def keyframe_selection(self, gt_color, gt_depth, gt_c2w, keyframe_list, keyframe_info, active_kf, grid, std_c2w):

        selected_keyframe = []
        # 1. overlap keyframes
        # get rays of target frame
        rays_o, rays_d, _, _ = self.get_rays(gt_color, gt_depth, gt_c2w, pixs_per_image = self.pix_per_keyframe)
        rays = Rays(rays_o, rays_d)

        # get num of unique intersect boxes
        level = grid.max_lod
        result = grid.raytrace(rays, level)
        pidx = torch.unique(result.pidx)

        # select keyframe by overlap counts
        overlap_counts = []
        for i in keyframe_list:
            # get rays of target keyframe
            gt_color, gt_depth, gt_c2w = keyframe_info[i]['color'], keyframe_info[i]['depth'], std_c2w @ keyframe_info[i]['c2w']
            rays_o, rays_d, _, _ = self.get_rays(gt_color, gt_depth, gt_c2w, pixs_per_image = self.pix_per_keyframe)
            rays = Rays(rays_o, rays_d)

            # get num of unique intersect boxes
            result = grid.raytrace(rays, level) #level
            pidx_i = torch.unique(result.pidx)

            # number of shared boxes
            combined = torch.cat((pidx_i, pidx), dim = -1)
            _, counts = torch.unique(combined, return_counts = True)
            overlap_counts.append((counts > 1).sum().cpu())

        selected_idx = np.argsort(overlap_counts)[-self.keyframe_num:]
        selected_keyframe_overlap = list(np.array(keyframe_list)[selected_idx])
        selected_keyframe += selected_keyframe_overlap

        # 2. pose diff keyframe
        # if len(active_kf_c2w['old']) > 0:
        #     trans_error = (np.array(active_kf_c2w['old'])[:, :3, 3] - np.array(active_kf_c2w['new'])[:, :3, 3]).mean()
        #     selected_idx = np.argsort(trans_error)[-self.keyframe_num:]
        #     selected_keyframe_active = list(np.array(active_kf_c2w['idx'])[selected_idx])
        #     selected_keyframe += selected_keyframe_active

        # 3. random
        selected_keyframe_random = random.sample(active_kf, min(len(active_kf), self.keyframe_num))
        selected_keyframe += selected_keyframe_random

        return list(np.unique(selected_keyframe))

    def get_params_optimized(self, grid_lr, occ_decoder_lr, rgb_decoder_lr, grid, decoders): 
        params = []

        if self.frustum_feature_selection:
            self.feat_grad = []
            for i in range(grid.num_lods):
                feat_num = len(self.feat_mask[i])
                if self.init_every:
                    self.feat_mask[i] = torch.ones_like(self.feat_mask[i], dtype=torch.bool).to(self.device)
                mask = self.feat_mask[i]
                feat_grad = grid.features[i][mask].clone()
                feat_grad = Variable(feat_grad.to(self.device), requires_grad=True)
                self.feat_grad.append(feat_grad)
                params.append({'params':self.feat_grad[i], 'lr': grid_lr[i]})
        else:
            for i in range(grid.num_lods):
                grid.features[i] = Variable(grid.features[i].to(self.device), requires_grad = True)
                params.append({'params':grid.features[i], 'lr': grid_lr[i]})

        params.append({'params': decoders['rgb'].parameters(), 'lr':rgb_decoder_lr}) 
        if self.first_init:
            params.append({'params': decoders['occ'].parameters(), 'lr': occ_decoder_lr})

        return params
    
    def update_keyframes(self, kf_infos):
        # new_kf_in
        last_kf_idx = int(kf_infos[-1][0])
        # print(f'last keyframe: {last_kf_idx}')
        local_ba = False
        global_ba = False

        if last_kf_idx not in self.keyframe_list:
            print(f'New keyframe {last_kf_idx} in. Waiting for local BA...')
            self.new_kf_in = True
            self.active_kf_c2w = {'idx': [], 'old': [], 'new': []}
            self.keyframe_list.append(last_kf_idx)

            # new kf's information
            info = kf_infos[-1]
            position = torch.tensor([info[1], info[2], info[3]]).to(self.device)
            orientation = np.array([info[4], info[5], info[6], info[7]])
            c2w = torch.zeros([4,4]).to(self.device)
            c2w[3, 3] = 1
            c2w[:3, :3] = torch.tensor(R.from_quat(orientation).as_matrix()).to(self.device)
            c2w[:3, 3] = position

            ret = self.frame_reader[last_kf_idx]
            gt_color, gt_depth, gt_c2w = ret['color'], ret['depth'], ret['pose']
            
            # self.keyframe_info[last_kf_idx] = {'idx': last_kf_idx, 'color': gt_color, 'depth': gt_depth, 'c2w': c2w, 'gt_c2w': gt_c2w}
            self.keyframe_info[last_kf_idx] = {'idx': last_kf_idx, 'c2w': c2w}
            self.keyframe_img[last_kf_idx] = {'color': gt_color, 'depth': gt_depth}
            if len(self.keyframe_list) <= 2:
                local_ba = True
        
        # if self.loop_close_detect:
        #     active_kf_c2w = self.active_kf_c2w
        #     self.active_kf_c2w = {'idx': [], 'old': [], 'new': []}
            
        #     for info in kf_infos:
        #         idx = int(info[0])
        #         if idx in self.keyframe_list:
        #             position = torch.tensor([info[1], info[2], info[3]]).to(self.device)
        #             orientation = np.array([info[4], info[5], info[6], info[7]])
        #             c2w = torch.zeros([4,4]).to(self.device)
        #             c2w[3, 3] = 1

        #             c2w[:3, :3] = torch.tensor(R.from_quat(orientation).as_matrix()).to(self.device)
        #             c2w[:3, 3] = position

        #             if (self.keyframe_info[idx]['c2w'] != c2w).any():
        #                 self.active_kf_c2w['idx'].append(idx)
        #                 self.active_kf_c2w['old'].append(self.keyframe_info[idx]['c2w'].cpu().numpy())
        #                 self.active_kf_c2w['new'].append(c2w.cpu().numpy())
        #                 self.keyframe_info[idx]['c2w'] = c2w
        #                 global_ba = True
        #     if last_kf_idx not in self.active_kf_c2w['idx']:
        #         global_ba = False
        #     if not global_ba:
        #         self.active_kf_c2w = active_kf_c2w

        # wait for local ba
        if self.new_kf_in:# and not global_ba:
            for info in kf_infos:
                idx = int(info[0])
                if idx in self.keyframe_list:
                    position = torch.tensor([info[1], info[2], info[3]]).to(self.device)
                    orientation = np.array([info[4], info[5], info[6], info[7]])
                    c2w = torch.zeros([4,4]).to(self.device)
                    c2w[3, 3] = 1

                    c2w[:3, :3] = torch.tensor(R.from_quat(orientation).as_matrix()).to(self.device)
                    c2w[:3, 3] = position

                    if (self.keyframe_info[idx]['c2w'] != c2w).any():
                        self.active_kf_c2w['idx'].append(idx)
                        self.active_kf_c2w['old'].append(self.keyframe_info[idx]['c2w'].cpu().numpy())
                        self.active_kf_c2w['new'].append(c2w.cpu().numpy())
                        self.keyframe_info[idx]['c2w'] = c2w
                        local_ba = True
            if last_kf_idx not in self.active_kf_c2w['idx'] and len(self.keyframe_list) > 2:
                local_ba = False


        if local_ba or global_ba:
            if len(self.keyframe_list) != len(kf_infos):
                new_keyframes = []
                for info in kf_infos:
                    new_keyframes.append(int(info[0])) 

                for keyframe in self.keyframe_list:
                    if keyframe not in new_keyframes:
                        self.keyframe_list.remove(keyframe)
                        del self.keyframe_info[keyframe]
                        del_keyframe = keyframe

                        for anchor in self.anchor_frames:
                            keyframe_list = self.keyframe_list_anchor[anchor]
                            if anchor == del_keyframe:
                                del self.keyframe_list_anchor[anchor]

                                self.anchor_frames.remove(anchor)
                                anchor = keyframe_list[1]

                                self.grids[anchor] = self.grids[del_keyframe]
                                self.occ_decoders[anchor] = self.occ_decoders[del_keyframe]
                                self.rgb_decoders[anchor] = self.rgb_decoders[del_keyframe]

                                del self.grids[del_keyframe]
                                del self.occ_decoders[del_keyframe]
                                del self.rgb_decoders[del_keyframe]

                                
                            if del_keyframe in keyframe_list:
                                keyframe_list.remove(del_keyframe)

                            self.keyframe_list_anchor[anchor] = keyframe_list
        
        if local_ba:
            self.new_kf_in = False
            print(f'Local BA done! # Active kf: {len(self.active_kf_c2w["idx"])}, # non-active kf: {len(self.keyframe_list) - len(self.active_kf_c2w["idx"])} , Active kf: {self.active_kf_c2w["idx"]}')

        # if global_ba:
        #     self.loop_close_detect = False
        #     local_ba = False
        #     print(f'Global BA done! # Active kf: {len(self.active_kf_c2w["idx"])}, Active kf: {self.active_kf_c2w["idx"]}')

        return local_ba, last_kf_idx, global_ba


    def grid_increment(self, std_c2w, change_frame_idx, grid = None):
        gt_depth = []
        c2w = []
        
        for idx in change_frame_idx:
            gt_depth.append(self.keyframe_img[idx]['depth'])
            c2w.append(std_c2w @ self.keyframe_info[idx]['c2w'])
        
        points = get_pc_from_depth(gt_depth, c2w, self.fx, self.fy, self.cx, self.cy, self.device)
        points.reshape(-1, 3)
        points = crop_pc(points, self.bbox)
        points = (points - self.origin) / self.scale

        if grid is None:
            grid = OctreeGrid.from_pointcloud(
                        pointcloud=points,
                        feature_dim=self.feature_dim,
                        base_lod=self.base_lod, 
                        num_lods=self.num_lods,
                        interpolation_type=self.interpolation_type, 
                        multiscale_type=self.multiscale_type,
                        feature_std=self.feature_std,
                        feature_bias=self.feature_bias,
                        dilate = self.dilate
                        )
            
        else:
            grid.update(points, self.device, dilate = self.dilate)

        # get mask of feat needed to optimize
        if self.frustum_feature_selection:
            H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy
            rays_o_list, rays_d_list = [], []
            for idx in change_frame_idx:
                kf = self.keyframe_info[idx]
                c2w = std_c2w @ kf['c2w']
                rays_o, rays_d = get_rays_all(H, W, fx, fy, cx, cy, c2w, self.device)
                rays_o_list.append(rays_o)
                rays_d_list.append(rays_d)
            rays_o = torch.cat(rays_o_list)
            rays_d = torch.cat(rays_d_list)
            rays = Rays(rays_o, rays_d)
            self.feat_mask = grid.get_mask_from_rays(rays)
        
        return grid

    def get_new_std_c2w(self, anchor_frame):
        c2w = self.keyframe_info[anchor_frame]['c2w']

        std_c2w = self.world_std_c2w @ c2w.inverse()
        
        return std_c2w
            
    def get_overlap(self, grid, est_c2w, std_c2w, num_pixels = None):
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy

        c2w = std_c2w @ est_c2w
        if num_pixels is None:
            batch_rays_o, batch_rays_d = get_rays_all(
                        H, W, fx, fy, cx, cy, c2w, self.device)
            num_pixels = H * W

        else:
            batch_rays_o, batch_rays_d, indices = get_samples(
                0, H, 0, W, num_pixels, H, W, fx, fy, cx, cy, c2w, self.device)

        batch_rays_o = (batch_rays_o - self.origin) / self.scale

        rays = Rays(batch_rays_o, batch_rays_d)
        level = grid.max_lod
        result = grid.raytrace(rays, level)
        ridx = torch.unique(result.ridx)

        num_ridx = ridx.shape[0]
        return num_ridx / num_pixels


    def c2wdiff(self, idx1, idx2, kf_infos):
        flag1 = False
        flag2 = False
        for info in kf_infos:
            idx = int(info[0])
            if idx == idx1:
                position = torch.tensor([info[1], info[2], info[3]]).to(self.device)
                orientation = np.array([info[4], info[5], info[6], info[7]])
                c2w1 = torch.zeros([4,4]).to(self.device)
                c2w1[3, 3] = 1

                c2w1[:3, :3] = torch.tensor(R.from_quat(orientation).as_matrix()).to(self.device)
                c2w1[:3, 3] = position
                flag1 = True

            if idx == idx2:
                position = torch.tensor([info[1], info[2], info[3]]).to(self.device)
                orientation = np.array([info[4], info[5], info[6], info[7]])
                c2w2 = torch.zeros([4,4]).to(self.device)
                c2w2[3, 3] = 1

                c2w2[:3, :3] = torch.tensor(R.from_quat(orientation).as_matrix()).to(self.device)
                c2w2[:3, 3] = position
                flag2 = False
            
            if flag1 and flag2:
                break
            
        angle, trans = getc2wError(c2w1, c2w2)

        return angle, trans
                

        

    def update_volume(self, anchor_frame, local_ba_flag, active_kf, std_c2w, vol_init = False):
            
        grid = self.grids[anchor_frame]

        occ_decoder = self.occ_decoders[anchor_frame]
        rgb_decoder = self.rgb_decoders[anchor_frame]
        decoders = {'occ': occ_decoder, 'rgb': rgb_decoder}


        if local_ba_flag or vol_init:
            # ret = self.keyframe_info[keyframe_list[-1]]
            # gt_color, gt_depth, c2w = ret['color'], ret['depth'], std_c2w @ ret['c2w']

            grid = self.grid_increment(std_c2w, active_kf, grid)

        
        # else:
        #     frame_update = random.choices(keyframe_list, weights=keyframe_list)[0]
        #     ret = self.keyframe_info[frame_update]
        #     gt_color, gt_depth, c2w = ret['color'], ret['depth'], std_c2w @ ret['c2w']

        params = self.get_params_optimized(self.grid_lr, self.occ_lr, self.rgb_lr, grid, decoders)

        optimizer = torch.optim.Adam(params)

        # selected_keyframes = self.keyframe_selection(gt_color, gt_depth, c2w, keyframe_list, self.keyframe_info, active_kf, grid, std_c2w)  
        selected_keyframes = random.sample(active_kf, min(len(active_kf), self.keyframe_num))
        if active_kf[-1] not in selected_keyframes:
            selected_keyframes.append(active_kf[-1])

        if self.verbose:
            print(Fore.GREEN)
            print(f"Epoch {self.i} mapping Frame Update: {selected_keyframes}" )
            print(Style.RESET_ALL)

        if self.first_init:
            for i in range(self.init_iters):
                self.update_grid(self.pix_per_frame, grid, decoders, optimizer, std_c2w, selected_keyframes)
            self.first_init = False
            self.i += 1

        else:
            for i in range(self.iters):
                self.update_grid(self.pix_per_frame, grid, decoders, optimizer, std_c2w, selected_keyframes)
            self.i += 1
        
        self.grids[anchor_frame] = grid

        self.occ_decoders[anchor_frame] = decoders['occ']
        self.rgb_decoders[anchor_frame] = decoders['rgb']
        
        torch.cuda.empty_cache()

        return grid, decoders

    def run(self):
        ret = self.frame_reader[self.first_init_frame]
        self.world_std_c2w = ret['pose']
        
        
        
        # self.anchor_frame = self.first_init_frame
        idx = self.first_init_frame
        
        decoders = self.init_local_volume_decoders()

        loop_idx = 0
        last_loop_close = 0
        new_volume_flag = True
        loop_close_flag = False
        if self.is_loop_close:
            loop_close = self.loop_closes[0]
        
        while True:
            kf_file = os.path.join(self.input_folder, 'keyframes', f'{int(idx)}.txt')
            if not os.path.exists(kf_file):
                idx += 1
            kf_infos = np.loadtxt(kf_file).reshape(-1, 8)
            last_kf_idx = int(kf_infos[-1][0])
            if last_kf_idx == self.first_init_frame:
                idx += 1
            else:
                anchor_frame = last_kf_idx
                idx = last_kf_idx
                active_kf_anchor = [last_kf_idx]
                std_c2w = self.world_std_c2w
                self.anchor_frame = anchor_frame
                break

        while True:
            if loop_idx >= len(self.loop_closes):
                loop_close = self.n_img
            else:
                loop_close = self.loop_closes[loop_idx]
            
            if new_volume_flag:
                vol_init = True
                # self.keyframe_list_anchor[anchor_frame] = active_kf_anchor
                self.keyframe_list_anchor[anchor_frame] = []
                self.anchor_frames.append(anchor_frame)
                self.occ_decoders[anchor_frame] = decoders['occ']
                self.rgb_decoders[anchor_frame] = decoders['rgb']
                self.grids[anchor_frame] = None
                new_map = 0
                if not self.first_init:
                    std_c2w = self.get_new_std_c2w(anchor_frame)

                # load keyframe files
                while True:
                    kf_file = os.path.join(self.input_folder, 'keyframes', f'{int(idx)}.txt')
                    if not os.path.exists(kf_file):
                        idx += 1
                    else:
                        kf_infos = np.loadtxt(kf_file).reshape(-1, 8)
                        last_kf_idx = int(kf_infos[-1][0])
                        if last_kf_idx < anchor_frame:
                            idx += 1
                        else:
                            break

            
            while True:
                
                while True:
                    kf_file = os.path.join(self.input_folder, 'keyframes', f'{int(idx)}.txt')
                    if not os.path.exists(kf_file):
                        idx += 1
                    else:
                        break
                
                kf_infos = np.loadtxt(kf_file).reshape(-1, 8)

                local_ba_flag, last_kf_idx, global_ba_flag = self.update_keyframes(kf_infos)

                if new_volume_flag:

                    # new keyframe in and local ba finished
                    if last_kf_idx not in self.keyframe_list_anchor[anchor_frame]:
                        self.keyframe_list_anchor[anchor_frame].append(last_kf_idx)
                    
                    if local_ba_flag:
                        # self.keyframe_img = {}
                        print(f'# KF of volume: {len(self.keyframe_list_anchor[anchor_frame])}')
                        active_kf_anchor = sorted(list(set(self.keyframe_list_anchor[anchor_frame]) & set(self.active_kf_c2w['idx'])))
                        if len(self.keyframe_list) <= 2:
                            active_kf_anchor = self.keyframe_list_anchor[anchor_frame]
                        std_c2w = self.get_new_std_c2w(anchor_frame)
                        # for kf_i in active_kf_anchor:
                        #     ret = self.frame_reader[kf_i]
                        #     self.keyframe_img[kf_i] = {'color': ret['color'], 'depth': ret['depth']}

                        
                        # need new map?
                        # if len(self.keyframe_list_anchor[anchor_frame]) > 10:
                            
                            # a new map required
                            # if len(self.keyframe_list_anchor[anchor_frame]) >= self.map_length:
                            # if len(active_kf_anchor) <= 0.5 * len(self.keyframe_list_anchor[anchor_frame]):
                        if len(self.active_kf_c2w['idx']) > 0:
                            if self.active_kf_c2w['idx'][0] - self.anchor_frame > self.map_length: 
                                if new_map == 0:
                                    new_anchor = self.active_kf_c2w['idx'][0]
                                if self.active_kf_c2w['idx'][0] == new_anchor:
                                    new_map += 1
                                else:
                                    new_anchor = self.active_kf_c2w['idx'][0]
                                    new_map = 1
                                
                            else:
                                new_map = 0

                            if new_map >= 3:
                                decoders['occ'] = copy.deepcopy(self.occ_decoders[anchor_frame])
                                decoders['rgb'] = copy.deepcopy(self.rgb_decoders[anchor_frame])
                                
                                self.visualizer.render_img(last_kf_idx, self.anchor_frames, idx - 1, self.grids, self.occ_decoders, self.rgb_decoders, vis=True)
                                self.visualizer.extract_mesh(last_kf_idx, self.anchor_frames, idx - 1, self.mesh_reso, self.grids, self.occ_decoders, self.rgb_decoders, vis_every=True, clean = self.clean_mesh)
                                self.save_models(f'vol_{len(self.anchor_frames)}')

                                if not loop_close_flag:
                                    self.anchor_frame = new_anchor
                                    new_volume_flag = True
                                    anchor_frame = last_kf_idx
                                    self.keyframe_img = {}
                                    ret = self.frame_reader[anchor_frame]
                                    self.keyframe_img[anchor_frame] = {'color': ret['color'], 'depth': ret['depth']}
                                    active_kf_anchor = [anchor_frame]
                                else:
                                    new_volume_flag = False
                                break
                            
                
                    self.update_volume(anchor_frame, local_ba_flag, active_kf_anchor, std_c2w, vol_init)
                    vol_init = False
                    idx += 1


                else:
                    if self.loop_close_detect:
                        if last_kf_idx not in self.keyframe_list_anchor[anchor_frame]:
                            self.keyframe_list_anchor[anchor_frame].append(last_kf_idx)
                    
                    if self.new_kf_in and last_kf_idx not in free_kf:
                        free_kf.append(last_kf_idx)
                        # active_kf_anchor.append(last_kf_idx)

                    if local_ba_flag:
                        # self.keyframe_img = {}
                        active_kf_anchors = []
                        num_active_kf_anchors = []
                        for anchor_frame in self.anchor_frames:
                            active_kf_anchor = list(set(self.keyframe_list_anchor[anchor_frame]) & set(self.active_kf_c2w['idx']))
                            active_kf_anchors.append(active_kf_anchor)
                            num_active_kf_anchors.append(len(active_kf_anchor))

                        anchor_idx = np.argmax(num_active_kf_anchors)
                        anchor_frame = self.anchor_frames[anchor_idx]
                        active_kf_anchor = np.array(active_kf_anchors[anchor_idx])
                        active_kf_anchor = list(active_kf_anchor[active_kf_anchor >= last_loop_close])
                        for kf in free_kf:
                            if kf in self.active_kf_c2w['idx']:
                                self.keyframe_list_anchor[anchor_frame].append(kf)
                                active_kf_anchor.append(kf)
                        free_kf = []
                        std_c2w = self.get_new_std_c2w(anchor_frame)
                        print(f'# active KF of volume: {len(active_kf_anchor)}')
                        # for kf_i in active_kf_anchor:
                        #     ret = self.frame_reader[kf_i]
                        #     self.keyframe_img[kf_i] = {'color': ret['color'], 'depth': ret['depth']}

                        # new volume needed
                        if np.max(num_active_kf_anchors) <= self.overlap_threshold:
                            new_volume_flag = True
                            
                            decoders['occ'] = copy.deepcopy(self.occ_decoders[anchor_frame])
                            decoders['rgb'] = copy.deepcopy(self.rgb_decoders[anchor_frame])

                            self.visualizer.render_img(last_kf_idx, self.anchor_frames, idx - 1, self.grids, self.occ_decoders, self.rgb_decoders, vis=True)
                            self.visualizer.extract_mesh(last_kf_idx, self.anchor_frames, idx - 1, self.mesh_reso, self.grids, self.occ_decoders, self.rgb_decoders, vis_every=True, clean = self.clean_mesh)
                            self.save_models(f'vol_{len(self.anchor_frames)}')
                            anchor_frame = last_kf_idx
                            self.keyframe_img = {}
                            ret = self.frame_reader[anchor_frame]
                            self.keyframe_img[anchor_frame] = {'color': ret['color'], 'depth': ret['depth']}
                            self.anchor_frame = active_kf_anchor[0]
                            active_kf_anchor = [anchor_frame]
                            break

                    # if global_ba_flag:
                        
                    #     self.visualizer.extract_mesh(f'loop_close_{last_kf_idx}', self.anchor_frames, idx - 1, self.mesh_reso, self.grids, self.occ_decoders, self.rgb_decoders, vis_every=False, clean = self.clean_mesh)
                        # self.keyframe_img = {}
                        # for kf_i in active_kf_anchor:
                        #     ret = self.frame_reader[kf_i]
                        #     self.keyframe_img[kf_i] = {'color': ret['color'], 'depth': ret['depth']}


                    
                    self.update_volume(anchor_frame, local_ba_flag, active_kf_anchor, std_c2w)
                    idx += 1

                # loop close detect
                if self.is_loop_close:
                    if last_kf_idx == loop_close and local_ba_flag:
                        print('Loop close detect! Wait for global ba ...')
                        # self.grids[anchor_frame] = copy.deepcopy(grid)
                        decoders['occ'] = copy.deepcopy(self.occ_decoders[anchor_frame])
                        decoders['rgb'] = copy.deepcopy(self.rgb_decoders[anchor_frame])
                        # self.keyframe_list_anchor[anchor_frame] = self.keyframe_list
                        
                        # idx = last_kf_idx
                        # self.visualizer.render_img(last_kf_idx, self.anchor_frames, idx, self.grids, self.occ_decoders, self.rgb_decoders, vis=True)
                        self.visualizer.extract_mesh(f'before loop close_{idx}', self.anchor_frames, idx - 1, self.mesh_reso, self.grids, self.occ_decoders, self.rgb_decoders, vis_every=True, clean = self.clean_mesh)
                        self.save_models(f'vol_{len(self.anchor_frames)}')

                        # anchor_frame = last_kf_idx
                        loop_idx += 1
                        loop_close_flag = True
                        new_volume_flag = False
                        self.loop_close_detect = True
                        free_kf = []
                        
                        last_loop_close = loop_close
                        break
                        
                    
                # if self.i % self.vis_freq == 0:
                #     self.visualizer.render_img(last_kf_idx, self.anchor_frames, idx - 1, self.grids, self.occ_decoders, self.rgb_decoders, vis=True)

                if self.i % self.mesh_freq == 0:
                    self.visualizer.extract_mesh(last_kf_idx, self.anchor_frames, idx - 1, self.mesh_reso, self.grids, self.occ_decoders, self.rgb_decoders, clean = self.clean_mesh)

                if self.i % self.ckpt_freq == 0:
                    self.save_models(last_kf_idx)

                if idx >= self.n_img - 1:
                    self.visualizer.render_img(last_kf_idx, self.anchor_frames, idx - 1, self.grids, self.occ_decoders, self.rgb_decoders, vis=True)
                    self.visualizer.extract_mesh('final', self.anchor_frames, idx - 1, self.mesh_reso, self.grids, self.occ_decoders, self.rgb_decoders, vis_every=True, clean = self.clean_mesh)
                    self.save_models('final')
                    break
            
            if idx >= self.n_img - 1:
                # TODO: post propcess
                kf_file = os.path.join(self.input_folder, 'keyframes', f'{idx}.txt')
                kf_infos = np.loadtxt(kf_file).reshape(-1, 8)
                
                self.get_keyframes(kf_infos, self.keyframe_list)

                for anchor_frame in self.anchor_frames:
                    grid = self.grids[anchor_frame]
                    occ_decoder = self.occ_decoders[anchor_frame]
                    rgb_decoder = self.rgb_decoders[anchor_frame]
                    decoders = {'occ': occ_decoder, 'rgb': rgb_decoder}

                    params = self.get_params_optimized(self.grid_lr, self.occ_lr, self.rgb_lr, grid, decoders)

                    optimizer = torch.optim.Adam(params)

                    std_c2w = self.get_new_std_c2w(anchor_frame)

                    for i in range(300):

                        self.update_grid(self.pix_per_frame*5, grid, decoders, optimizer, std_c2w, self.keyframe_list_anchor[anchor_frame])

                    self.grids[anchor_frame] = grid

                    self.occ_decoders[anchor_frame] = decoders['occ']
                    self.rgb_decoders[anchor_frame] = decoders['rgb']
                self.visualizer.render_img(last_kf_idx, self.anchor_frames, idx, self.grids, self.occ_decoders, self.rgb_decoders, vis=True)
                self.visualizer.extract_mesh('postprocess', self.anchor_frames, idx, self.mesh_reso, self.grids, self.occ_decoders, self.rgb_decoders, vis_every=True, clean = self.clean_mesh)
                self.save_models('postprocess')

                break



    def load_model(self, models_path):
        models = torch.load(models_path)
        self.anchor_frames, self.grids, self.occ_decoders, self.rgb_decoders = models['anchors'], models['grids'], models['occ_decoders'], models['rgb_decoders']
        
    
    def render_img(self, frame_idx, pose_file_idx, vis = True, fuse_method = 'max'):

        return self.visualizer.render_img(frame_idx, self.anchor_frames, pose_file_idx, self.grids, self.occ_decoders, self.rgb_decoders, vis=vis, fuse_method=fuse_method)

    def extract_mesh(self, idx, pose_file_idx, vis_every = True, clean = True):

        return self.visualizer.extract_mesh(idx, self.anchor_frames, pose_file_idx, self.mesh_reso, self.grids, self.occ_decoders, self.rgb_decoders, vis_every=vis_every, clean = clean)


                

                

