import torch
from torch.nn.functional import grid_sample


def batched_back_project_loop_view(
        coords, origin, voxel_size, feats, KRcam,
        img_res=224, reverse_x=False, float_type=torch.float32, concat_z=True
):
    '''
    Unproject the image features to form a 3D dense feature volume

    :param coords: coordinates of voxels,
    dim: (batch size, num of voxels per sample, 3) (3: x, y, z)
    :param origin: origin of the partial voxel volume (xyz position of voxel (0, 0, 0))
    dim: (batch size, 3) (3: x, y, z)
    :param voxel_size: floats specifying the size of a voxel
    :param feats: image features
    dim: (num of views, batch size, C, H, W)
    :param KRcam: projection matrix
    dim: (num of views, batch size, 4, 4)
    :return: feature_volume_all: 3D feature volumes
    dim: (batch size, num of voxels per sample, c + 1)
    :return: count: number of times each voxel can be seen
    dim: (batch size, num of voxels per sample)
    '''
    n_views, bs, c, h, w = feats.shape
    coords_batch = coords
    origin_batch = origin
    feats_batch = feats
    proj_batch = KRcam

    grid_batch = coords_batch * voxel_size + origin_batch.to(dtype=float_type)[:, None, :]
    rs_grid = grid_batch.unsqueeze(0).expand(n_views, -1, -1, -1)
    rs_grid = rs_grid.permute(0, 1, 3, 2).contiguous()
    nV = rs_grid.shape[-1]
    rs_grid = torch.cat([rs_grid, torch.ones([n_views, bs, 1, nV], device='cuda', dtype=float_type)], dim=2)

    # Project grid
    im_p = proj_batch @ rs_grid
    im_x, im_y, im_z = im_p[:, :, 0], im_p[:, :, 1], im_p[:, :, 2]
    im_x = im_x / im_z
    im_y = im_y / im_z
    if reverse_x:
        im_x = img_res - 1 - im_x
    im_x = im_x / img_res * w
    im_y = im_y / img_res * h

    # im_z = -im_z

    im_grid = torch.stack([2 * im_x / (w - 1) - 1, 2 * im_y / (h - 1) - 1], dim=-1)
    mask = im_grid.abs() <= 1
    mask = (mask.sum(dim=-1) == 2) & (im_z > 0)

    feats_batch = feats_batch.view(n_views, bs, c, h, w).contiguous()
    im_grid = im_grid.view(n_views, bs, 1, -1, 2)

    mask = mask.view(n_views, bs, -1)
    im_z = im_z.view(n_views, bs, -1)

    features = 0
    for nview in range(n_views):
        featuresi = grid_sample(feats_batch[nview], im_grid[nview],
                                padding_mode='zeros', align_corners=True).reshape(bs, c, -1)
        featuresi[mask[nview].unsqueeze(1).expand(-1, c, -1) == False] = 0
        features += featuresi
    im_z[~mask] = 0

    count = mask.sum(dim=0).to(dtype=float_type)

    # aggregate multi view
    # features = features.sum(dim=0)
    mask = mask.sum(dim=0)
    invalid_mask = mask == 0
    mask[invalid_mask] = 1
    in_scope_mask = mask.unsqueeze(1)
    features /= in_scope_mask
    features = features.permute(0, 2, 1).contiguous()
    if concat_z:
        # concat normalized depth value
        im_z = im_z.sum(dim=0).unsqueeze(2) / in_scope_mask.permute(0, 2, 1).contiguous()
        im_z_mean = [iz[iz > 0].mean() for iz in im_z]
        im_z_std = torch.stack([torch.norm(iz[iz > 0] - im_z_mean[i]) + 1e-5 for i, iz in enumerate(im_z)])
        im_z_norm = torch.stack([(iz - im_z_mean[i]) / im_z_std[i] for i, iz in enumerate(im_z)])
        im_z_norm[im_z <= 0] = 0
        features = torch.cat([features, im_z_norm], dim=2)

    return features, count


def batched_back_project_dev(coords, origin, voxel_size, feats, KRcam, img_res=224, camera_positions=None, pluker_rays=False):
    '''
    Unproject the image features to form a 3D dense feature volume

    :param coords: coordinates of voxels,
    dim: (batch size, num of voxels per sample, 3) (3: x, y, z)
    :param origin: origin of the partial voxel volume (xyz position of voxel (0, 0, 0))
    dim: (batch size, 3) (3: x, y, z)
    :param voxel_size: floats specifying the size of a voxel
    :param feats: image features
    dim: (num of views, batch size, C, H, W)
    :param KRcam: projection matrix
    dim: (num of views, batch size, 4, 4)
    :param pluker_rays: whether to cat pluker rays
    :return: feature_volume_all: 3D feature volumes
    dim: (batch size, num of voxels per sample, c + 1)
    :return: count: number of times each voxel can be seen
    dim: (batch size, num of voxels per sample)
    '''
    n_views, bs, c, h, w = feats.shape
    occ_reso = int(round(coords.shape[1] ** (1 / 3)))
    if pluker_rays is True:
        ret = torch.zeros([bs, (c + 6) * n_views, occ_reso, occ_reso, occ_reso], dtype=torch.float32, device=coords.device)
    else:
        ret = torch.zeros([bs, c * n_views, occ_reso, occ_reso, occ_reso], dtype=torch.float32, device=coords.device)
    # all batch share the same grid
    coords_batch = coords
    origin_batch = origin
    grid_batch = (coords_batch * voxel_size - 1).permute(1, 0)  # assert origin is -1,-1,-1
    rs_grid = grid_batch.unsqueeze(0).expand(n_views, -1, -1)
    rs_grid = rs_grid.permute(0, 2, 1).contiguous()
    nV = rs_grid.shape[-1]
    rs_grid = torch.cat([rs_grid, torch.ones([n_views, 1, nV], device='cuda', dtype=torch.float32)], dim=1)

    for i in range(bs):
        feats_batch = feats[:, i, :, :]
        proj_batch = KRcam[:, i, :, :]
        if camera_positions is not None:
            camera_position_batch = camera_positions[:, i]
        # Project grid
        im_p = proj_batch @ rs_grid
        im_x, im_y, im_z = im_p[:, 0], im_p[:, 1], im_p[:, 2]
        im_x = im_x / im_z
        im_y = im_y / im_z
        im_x = im_x / img_res * w
        im_y = im_y / img_res * h

        im_grid = torch.stack([2 * im_x / (w - 1) - 1, 2 * im_y / (h - 1) - 1], dim=-1)
        mask = im_grid.abs() <= 1
        mask = (mask.sum(dim=-1) == 2) & (im_z > 0)

        feats_batch = feats_batch.view(n_views, c, h, w)
        im_grid = im_grid.view(n_views, 1, -1, 2)

        features = grid_sample(feats_batch, im_grid, padding_mode='zeros', align_corners=True)
        features = features.view(n_views, c, -1)
        mask = mask.view(n_views, -1)
        im_z = im_z.view(n_views, -1)
        # remove nan
        features[mask.unsqueeze(1).expand(-1, c, -1) == False] = 0
        im_z[mask == False] = 0
        # assert mode is cat
        features = features.reshape(n_views * c, -1)
        features = features.permute(1, 0).contiguous()

        o = camera_position_batch
        v = grid_batch
        d = v[:, None, :] - o[None, :, :]  # nVoxel, nView, 3
        d = torch.nn.functional.normalize(d, dim=-1)
        o = -o
        oxd = torch.cross(o[None], d, dim=-1)  # nVoxel, nView, 3
        plucker_ray_feat = torch.cat([oxd, d], dim=-1).reshape(nV, -1)

        features_tmp = features.reshape(nV, n_views, c)
        plucker_ray_feat_tmp = plucker_ray_feat.reshape(nV, n_views, 6)
        tmp = torch.cat([features_tmp, plucker_ray_feat_tmp], dim=-1)
        tmp = tmp.reshape(occ_reso, occ_reso, occ_reso, n_views * (c + 6)).permute(3, 0, 1, 2)
        ret[i] = tmp

    return ret


def back_project_sparse_dev(pts, batch_id, feats, KRcam, img_res=320,
                            reverse_x=False, mode="mean",
                            cat_KRcam=False, cat_voxel_cam_offset=False,
                            plucker_ray=False,
                            camera_positions=None,
                            arrange='by_type',
                            dtype=torch.float32):
    '''
    Unproject the image features to form a 3D sparse feature volume

    :param arrange: how features from different view and different types are filled in the voxel. by_type: fill by concat views then by type. by_view: fill by types then by view
    :param coords: coordinates of voxels,
    dim: (batch size, num of voxels per sample, 3) (3: x, y, z)
    :param feats: image features
    dim: (num of views, batch size, C, H, W)
    :param KRcam: projection matrix
    dim: (num of views, batch size, 4, 4)
    :return: feature_volume_all: 3D feature volumes
    dim: (batch size, num of voxels per sample, c + 1)
    :return: count: number of times each voxel can be seen
    dim: (batch size, num of voxels per sample)
    '''
    n_views, bs, c, h, w = feats.shape

    if mode == "cat":
        if plucker_ray is True:
            assert cat_KRcam is False
            assert cat_voxel_cam_offset is False
            ret = torch.zeros([pts.shape[0], (c + 6) * n_views], dtype=dtype, device=pts.device)
        else:
            if cat_KRcam is False:
                ret = torch.zeros([pts.shape[0], c * n_views], dtype=dtype, device=pts.device)
            else:
                if cat_voxel_cam_offset is False:
                    ret = torch.zeros([pts.shape[0], (c + 12) * n_views], dtype=dtype, device=pts.device)
                else:
                    ret = torch.zeros([pts.shape[0], (c + 12 + 3) * n_views], dtype=dtype, device=pts.device)
    elif mode == "mean":
        ret = torch.zeros([pts.shape[0], c], dtype=dtype, device=pts.device)
    else:
        raise NotImplementedError()

    for i in range(bs):
        grid_batch = pts[batch_id == i]
        rs_grid = grid_batch.unsqueeze(0).expand(n_views, -1, -1)
        rs_grid = rs_grid.permute(0, 2, 1).contiguous()
        nV = rs_grid.shape[-1]
        rs_grid = torch.cat([rs_grid, torch.ones([n_views, 1, nV], device="cuda", dtype=dtype)], dim=1)
        # Project grid
        feats_batch = feats[:, i]
        proj_batch = KRcam[:, i]
        if camera_positions is not None:
            camera_position_batch = camera_positions[:, i]

        im_p = proj_batch @ rs_grid
        im_x, im_y, im_z = im_p[:, 0], im_p[:, 1], im_p[:, 2]
        im_x = im_x / im_z
        im_y = im_y / im_z
        if reverse_x:
            raise NotImplementedError()
        im_x = im_x / img_res * w
        im_y = im_y / img_res * h

        im_grid = torch.stack([2 * im_x / (w - 1) - 1, 2 * im_y / (h - 1) - 1], dim=-1)
        mask = im_grid.abs() <= 1
        mask = (mask.sum(dim=-1) == 2) & (im_z > 0)

        feats_batch = feats_batch.view(n_views, c, h, w)
        im_grid = im_grid.view(n_views, 1, -1, 2)

        features = grid_sample(feats_batch, im_grid, padding_mode='zeros', align_corners=True)

        features = features.view(n_views, c, -1)
        mask = mask.view(n_views, -1)
        im_z = im_z.view(n_views, -1)
        # remove nan
        features[mask.unsqueeze(1).expand(-1, c, -1) == False] = 0
        im_z[mask == False] = 0

        # count[batch_ind] = mask.sum(dim=0).float()
        if mode == "cat":
            features = features.reshape(n_views * c, -1)
            features = features.permute(1, 0).contiguous()
            if plucker_ray is True:
                assert cat_KRcam is False
                assert cat_voxel_cam_offset is False
                o = camera_position_batch
                v = grid_batch
                d = v[:, None, :] - o[None, :, :]  # nVoxel, nView, 3
                d = torch.nn.functional.normalize(d, dim=-1)
                o = -o
                oxd = torch.cross(o[None], d, dim=-1)  # nVoxel, nView, 3
                plucker_ray_feat = torch.cat([oxd, d], dim=-1).reshape(nV, -1)
                if arrange == 'by_type':
                    ret[batch_id == i, :features.shape[1]] = features
                    ret[batch_id == i, features.shape[1]:] = plucker_ray_feat
                elif arrange == 'by_view':
                    features_tmp = features.reshape(nV, n_views, 3)
                    plucker_ray_feat_tmp = plucker_ray_feat.reshape(nV, n_views, 6)
                    tmp = torch.cat([features_tmp, plucker_ray_feat_tmp], dim=-1)
                    tmp = tmp.reshape(nV, n_views * 9)
                    ret[batch_id == i] = tmp
                else:
                    raise NotImplementedError()
            else:
                if cat_KRcam is False:
                    ret[batch_id == i] = features
                else:
                    if cat_voxel_cam_offset is False:
                        ret[batch_id == i, :features.shape[1]] = features
                        ret[batch_id == i, features.shape[1]:] = proj_batch[:, :3, :].reshape(-1)[None].repeat(features.shape[0], 1)
                    else:
                        ret[batch_id == i, :features.shape[1]] = features
                        proj_features = proj_batch[:, :3, :].reshape(-1)[None].repeat(features.shape[0], 1)
                        ret[batch_id == i, features.shape[1]:features.shape[1] + proj_features.shape[1]] = proj_features
                        offset = grid_batch[:, None, :] - camera_position_batch[None, :, :]
                        direction = torch.nn.functional.normalize(offset, dim=-1)
                        direction_feat = direction.reshape(nV, -1)
                        ret[batch_id == i, features.shape[1] + proj_features.shape[1]:] = direction_feat
        elif mode == "mean":
            # aggregate multi view
            features = features.sum(dim=0)
            mask = mask.sum(dim=0)
            invalid_mask = mask == 0
            mask[invalid_mask] = 1
            in_scope_mask = mask.unsqueeze(0)
            features /= in_scope_mask
            features = features.permute(1, 0).contiguous()

            ret[batch_id == i] = features
        else:
            raise NotImplementedError()

    return ret


def back_project_sparse_dev_with_transformer(pts, batch_id, feats, KRcam, transformer, img_res=320):
    '''
    Unproject the image features to form a 3D sparse feature volume

    :param coords: coordinates of voxels,
    dim: (batch size, num of voxels per sample, 3) (3: x, y, z)
    :param feats: image features
    dim: (num of views, batch size, C, H, W)
    :param KRcam: projection matrix
    dim: (num of views, batch size, 4, 4)
    :return: feature_volume_all: 3D feature volumes
    dim: (batch size, num of voxels per sample, c + 1)
    :return: count: number of times each voxel can be seen
    dim: (batch size, num of voxels per sample)
    '''
    n_views, bs, c, h, w = feats.shape

    ret = torch.zeros([pts.shape[0], c], dtype=torch.float32, device=pts.device)

    for i in range(bs):
        grid_batch = pts[batch_id == i]
        rs_grid = grid_batch.unsqueeze(0).expand(n_views, -1, -1)
        rs_grid = rs_grid.permute(0, 2, 1).contiguous()
        nV = rs_grid.shape[-1]
        rs_grid = torch.cat([rs_grid, torch.ones([n_views, 1, nV], device="cuda", dtype=torch.float32)], dim=1)
        # Project grid
        feats_batch = feats[:, i]
        proj_batch = KRcam[:, i]

        im_p = proj_batch @ rs_grid
        im_x, im_y, im_z = im_p[:, 0], im_p[:, 1], im_p[:, 2]
        im_x = im_x / im_z
        im_y = im_y / im_z

        im_x = im_x / img_res * w
        im_y = im_y / img_res * h

        im_grid = torch.stack([2 * im_x / (w - 1) - 1, 2 * im_y / (h - 1) - 1], dim=-1)
        mask = im_grid.abs() <= 1
        mask = (mask.sum(dim=-1) == 2) & (im_z > 0)

        feats_batch = feats_batch.view(n_views, c, h, w)
        im_grid = im_grid.view(n_views, 1, -1, 2)

        features = grid_sample(feats_batch, im_grid, padding_mode='zeros', align_corners=True)

        features = features.view(n_views, c, -1)
        mask = mask.view(n_views, -1)
        im_z = im_z.view(n_views, -1)
        # remove nan
        features[mask.unsqueeze(1).expand(-1, c, -1) == False] = 0  # nview,C,nvoxels
        im_z[mask == False] = 0
        fused_features = transformer(features.permute(2, 0, 1))
        # aggregate multi view
        # features = features.sum(dim=0)
        # mask = mask.sum(dim=0)
        # invalid_mask = mask == 0
        # mask[invalid_mask] = 1
        # in_scope_mask = mask.unsqueeze(0)
        # features /= in_scope_mask
        # features = features.permute(1, 0).contiguous()

        ret[batch_id == i] = fused_features
        # else:
        #     raise NotImplementedError()

    return ret


# grid_ = None


@torch.no_grad()
def generate_grid(n_vox, interval, dtype=torch.float32, device='cuda'):
    # global grid_
    # if grid_ is None:
    grid_range = [torch.arange(0, n_vox[axis], interval, device=device) for axis in range(3)]
    grid = torch.stack(torch.meshgrid(grid_range[0], grid_range[1], grid_range[2], indexing='ij'))
    grid = grid.unsqueeze(0).to(dtype=dtype)  # 1 3 dx dy dz
    grid = grid.view(1, 3, -1)
    grid_ = grid
    # else:
    #     pass
    # print("reuse grid")
    return grid_
