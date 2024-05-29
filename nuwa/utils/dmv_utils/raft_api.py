import os.path

import numpy as np
import torch
from PIL import Image
from easydict import EasyDict

from nuwa.models.flow.raft.raft import RAFT
from nuwa.models.flow.raft.utils.utils import InputPadder

DEVICE = 'cuda'


def load_image(imfile):
    if isinstance(imfile, str):
        imfile = Image.open(imfile)
    img = np.array(imfile).astype(np.uint8)[..., :3]
    img = torch.from_numpy(img).permute(2, 0, 1).float()[None].to(DEVICE)
    return img


def setup_model(args):
    model = RAFT(args)
    ckpt = torch.load(args.model)
    ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
    model.load_state_dict(ckpt)

    model.to(DEVICE)
    model.eval()
    return model


class RAFTAPIHelper:
    _model = None
    _ckpt = None

    @staticmethod
    def get_instance(ckpt=None, small=False, iters=32, mixed_precision=False):
        if ckpt is None:
            ckpt = os.path.expanduser("~/.cache/raft-sintel.pth")
        if RAFTAPIHelper._model is None:
            model = setup_model(EasyDict({'model': ckpt, 'small': small,
                                          'iters': iters, 'mixed_precision': mixed_precision}))
            RAFTAPIHelper._model = model
        return RAFTAPIHelper._model


@torch.no_grad()
def raft_optical_flow_api(frame0, frame1, ckpt=None, small: bool = False, iters: int = 32, mixed_precision=False):
    """

    :param frame0: np.ndarray h,w,3 uint8
    :param frame1: np.ndarray h,w,3 uint8
    :param ckpt: https://drive.google.com/file/d/1fubTHIa_b2C8HqfbPtKXwoRd9QsYxRL6/view?usp=drive_link
    :param small:
    :param iters:
    :return:
    """

    image1 = load_image(frame0)
    image2 = load_image(frame1)
    padder = InputPadder(image1.shape)
    image1, image2 = padder.pad(image1, image2)
    model = RAFTAPIHelper.get_instance(ckpt=ckpt, small=small, iters=iters, mixed_precision=mixed_precision)
    flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
    # flow = flow_predictions[-1]
    # ph0, ph1 = image1_pad_dict['pad_ht1']
    # pw0, pw1 = image1_pad_dict['pad_wd1']
    # if ph0 + ph1 != 0:
    #     flow = flow[:, :, ph0:-ph1]
    # if pw0 + pw1 != 0:
    #     flow = flow[:, :, :, pw0:-pw1]
    # assert flow.shape[2:] == frame0.size[::-1]
    # zarr.save(output_path, flow.squeeze(0).permute(1, 2, 0).cpu().numpy())
    flow_up = padder.unpad(flow_up)
    flow_up = flow_up.squeeze(0).permute(1, 2, 0)
    return flow_up
