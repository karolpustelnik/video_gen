from src.training.networks import *
from copy import deepcopy
from src.training.tsm import TemporalShift
import torch

def make_block_temporal(stage, this_segment):
                blocks = list(stage.children())
                print('=> Processing stage with {} blocks'.format(len(blocks)))
                for i, b in enumerate(blocks):
                    print(b)
                    blocks[i] = TemporalShift(b, n_segment=this_segment, n_div=3)
                return torch.nn.Sequential(*(blocks))

class DotDict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __deepcopy__(self, memo=None):
        return DotDict(deepcopy(dict(self), memo=memo))


cfg = {'num_frames_div_factor': 2, 'concat_res': 16, 'sampling': {'num_frames_per_video':8, 'max_num_frames': 1024}}
cfg = DotDict(cfg)
cfg.sampling = DotDict(cfg.sampling)
#discrimiminator = Discriminator(c_dim = 1, img_resolution = 256, img_channels = 3, cfg = cfg)
disc_block = DiscriminatorBlock(0, 64, 64, 128, 3, 0, cfg = cfg)
print(disc_block)

test_tensor = torch.rand(48, 3, 128, 128)
disc_block(None, test_tensor)
#print(disc_tsm(test_tensor, test_tensor))