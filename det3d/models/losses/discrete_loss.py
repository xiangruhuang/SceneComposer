import torch
import torch.nn as nn
import torch.nn.functional as F
from det3d.core.utils.center_utils import (
    _transpose_and_gather_feat,
    _transpose_and_gather_feat_by_batch
)

class RegLoss(nn.Module):
  '''Regression loss for an output tensor
    Arguments:
      output (batch x dim x h x w)
      mask (batch x max_objects)
      ind (batch x max_objects)
      target (batch x max_objects x dim)
  '''
  def __init__(self):
    super(RegLoss, self).__init__()
  
  def forward(self, output, target, ind, batch):
    pred = _transpose_and_gather_feat_by_batch(output, ind, batch)

    loss = F.l1_loss(pred, target, reduction='none')
    loss = loss / (ind.shape[0] + 1e-4)
    loss = loss.sum(dim=0)
    return loss

class FastFocalLoss(nn.Module):
  '''
  Reimplemented focal loss, exactly the same as the CornerNet version.
  Faster and costs much less memory.
  '''
  def __init__(self):
    super(FastFocalLoss, self).__init__()

  def forward(self, out, target, ind, batch, cat):
    '''
    Arguments:
      out, target: B x C x H x W
      ind, batch: N_pos
      cat (category id for peaks): N_pos 
    '''
    gt = torch.pow(1 - target, 4)
    neg_loss = torch.log(1 - out) * torch.pow(out, 2) * gt
    neg_loss = neg_loss.sum()

    # [N_pos, N_class]
    pos_pred_pix = _transpose_and_gather_feat_by_batch(out, ind, batch)
    pos_pred = pos_pred_pix.gather(1, cat.unsqueeze(1).long())

    num_pos = ind.shape[0]
    pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
    pos_loss = pos_loss.sum()
    if num_pos == 0:
      return - neg_loss
    return - (pos_loss + neg_loss) / num_pos
