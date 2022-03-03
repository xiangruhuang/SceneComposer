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
  
  def forward(self, output, mask, ind, target):
    pred = _transpose_and_gather_feat(output, ind)
    mask = mask.float().unsqueeze(2) 

    loss = F.l1_loss(pred*mask, target*mask, reduction='none')
    loss = loss / (mask.sum() + 1e-4)
    loss = loss.transpose(2 ,0).sum(dim=2).sum(dim=1)
    return loss

class RegLoss2(nn.Module):
  '''Regression loss for an output tensor
    Arguments:
      output (batch x dim x h x w)
      mask (batch x max_objects)
      ind (batch x max_objects)
      target (batch x max_objects x dim)
  '''
  def __init__(self):
    super(RegLoss2, self).__init__()
  
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

  def forward(self, out, target, ind, mask, cat):
    '''
    Arguments:
      out, target: B x C x H x W
      ind, mask: B x M
      cat (category id for peaks): B x M
    '''
    mask = mask.float()
    gt = torch.pow(1 - target, 4)
    neg_loss = torch.log(1 - out) * torch.pow(out, 2) * gt
    neg_loss = neg_loss.sum()

    pos_pred_pix = _transpose_and_gather_feat(out, ind) # B x M x C
    pos_pred = pos_pred_pix.gather(2, cat.unsqueeze(2)) # B x M
    num_pos = mask.sum()
    pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2) * \
               mask.unsqueeze(2)
    pos_loss = pos_loss.sum()
    if num_pos == 0:
      return - neg_loss
    return - (pos_loss + neg_loss) / num_pos

class FastFocalLoss2(nn.Module):
  '''
  Reimplemented focal loss, exactly the same as the CornerNet version.
  Faster and costs much less memory.
  '''
  def __init__(self):
    super(FastFocalLoss2, self).__init__()

  def forward(self, out, target, ind, batch, neg_ind, neg_batch, cat):
    '''
    Arguments:
      out, target: B x C x H x W
      ind, mask: B x M
      cat (category id for peaks): B x M
    '''
    #gt = torch.pow(target, 4)

    out_neg = _transpose_and_gather_feat_by_batch(out, neg_ind, neg_batch)

    neg_loss = torch.log(1 - out_neg) * torch.pow(out_neg, 2)
    neg_loss = neg_loss.sum()

    # [N, C]
    pos_loss = torch.log(out) * torch.pow(1 - out, 2) * target
    #pos_pred_pix = _transpose_and_gather_feat_by_batch(out, ind, batch)
    #pos_pred = pos_pred_pix.gather(1, cat.unsqueeze(2)) # B x M

    num_pos = ind.shape[0]
    #pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2) * \
    #           mask.unsqueeze(2)
    pos_loss = pos_loss.sum()
    if num_pos == 0:
      return - neg_loss
    return - (pos_loss + neg_loss) / num_pos
