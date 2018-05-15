def compute_score_loss(logits, targets, train_match):
    """
    logits, shape [N, num_all_db, 8]
    targets, shape [N, num_all_db, 8]
    train_match，shape [batch_size, num_default_boxes, 4], 这部分有点问题，一个box可能会预测4个值，所以会有4个match,如果box会预测两个
    值，我们也在一次训练中也随机取一个位置的进行训练
    注意，这里如果有个box是-1,那么则有4个值都是-1，但是为了计算损失函数，我们只取一个值为-1，其他的值设置为0，在每次训练中，应当随机选择
    一个值为-1【0， 1，2， 3】
    
    每个batch_size的match可能不一样，这里主要是说的数量，因为某些可能数量不够，但我们保持一个比例,
    假设返回正样本为1，负样本为-1， 不计算损失函数的为0,利用+1和-1来计算损失函数
    """
    logits = logits.view(-1, 2)
    targets = targets.view(-1, 2)
    targets = torch.nonzero(targets)[:, 1].long()
    
    train_match = train_match.view(-1, 1)
    # 这里负样本也需要计算损失函数
    train_match[train_match == -1] = 1
    train_match = train_match.byte()
    
    logits = torch.mask_selected(logits, mask=train_match).view(-1, 2)
    targets = torch.mask_selected(targets, mask=train_match).view(-1)
    
    loss = F.cross_entropy(input=logits, target=targets)
    
    return loss


def compute_offset_loss(offsets, targets, train_match):
    """
    offsets, shape [N, num_all_db, 16]
    targets, shape [N, ]
    train_match， shape [N, num_all_db, 4]
    """
    
    offsets = offsets.view(-1, 4)
    # todo 这里需要对输入做些限制
    targets = targets.view(-1, 4)
    train_match = train_match.view(-1, 1)
    
    offsets = torch.mask_selected(offsets, mask=train_match).view(-1, 4)
    
    assert offsets.size() == targets.size()
    
    loss = F.smooth_l1_loss(input=offsets, target=targets)
    
    return loss

def compute_seg_loss(segs, targets):
    
    """
    segs: [4, h, w], 预测的值，应该是要经过一个sigmoid的
    targets: [4, h , w] 是某层的mask，则该值为1
    """
    
    
