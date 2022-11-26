

def dice_loss(prediction, target):
    smooth = 1.0
    i_flat = prediction.view(-1)
    t_flat = target.view(-1)
    intersection = (i_flat * t_flat).sum()
    return 1 - ((2. * intersection + smooth) / (i_flat.sum() + t_flat.sum() + smooth))

