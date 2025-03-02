import torch
import torch.nn as nn


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        num = targets.size(0)
        #print(logits.shape)
        #print(targets.shape)

        probs = torch.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        #print("\nm1:",m1.shape)
        #print("m2:", m2.shape)
        intersection = m1 * m2

        score = (
            2.0
            * (intersection.sum(1) + self.smooth)
            / (m1.sum(1) + m2.sum(1) + self.smooth)
        )
        score = 1 - score.sum() / num
        return score

