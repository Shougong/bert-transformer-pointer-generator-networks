import torch
from torch.autograd import Variable
from torch.nn import Module, KLDivLoss, NLLLoss


# noinspection PyTypeChecker,PyArgumentList
class LabelSmoothing(Module):

    def __init__(self, size, padding_idx, smoothing=0.1):
        super(LabelSmoothing, self).__init__()
        self.criterion = KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist

        loss = self.criterion(x, Variable(true_dist, requires_grad=False))
        pred = torch.argmax(x, dim=-1)
        
        _, correct, _ = get_token_accuracy(target.cpu().detach(), pred.cpu().detach(), ignore_index=self.padding_idx)

        return loss, correct


class NLL(Module):

    def __init__(self, padding_idx):
        super(NLL, self).__init__()
        self.padding_idx = padding_idx
        self.criterion = NLLLoss(ignore_index=padding_idx, reduction='sum')

    def forward(self, x, target):
        loss = self.criterion(x, target)
        pred = torch.argmax(x, dim=-1)
        _, correct, _ = get_token_accuracy(target.cpu().detach(), pred.cpu().detach(), ignore_index=self.padding_idx)
        return loss, correct


class SimpleLossCompute:

    def __init__(self, criterion, opt=None, accumulation=4):
        self.criterion = criterion
        self.opt = opt
        self.accumulation = accumulation

    def __call__(self, x, y, step, norm):
        loss, correct = self.criterion(x.reshape(-1, x.size(-1)), y.reshape(-1))
        loss = loss / (norm * self.accumulation)

        if self.opt is not None:
            loss.backward()

            if step % self.accumulation == 0:
                self.opt.step()
                self.opt.optimizer.zero_grad()

        return loss.item() * norm * self.accumulation, correct