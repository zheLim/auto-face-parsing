import torch


class OhemLoss(torch.nn.Module):
    def __init__(self, num_classes, ohem_thresh=0.9, batch_size=None, width=None, min_keep=None, epsilon=1e-7, gamma=2.0):
        super(OhemLoss, self).__init__()

        if batch_size is not None and width is not None:
            self.min_keep = width ** 2 // 16

        self.register_buffer('num_classes', torch.IntTensor(num_classes))
        self.register_buffer('epsilon', torch.tensor(epsilon))
        self.register_buffer('threshold', -torch.log(torch.tensor(ohem_thresh)))

        self.ce_loss = torch.nn.CrossEntropyLoss(reduction='none')

    def __call__(self, logits, labels):
        ohem_loss = self.ce_loss(logits, labels)

        floss = self.online_hard_example_mining(ohem_loss)
        return torch.mean(floss)

    def online_hard_example_mining(self, losses):
        losses_sorted, _ = torch.sort(torch.reshape(losses, [-1]), descending=True)
        if losses_sorted[self.min_keep] < self.threshold:
            return losses_sorted[self.min_keep]
        return losses_sorted[losses_sorted > self.threshold]