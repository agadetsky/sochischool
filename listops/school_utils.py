import torch


def compute_metrics(sample, arcs, lengths):
    one = torch.tensor(1.0).cuda() if sample.is_cuda else torch.tensor(1.0)
    zero = torch.tensor(0.0).cuda() if sample.is_cuda else torch.tensor(0.0)
    # Compute true/false positives/negatives for metric calculations.
    maxlen = arcs.shape[-1]
    pad_tn = maxlen - lengths
    tp = torch.where(sample * arcs == 1.0, one, zero).sum((-1, -2))
    tn = torch.where(sample + arcs == 0.0, one, zero).sum((-1, -2)) - pad_tn
    fp = torch.where(sample - arcs == 1.0, one, zero).sum((-1, -2))
    fn = torch.where(sample - arcs == -1.0, one, zero).sum((-1, -2))

    # Calculate precision (attachment).
    precision = torch.mean(tp / (tp + fp)).cpu()
    # Calculate recall.
    recall = torch.mean(tp / (tp + fn)).cpu()

    return precision, recall