import torch
import torch.nn as nn
import torch.nn.functional as F


def contrastive_accuracy(sim_matrix):
    """Contrastive accuracy estimate"""
    batch_size = sim_matrix.shape[0]
    r = torch.arange(batch_size, device=sim_matrix.device)

    contr_acc_1 = (sim_matrix.argmax(0) == r).sum() / batch_size
    contr_acc_2 = (sim_matrix.argmax(1) == r).sum() / batch_size
    contr_acc = (contr_acc_1.float() + contr_acc_2.float()) / 2
    return contr_acc


def contrastive_top_one_accuracy_with_ids(similarity_matrix, speaker_ids):
    """Compute the proportion of rows of twin_sim_matrix where the most similar off-diagonal
       example has the same speaker ID as the example on the diagonal.
    """
    assert similarity_matrix.dim() == 2
    assert speaker_ids.dim() == 1
    assert similarity_matrix.size(0) == similarity_matrix.size(1)

    sim_mx_masked = torch.clone(similarity_matrix)
    r = torch.arange(sim_mx_masked.size(0), device=similarity_matrix.device)
    sim_mx_masked[r, r] = -torch.inf

    where_max = sim_mx_masked.argmax(1)
    ids_max = speaker_ids[where_max]
    ids_gt = speaker_ids

    res = (ids_max == ids_gt).float().mean()

    return res


class NTXentLoss(nn.Module):
    """Used in SimCLR. Implementation adapted from:
    https://github.com/dhruvbird/ml-notebooks/tree/main/nt-xent-loss (MIT License)

    Added option of applying margin to positive pairings.

    Moved temp scaling before self-similarity exlusion to allow
    for trainable temperature (scaling by matrix with -inf resulted
    in temperature=NaN).
    """

    def __init__(
        self,
        temperature: float = 0.1,
        learn_temperature: bool = False,
        margin: float = 0,
    ):
        super().__init__()
        if learn_temperature:
            self.temperature = nn.Parameter(torch.tensor(temperature))
        else:
            self.temperature = torch.tensor(temperature)

        self.margin = margin

    def forward(self, emb_a, emb_b):
        # emb_a and emb_b have augmentations of the same sample
        # at corresponding indices
        assert (
            emb_a.shape == emb_b.shape
        ), "Corresponding embs must have identical shapes"
        batch_size = emb_a.shape[0]
        # concatenate embs to maximize pair count (symmetric loss)
        emb_ab = torch.cat((emb_a, emb_b), dim=0)

        # calculate cosine similarity: normalize + matmul
        emb_ab_norm = F.normalize(emb_ab, dim=-1)
        sim_matrix = torch.mm(emb_ab_norm, emb_ab_norm.T)
        # apply temperature scaling
        scaled_sim_matrix = sim_matrix / self.temperature

        # exclude self-similarity
        self_exclusion_mask = torch.eye(2 * batch_size, device=sim_matrix.device).bool()
        sim_matrix_ex = scaled_sim_matrix.masked_fill(
            self_exclusion_mask, float("-inf")
        )

        # subtract margin from positive sample similarities
        if self.margin > 0:
            pos_indices = torch.arange(2 * batch_size)
            pos_indices = (pos_indices, torch.roll(pos_indices, batch_size))
            sim_matrix_ex[pos_indices] -= self.margin

        # Ground truth labels
        target = torch.arange(2 * batch_size, device=sim_matrix.device)
        target = torch.roll(target, batch_size)

        loss = F.cross_entropy(sim_matrix_ex, target, reduction="mean")

        with torch.no_grad():
            contr_acc = contrastive_accuracy(sim_matrix[:batch_size, batch_size:])

            metrics = {
                "loss": float(loss.item()),
                "contr_acc": contr_acc,
                "loss_temperature": self.temperature.item(),
            }

        return loss, metrics


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    Adapted from: https://github.com/ivanpanshin/SupCon-Framework/blob/main/tools/losses.py (MIT License)
    """
    def __init__(self, temperature: float = 0.1, learn_temperature: bool = False, margin: float = 0.0):
        super().__init__()
        if learn_temperature:
            self.temperature = nn.Parameter(torch.tensor(temperature))
        else:
            # Register as buffer so it moves with the module to the correct device
            self.register_buffer('temperature', torch.tensor(temperature))

        self.margin = margin

    def forward(self, embd_a, embd_b, labels=None):
        """Compute loss for model. If both `speaker_ids` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            speaker_ids: ground truth of shape [bsz].
        Returns:
            A loss scalar.
        """
        device = embd_a.device
        features = torch.stack([embd_a, embd_b], dim=1)

        if len(features.shape) != 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')

        batch_size = features.shape[0]

        if labels is not None and labels.numel() > 0:
            speaker_ids = labels.contiguous().view(-1, 1)
            if speaker_ids.shape[0] != batch_size:
                raise ValueError('Num of speaker_ids does not match num of features')
            mask = torch.eq(speaker_ids, speaker_ids.T).float().to(device)
        else:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)

        contrast_count   = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        contrast_feature = F.normalize(contrast_feature, dim=1)

        anchor_feature = contrast_feature
        anchor_count = contrast_count

        # compute logits
        pairwise_cos = torch.matmul(anchor_feature, contrast_feature.T)

        if self.margin > 0:
            r = torch.arange(2 * batch_size, device=device)
            pairwise_cos[r, (r + batch_size) % (2 * batch_size)] -= self.margin

        anchor_dot_contrast = pairwise_cos / self.temperature

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        logits = torch.clamp(logits, min=-50.0, max=50.0)  # prevent exp overflow/underflow

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        # Use indexing instead of scatter to avoid CUDA engine issues
        logits_mask = torch.ones_like(mask)
        diag_indices = torch.arange(batch_size * anchor_count, device=device)
        logits_mask[diag_indices, diag_indices] = 0
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True).clamp(min=1e-8))

        # compute mean of log-likelihood over positive
        mask_sum = mask.sum(1)
        # Avoid division by zero - clamp to prevent numerical issues
        mask_sum = torch.clamp(mask_sum, min=1e-8)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum

        # loss
        loss = -1 * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        with torch.no_grad():
            if labels is not None and labels.numel() > 0:
                unique_spk_ids, unique_counts = torch.unique(labels, return_counts=True)
                num_repeating_speakers = torch.sum(unique_counts > 1)
                # contrastive_accuracy() works on two matched batches so a subset of the combined similarity matrix
                # is required
                contr_acc = contrastive_top_one_accuracy_with_ids(anchor_dot_contrast, labels.repeat(2))
                contr_norm_acc = contr_acc ** (1 / (batch_size - 1))
            else:
                num_repeating_speakers = torch.tensor(0, device=device)
                contr_acc = torch.tensor(0.0, device=device)
                contr_norm_acc = torch.tensor(0.0, device=device)

            temp_value = self.temperature.detach() if isinstance(self.temperature, torch.Tensor) else torch.tensor(self.temperature, device=device)
            metrics = {
                'loss': float(loss.detach()),
                'contr_acc': float(contr_acc.detach()) if isinstance(contr_acc, torch.Tensor) else contr_acc,
                'contr_norm_acc': float(contr_norm_acc.detach()) if isinstance(contr_norm_acc, torch.Tensor) else contr_norm_acc,
                'contr_error_rate': 1.0 - (float(contr_acc.detach()) if isinstance(contr_acc, torch.Tensor) else contr_acc),
                'loss_temperature': float(temp_value.item()) if isinstance(temp_value, torch.Tensor) else temp_value,
                'loss_num_repeating_speakers': float(num_repeating_speakers.detach()) if isinstance(num_repeating_speakers, torch.Tensor) else num_repeating_speakers,
            }

        return loss, metrics
