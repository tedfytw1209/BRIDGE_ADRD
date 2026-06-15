import torch
import config
import torch.nn.functional as F


def multimodal_data_augmentation(x_dict, feat_mask_dict, A_hat):
    """
    Augmented view (G_tilde) using feature masking and edge dropping.
    Feature masking: X_tilde = [x^1 ⊙ R^1, ..., x^M ⊙ R^M]
    Edge dropping: A_tilde = R^a ⊙ A_hat
    """
    cfg = config.config
    prob_edge = cfg["prob_edge"]
    categorical_modalities = set(cfg['categorical_modalities'])

    x_tilde_dict = {}
    feat_mask_tilde_dict = {}
    R_dict = {}

    # Feature masking: X_tilde = X ⊙ R
    for m in sorted(x_dict.keys()):
        x_m = x_dict[m]
        feat_mask = feat_mask_dict[m]
        prob_key = f"prob_{m}"
        prob = cfg[prob_key]

        # keep mask: 1=keep, 0=hide
        if m in categorical_modalities:
            # Row-wise masking: mask entire one-hot vector per patient
            patient_keep = torch.bernoulli(torch.ones(x_m.size(0), 1, device=x_m.device) * (1 - prob))
            keep = patient_keep.expand_as(x_m)
        else:
            keep = torch.bernoulli(torch.ones_like(x_m) * (1 - prob))
        R_m = (1 - keep) * feat_mask  # R^m: tracks which OBSERVED features were masked
        x_tilde_m = x_m * keep
        feat_mask_tilde = feat_mask * keep

        x_tilde_dict[m] = x_tilde_m
        feat_mask_tilde_dict[m] = feat_mask_tilde
        R_dict[m] = R_m

    # Edge dropping: A_tilde = R^a ⊙ A_hat
    R_a = torch.bernoulli(torch.ones_like(A_hat) * (1 - prob_edge))
    A_tilde = A_hat * R_a

    return x_tilde_dict, feat_mask_tilde_dict, A_tilde, R_dict


def contrastive_similarity(z1, z2, normalize=True):
    """Normalized dot product similarity."""
    if normalize:
        z1, z2 = F.normalize(z1, dim=-1), F.normalize(z2, dim=-1)
    return z1 @ z2.T


def compute_cl_loss(z_anchor, A_anchor, S_same, S_cross, eps=1e-8):
    """Single direction contrastive loss (anchor vs same-view and cross-view neighbors)."""
    K = (A_anchor > 0).float()

    # Positives: self + same-view neighbors + cross-view neighbors
    pos_self = torch.diagonal(S_cross)
    pos_same = (S_same * K).sum(dim=1)
    pos_cross = (S_cross * K).sum(dim=1)
    numerator = pos_self + pos_same + pos_cross

    # Denominator: all pairs except self-self in same view
    denominator = S_same.sum(dim=1) + S_cross.sum(dim=1) - torch.diagonal(S_same)

    # Number of positives: 2|K| + 1
    num_pos = 2 * K.sum(dim=1) + 1

    loss = -torch.log(torch.clamp(numerator, min=eps) / torch.clamp(denominator, min=eps))
    loss = loss / torch.clamp(num_pos, min=1.0)
    loss = loss[torch.isfinite(loss)]

    return loss.mean() if loss.numel() > 0 else torch.tensor(0.0, device=z_anchor.device, requires_grad=True)


def contrastive_loss(z, z_tilde, A_hat, A_tilde):
    """
    Bidirectional Graph Contrastive Loss (scSimGCL style).
    L = alpha1 * L_forward + alpha2 * L_backward
    """
    cfg = config.config
    tau = cfg["tau"]
    alpha1 = cfg["alpha1"]
    alpha2 = cfg["alpha2"]

    # Similarity matrices
    f = lambda x: torch.exp(x / tau)
    S_zz = f(contrastive_similarity(z, z))              # original-original
    S_tt = f(contrastive_similarity(z_tilde, z_tilde))  # aug-aug
    S_zt = f(contrastive_similarity(z, z_tilde))        # cross-view

    # Forward: z as anchor (uses A_hat for neighbors)
    L_fwd = compute_cl_loss(z, A_hat, S_zz, S_zt)

    # Backward: z_tilde as anchor (uses A_tilde for neighbors)
    L_bwd = compute_cl_loss(z_tilde, A_tilde, S_tt, S_zt.T)

    return alpha1 * L_fwd + alpha2 * L_bwd


def compute_imputation_loss(x_dict, x_hat_dict, R_dict, categorical_modalities=None):
    """
    Imputation Loss (L^imputation).
    - Continuous modalities: L1 loss on masked positions
    - Categorical modalities: cross-entropy on masked patients
    """
    categorical_modalities = set(categorical_modalities or [])

    device = next(iter(x_dict.values())).device
    total_loss = 0.0
    masked_elements = 0

    for m in sorted(R_dict.keys()):
        x_m = x_dict[m]
        x_hat_m = x_hat_dict[m]
        R_m = R_dict[m]

        if m in categorical_modalities:
            # Cross-entropy for single-label categorical modalities
            patient_masked = R_m.sum(dim=1) > 0
            count = patient_masked.sum()
            if count == 0:
                continue
            x_m_safe = torch.where(torch.isfinite(x_m), x_m, torch.zeros_like(x_m))
            targets = x_m_safe[patient_masked].argmax(dim=-1)
            logits = x_hat_m[patient_masked]
            total_loss += F.cross_entropy(logits, targets, reduction='sum')
            masked_elements += count
        else:
            # L1 for continuous modalities
            count = R_m.sum()
            masked_elements += count
            if count == 0:
                continue
            x_m_safe = torch.where(torch.isfinite(x_m), x_m, torch.zeros_like(x_m))
            total_loss += torch.abs((x_hat_m - x_m_safe) * R_m).sum()

    if masked_elements == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    return total_loss / masked_elements


def compute_joint_loss(L_gcl, L_imp, lambda_imp, lambda_cl=1.0):
    return lambda_cl * L_gcl + lambda_imp * L_imp