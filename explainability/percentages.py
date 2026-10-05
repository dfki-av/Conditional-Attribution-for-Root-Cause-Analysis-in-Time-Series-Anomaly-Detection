"""Feature contribution normalization used by the original experiment."""
import torch


def shap_percentages_clip_whole(shap_anom):
        """
        Method B: Convert SHAP values to % contributions by clipping negatives to zero.
        shap_anom: 1D torch.Tensor of shape (feats,)
        returns: torch.Tensor of shape (feats,)
        """
        clipped = torch.clamp(shap_anom, min=0.0)  # shape: (feats,)
        total_clipped = torch.sum(clipped) + 1e-9   # scalar sum
        perc_clip = clipped / total_clipped         # shape: (feats,)
        return (perc_clip*100)


def shap_percentages_clip_window(shap_anom_per_window: torch.Tensor) -> torch.Tensor:
    """
    Convert SHAP values to % contributions per window by clipping negatives to zero.
    shap_anom_per_window: (N, feats)
    returns: (N, feats) where each row sums to 100
    """
    clipped = torch.clamp(shap_anom_per_window, min=0.0)     # (N, feats)
    total = clipped.sum(dim=1, keepdim=True) + 1e-9          # (N, 1)
    perc = clipped / total                                   # (N, feats)
    return perc * 100
