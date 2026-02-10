import torch

class SPLTarget:
    """
    Self-Paced Learning target baseado em:
    - histórico temporal de loss
    - variância (CV)
    - EMA de thresholds
    """

    def __init__(self, ema_alpha=0.9):
        self.ema_alpha = ema_alpha
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.running_var_threshold = None
        self.running_loss_threshold = None

    @torch.no_grad()
    def compute(self, hist_seq, current_losses, epoch):
        loss_history = hist_seq[:, :, 0]  

        mean = loss_history.mean(dim=1)
        std = loss_history.std(dim=1)

        eps = 1e-3
        cv = std / (mean + eps)
        cv = torch.clamp(cv, max=5.0)

        sample_vars = torch.log1p(cv)

        var_percentile = min(0.9, 0.6 + epoch * 0.01)
        loss_percentile = min(0.9, 0.6 + epoch * 0.01)

        batch_var_thresh = torch.quantile(sample_vars, var_percentile)
        batch_loss_thresh = torch.quantile(
            current_losses.detach(),
            loss_percentile
        )

        tau_loss = 0.5 * torch.std(current_losses).detach()
        tau_var = 0.5 * torch.std(sample_vars).detach()

        tau_loss = torch.clamp(tau_loss, min=1e-6)
        tau_var = torch.clamp(tau_var, min=1e-6)

        if self.running_var_threshold is None:
            self.running_var_threshold = batch_var_thresh
            self.running_loss_threshold = batch_loss_thresh
        else:
            self.running_var_threshold = (
                self.ema_alpha * self.running_var_threshold
                + (1 - self.ema_alpha) * batch_var_thresh
            )

            self.running_loss_threshold = (
                self.ema_alpha * self.running_loss_threshold
                + (1 - self.ema_alpha) * batch_loss_thresh
            )

        v_var = torch.sigmoid(
            (self.running_var_threshold - sample_vars) / (tau_var + 1e-8)
        )

        v_loss = torch.sigmoid(
            (self.running_loss_threshold - current_losses.detach())
            / (tau_loss + 1e-8)
        )

        v_star = v_var * v_loss

        return v_star.float().view(-1, 1).to(current_losses.device)
