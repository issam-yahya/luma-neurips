"""
LUMA -- low-rank cross-period forecaster.

Architecture (per sequence):
    1. mean-normalize
    2. cross-period 1D-conv aggregator (residual)
    3. reshape into (period_len)-length segments
    4. low-rank linear map across segments: W = U diag(s) V^T
    5. reshape back, crop to pred_len, denormalize

The only two structural knobs are `period_len` (P) and `rank` (R).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LowRankMap(nn.Module):
    """Low-rank linear map on the last dim: y = x V diag(s) U^T (no bias)."""

    def __init__(self, in_features: int, out_features: int, rank: int = 4,
                 init: str = "orthogonal"):
        super().__init__()
        r = max(1, min(rank, in_features, out_features))
        self.in_features = in_features
        self.out_features = out_features
        self.rank = r

        self.U = nn.Parameter(torch.empty(out_features, r))
        self.s = nn.Parameter(torch.empty(r))
        self.V = nn.Parameter(torch.empty(in_features, r))
        self.reset_parameters(init)

    def reset_parameters(self, init: str = "orthogonal"):
        if init == "orthogonal":
            nn.init.orthogonal_(self.U)
            nn.init.orthogonal_(self.V)
        else:
            nn.init.xavier_uniform_(self.U)
            nn.init.xavier_uniform_(self.V)
        nn.init.uniform_(self.s, a=0.01, b=0.05)

    @property
    def weight(self):
        return self.U @ (self.s.unsqueeze(0) * self.V.T)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x @ self.V
        y = y * self.s
        y = y @ self.U.T
        return y


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.enc_in = int(configs.enc_in)
        self.period_len = int(getattr(configs, 'period_len', 24))
        self.task_name = getattr(configs, 'task_name', 'long_term_forecast')

        # reconstruction (AD / imputation / pred_len==0) keeps seg_num_y = seg_num_x
        self._reconstruction = (
            self.task_name in ('anomaly_detection', 'imputation')
            or self.pred_len == 0
        )
        self._classification = (self.task_name == 'classification')

        self.seg_num_x = math.ceil(self.seq_len / self.period_len)
        self.seg_num_y = (
            self.seg_num_x if self._reconstruction
            else max(1, math.ceil(self.pred_len / self.period_len))
        )

        # cross-period aggregator (residual)
        self.conv1d = nn.Conv1d(
            in_channels=1, out_channels=1,
            kernel_size=1 + 2 * (self.period_len // 2),
            stride=1, padding=self.period_len // 2,
            padding_mode="zeros", bias=False,
        )

        rank = int(getattr(configs, 'rank', 2))
        self.lowrank = LowRankMap(self.seg_num_x, self.seg_num_y, rank=rank,
                                  init="other")

        print(f"[LUMA] period_len={self.period_len}  rank={rank}  "
              f"seg_x={self.seg_num_x}  seg_y={self.seg_num_y}  "
              f"task={self.task_name}")

    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        B = x.shape[0]

        # mean-normalize per sequence; (B,T,C) -> (B,C,T)
        seq_mean = torch.mean(x, dim=1, keepdim=True)
        x = (x - seq_mean).permute(0, 2, 1)
        Tcur = x.size(-1)

        # residual 1D-conv aggregation
        x = self.conv1d(x.reshape(-1, 1, Tcur)).reshape(B, self.enc_in, Tcur) + x

        # right-pad time to a multiple of period_len
        T_pad = self.seg_num_x * self.period_len
        pad_right = T_pad - x.size(-1)
        if pad_right > 0:
            x = F.pad(x, (0, pad_right), mode='constant', value=0.0)

        # (B, C, T_pad) -> (B*C, period_len, seg_num_x)
        x = x.reshape(-1, self.seg_num_x, self.period_len).permute(0, 2, 1)

        # low-rank cross-period predictor: seg_num_x -> seg_num_y
        y = self.lowrank(x)

        # (B*C, period_len, seg_num_y) -> (B, C, seg_num_y * period_len)
        y = y.permute(0, 2, 1).reshape(B, self.enc_in, self.seg_num_y * self.period_len)

        if self._classification:
            return y.mean(dim=-1)                    # (B, C) global pooled feature

        target_T = self.seq_len if self._reconstruction else self.pred_len
        y = y[..., :target_T]
        y = y.permute(0, 2, 1) + seq_mean            # (B, T, C) + denorm
        return y
