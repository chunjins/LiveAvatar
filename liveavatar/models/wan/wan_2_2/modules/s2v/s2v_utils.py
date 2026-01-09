# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import numpy as np
import torch
from ....inference_utils import conditional_compile

# @conditional_compile
def rope_precompute(x, grid_sizes, freqs, start=None):
    b, s, n, c = x.size(0), x.size(1), x.size(2), x.size(3) // 2

    # split freqs
    if type(freqs) is list:
        trainable_freqs = freqs[1]
        freqs = freqs[0]
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    # loop over samples
    output = torch.view_as_complex(x.detach().to(torch.float16).reshape(b, s, n, -1,
                                                      2))
    seq_bucket = [0]
    if not type(grid_sizes) is list:
        grid_sizes = [grid_sizes]
    for g in grid_sizes:
        if not type(g) is list:
            g = [torch.zeros_like(g), g]
        batch_size = g[0].shape[0]
        for i in range(batch_size):
            if start is None:
                f_o, h_o, w_o = g[0][i]
            else:
                f_o, h_o, w_o = start[i]

            f, h, w = g[1][i]
            t_f, t_h, t_w = g[2][i]
            seq_f, seq_h, seq_w = f - f_o, h - h_o, w - w_o
            seq_len = int(seq_f * seq_h * seq_w)
            if seq_len > 0:
                if t_f > 0:
                    factor_f, factor_h, factor_w = (t_f / seq_f).item(), (
                        t_h / seq_h).item(), (t_w / seq_w).item()
                    # Generate a list of seq_f integers starting from f_o and ending at math.ceil(factor_f * seq_f.item() + f_o.item())
                    if f_o >= 0:
                        f_sam = np.linspace(f_o.item(), (t_f + f_o).item() - 1,
                                            seq_f).astype(int).tolist()
                    else:
                        f_sam = np.linspace(-f_o.item(),
                                            (-t_f - f_o).item() + 1,
                                            seq_f).astype(int).tolist()
                    h_sam = np.linspace(h_o.item(), (t_h + h_o).item() - 1,
                                        seq_h).astype(int).tolist()
                    w_sam = np.linspace(w_o.item(), (t_w + w_o).item() - 1,
                                        seq_w).astype(int).tolist()

                    assert f_o * f >= 0 and h_o * h >= 0 and w_o * w >= 0
                    freqs_0 = freqs[0][f_sam] if f_o >= 0 else freqs[0][
                        f_sam].conj()
                    freqs_0 = freqs_0.view(seq_f, 1, 1, -1)

                    freqs_i = torch.cat([
                        freqs_0.expand(seq_f, seq_h, seq_w, -1),
                        freqs[1][h_sam].view(1, seq_h, 1, -1).expand(
                            seq_f, seq_h, seq_w, -1),
                        freqs[2][w_sam].view(1, 1, seq_w, -1).expand(
                            seq_f, seq_h, seq_w, -1),
                    ],
                                        dim=-1).reshape(seq_len, 1, -1)
                elif t_f < 0:
                    freqs_i = trainable_freqs.unsqueeze(1)
                # apply rotary embedding
                output[i, seq_bucket[-1]:seq_bucket[-1] + seq_len] = freqs_i
        seq_bucket.append(seq_bucket[-1] + seq_len)
    return output



if __name__ == "__main__":
    # python path
    import sys
    sys.path.append('Causvid')
    x = torch.randn(1, 3375, 40,128).to('cuda')
    grid_sizes = [[torch.tensor([[0, 0, 0]]).to('cuda'), torch.tensor([[3, 45, 25]]).to('cuda'), torch.tensor([[3, 45, 25]]).to('cuda')]]
    # freqs = torch.randn(1, 1024, 16, 16).to('cuda')
    start = None
    d = 128
    from liveavatar.models.wan.wan_2_2.modules.model import rope_params
    freqs = torch.cat([
            rope_params(16384, d - 4 * (d // 6)),
            rope_params(16384, 2 * (d // 6)),
            rope_params(16384, 2 * (d // 6))
        ],
                               dim=1).to('cuda')
    torch.cuda.synchronize()
    import time;t_start = time.time()
    output = rope_precompute(x.to('cuda'), grid_sizes, freqs.to('cuda'), start)
    torch.cuda.synchronize()
    print(f"rope precompute time: {time.time() - t_start}")
    x = torch.randn(1, 3375, 40,128).to('cuda')
    torch.cuda.synchronize()
    import time;t_start = time.time()
    output = rope_precompute(x.to('cuda'), grid_sizes, freqs.to('cuda'), start)
    torch.cuda.synchronize()
    print(f"rope precompute time 2: {time.time() - t_start}")
    
    # print(output.shape)
    # from liveavatar.models.wan.wan_2_2.modules.s2v.s2v_utils_ori import rope_precompute as rope_precompute_ori
    # torch.cuda.synchronize()
    # import time;t_start = time.time()
    # output_ori = rope_precompute_ori(x.to('cuda'), grid_sizes, [freqs.to('cuda'), freqs.to('cuda'), freqs.to('cuda')], start)
    # print(f"rope precompute time ori: {time.time() - t_start}")

    # torch.cuda.synchronize()
    # import time;t_start = time.time()
    # output_ori = rope_precompute(x.to('cuda'), grid_sizes, [freqs.to('cuda'), freqs.to('cuda'), freqs.to('cuda')], start)
    # print(f"rope precompute time 2: {time.time() - t_start}")