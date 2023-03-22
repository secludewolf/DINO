from torch import Tensor


def reshape_tensor_32(t: Tensor):
    """
    [bs,3,h,w]
    """
    b, _, h, w = t.shape
    if h % 32 != 0:
        h = h - h % 32
    if w % 32 != 0:
        w = w - w % 32
    t = t[..., :h, :w]
    return t, h, w
