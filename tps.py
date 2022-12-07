import torch

def tps_theta_from_points(c_src, c_dst, reduced=False):
    delta = c_src - c_dst
    cx = torch.cat((c_dst, delta[:, 0].unsqueeze(1)), dim=1)
    cy = torch.cat((c_dst, delta[:, 1].unsqueeze(1)), dim=1)
    theta_dx = TPS.fit(cx, reduced=reduced)
    theta_dy = TPS.fit(cy, reduced=reduced)
    return torch.stack((theta_dx, theta_dy), dim=-1)

xs = torch.arange(0, 1000).to(torch.float32).cuda(0)
yx = torch.arange(0, 1000).to(torch.float32).cuda(0)
grid_debug = torch.meshgrid([xs, yx])
grid_debug = torch.stack(grid_debug, dim=2)

def uniform_grid(shape, normlize=True):
    '''Uniform grid coordinates.

    Params
    ------
    shape : tuple
        HxW defining the number of height and width dimension of the grid

    Returns
    -------
    points: HxWx2 tensor
        Grid coordinates over [0,1] normalized image range.
    '''

    H, W = shape[:2]
#     if normlize:
#         xs = torch.range(0, W-1) / float(W)
#         yx = torch.range(0, H-1) / float(H)
#     else:
#         xs = torch.range(0, W-1)
#         yx = torch.range(0, H-1)
#     c = torch.meshgrid([xs, yx])
#     c = torch.stack(c, dim=2)
    c = grid_debug[0:H, 0:W, :].clone()
    if normlize:
        c[:,:,0] /= float(W)  # xs
        c[:,:,1] /= float(H)  #ys
    return c

def tps_grid(theta, c_dst, dshape):
    device = c_dst.device
    ugrid = uniform_grid(dshape).to(device)
    reduced = c_dst.shape[0] + 2 == theta.shape[0]
    dx = TPS.z(ugrid.reshape((-1, 2)), c_dst, theta[:, 0]).reshape(dshape[:2])
    dy = TPS.z(ugrid.reshape((-1, 2)), c_dst, theta[:, 1]).reshape(dshape[:2])
    dgrid = torch.stack((dx, dy), -1) 
    grid = dgrid + ugrid
    return grid  # H'xW'x2 grid[i,j] in range [0..1]


def tps_grid_to_remap(grid, sshape):
    '''Convert a dense grid to OpenCV's remap compatible maps.

    Params
    ------
    grid : HxWx2 array
        Normalized flow field coordinates as computed by compute_densegrid.
    sshape : tuple
        Height and width of source image in pixels.


    Returns
    -------
    mapx : HxW array
    mapy : HxW array
    '''

    mx = (grid[:, :, 0] * sshape[1]).cpu().numpy().astype(np.float32)
    my = (grid[:, :, 1] * sshape[0]).cpu().numpy().astype(np.float32)

    return mx, my

class TPS:
    @staticmethod
    def fit(c, lambd=0., reduced=False):
        # n = c.shape[0]
        device = c.device
        n = c.size(0)
        # TPS.d(c, c).sum().backward(retain_graph=True)
        U = TPS.u(TPS.d(c, c))
        K = U + torch.eye(n, device=device, dtype=torch.float32) * lambd

        P = torch.ones((n, 3), device=device, dtype=torch.float32)
        P[:, 1:] = c[:, :2]

        v = torch.zeros(n + 3, device=device, dtype=torch.float32)
        v[:n] = c[:, -1]

        A = torch.zeros((n + 3, n + 3), device=device, dtype=torch.float32)
        A[:n, :n] = K
        A[:n, -3:] = P
        A[-3:, :n] = P.T
        try:
            theta, LU = torch.solve(v.unsqueeze(1), A)  # p has structure w,a
            theta = theta.squeeze(1)
        except:
            theta = torch.ones_like(A[:, 0])
        return theta[1:] if reduced else theta

    @staticmethod
    def d(a, b):
        # dd = torch.norm(a)
        # dd.sum().backward(retain_graph=True)
        # # return np.sqrt(np.square(a[:, None, :2] - b[None, :, :2]).sum(-1))
        # d1 = (a[:, None, :2] - b[None, :, :2])
        # d1.sum().backward(retain_graph=True)
        # d2 = d1**2
        # d2.sum().backward(retain_graph=True)
        # d3 = d2.sum(-1)
        # d3.sum().backward(retain_graph=True)
        # d4 = torch.sqrt(d3)
        # d4.sum().backward(retain_graph=True)
        return torch.sqrt(1e-6 + ((a[:, None, :2] - b[None, :, :2])**2).sum(-1))

    @staticmethod
    def u(r):
        return r ** 2 * torch.log(r + 1e-6)

    @staticmethod
    def z(x, c, theta):
        assert x.ndim >= 2
        U = TPS.u(TPS.d(x, c))
        w, a = theta[:-3], theta[-3:]
        reduced = theta.shape[0] == c.shape[0] + 2
        if reduced:
            w = torch.cat((-torch.sum(w, dim=0, keepdim=True), w))
        b = torch.matmul(U, w)
        return a[0] + a[1] * x[:, 0] + a[2] * x[:, 1] + b