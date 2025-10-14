import os, argparse, re
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from matplotlib.path import Path
from scipy.interpolate import interp1d
import cv2
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--psf_root", default='./PSF_data')
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--iter", type=int, default=5000)
parser.add_argument("--tau", type=float, default=0.1)
parser.add_argument("--tv_albedo_weight", type=float, default=1e-6*5)
parser.add_argument("--plane_reg_weight", type=float, default=2.0)
args = parser.parse_args()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_SHAPE = (520, 656)
PIXEL_SIZE_SRC_MM = 0.015/4/4096 * 1e3
CAPTURE_KEYS = ("1563.60nm", "1564.90nm")
TARGET_SIZE_MM = (2.6, 3.28)
from typing import Optional
def parse_depth_mm(folder):
    return float(re.search(r"_dist_([0-9.]+)mm", folder).group(1))
# Fit a plane and return mean squared residual on mask
def plane_residual_loss(depth_map: torch.Tensor,
                        mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    Ny, Nx = depth_map.shape
    yy, xx = torch.meshgrid(
        torch.arange(Ny, device=depth_map.device, dtype=depth_map.dtype),
        torch.arange(Nx, device=depth_map.device, dtype=depth_map.dtype),
        indexing='ij'
    )

    if mask is None:
        m = torch.ones_like(depth_map)
    else:
        m = mask.to(depth_map.dtype)
    x_vec = xx[m > 0].view(-1, 1)
    y_vec = yy[m > 0].view(-1, 1)
    z_vec = depth_map[m > 0].view(-1, 1)

    A = torch.cat([x_vec, y_vec, torch.ones_like(x_vec)], dim=1)

    theta = torch.linalg.lstsq(A, z_vec).solution
    a, b, c = theta.squeeze()

    plane = a * xx + b * yy + c
    resid2 = ((depth_map - plane) ** 2)[m > 0]

    return resid2.mean()
# Build a 2D Gaussian kernel used to smooth H and Λ before soft depth assignment (noise suppression).
def get_gaussian_kernel(kernel_size=15, sigma=3, device='cpu'):
    ax = torch.arange(kernel_size, dtype=torch.float32, device=device) - kernel_size // 2
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, kernel_size, kernel_size)
gaussian_kernel = get_gaussian_kernel(kernel_size=151, sigma=34, device=DEVICE)
def load_psf_stack(prefix_key, root_dir="."):
    folders = sorted(
        [d for d in os.listdir(root_dir)
         if f"lam_{prefix_key}" in d and "dist_" in d and
            os.path.isdir(os.path.join(root_dir, d))],
        key=parse_depth_mm
    )
    psf_list, z_list = [], []


    for folder in folders:
        z_mm = parse_depth_mm(folder)
        psf = np.load(os.path.join(root_dir, folder, "PSF_I_2D.npy")).astype(np.float32)
        h_crop = round(TARGET_SIZE_MM[0] / PIXEL_SIZE_SRC_MM)
        w_crop = round(TARGET_SIZE_MM[1] / PIXEL_SIZE_SRC_MM)
        cy, cx = psf.shape[0] // 2, psf.shape[1] // 2
        psf_crop = psf[cy - h_crop // 2 : cy + h_crop // 2,
                       cx - w_crop // 2 : cx + w_crop // 2]

        psf_rs = zoom(psf_crop, (TARGET_SHAPE[0]/h_crop, TARGET_SHAPE[1]/w_crop), order=1)

        psf_tensor = torch.tensor(psf_rs, dtype=torch.float32, device=gaussian_kernel.device)
        psf_tensor = psf_tensor/psf_tensor.sum()
        psf_tensor = psf_tensor.unsqueeze(0).unsqueeze(0)
        psf_list.append(psf_tensor.squeeze(0).squeeze(0).cpu().numpy())
        z_list.append(z_mm)

    z_arr = np.array(z_list, np.float32)
    psf_stack = np.stack(psf_list, axis=0).astype(np.float32)
    return z_arr, psf_stack
def psf_to_k(psf_np):
    psf = torch.tensor(psf_np, device=DEVICE)
    Ny, Nx = psf.shape[-2:]
    pad_top = Ny // 2
    pad_bottom = Ny - pad_top
    pad_left = Nx // 2
    pad_right = Nx - pad_left
    pad = (pad_left, pad_right, pad_top, pad_bottom)
    psf_padded = F.pad(psf, pad)
    return torch.fft.fftn(psf_padded, dim=(-2, -1))
def tv_norm(x: torch.Tensor) -> torch.Tensor:
    dh = x[:, 1:] - x[:, :-1]
    dw = x[1:, :] - x[:-1, :]
    return dh.abs().sum() + dw.abs().sum()
meas_imgs = []
data_folder = './'
filenames = ["pic_06.npy", "pic_01.npy"]
import scipy.ndimage
for fn in filenames:
    img = np.load(os.path.join(data_folder, fn)).astype(np.float32)
    img = img[:,0:656]
    img_blurred = img
    img_norm = img_blurred / img_blurred.sum()
    meas_imgs.append(torch.tensor(img_norm, device=DEVICE, dtype=torch.float32))
img1 = meas_imgs[1].cpu().numpy()
img1_u8 = (255 * (img1 - img1.min()) / (img1.max() - img1.min())).astype(np.uint8)
import cv2.ximgproc as xip
binary = xip.niBlackThreshold(
    img1_u8, 255, cv2.THRESH_BINARY,
    blockSize=121, k=0.6,
    binarizationMethod=xip.BINARIZATION_SAUVOLA
)*(img1_u8>15)
for _i in range(8):
    kernel = np.ones((25, 25), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=1)
    kernel = np.ones((5, 5), np.uint8)
    eroded = cv2.erode(binary, kernel, iterations=7)
    binary = eroded
eroded = eroded.astype(np.float32) / 255.0
np.save("eroded.npy", eroded)
eroded = torch.tensor(eroded, device=DEVICE, dtype=torch.float32)
# read PSF
psf_k_list, z_mm = [], None
for key in CAPTURE_KEYS:
    z_raw, psf_raw = load_psf_stack(key, args.psf_root)
    if z_mm is None:
        z_mm = z_raw
    psf_k_list.append(psf_to_k(psf_raw))
Nz, Ny, Nx = psf_k_list[0].shape
Ny = Ny//2
Nx = Nx//2
z_min, z_max = float(z_mm.min()), float(z_mm.max())
eps = 1e-6
# Init learnable Λ and H
a_init = torch.randn((Ny, Nx), device=DEVICE) * 0.1
albedo_param = a_init.clone().requires_grad_()
pixel_size_mm = 5e-3
depth_init = 7.00
# Initialize slope roughly opposite to the true angle (+60°) for robustness testing
slope_init = float(np.tan(np.deg2rad(-60.0)))
sx_val = slope_init
sy_val = 0.0
xs = torch.arange(Nx, device=DEVICE).float() - (Nx - 1) / 2
ys = torch.arange(Ny, device=DEVICE).float() - (Ny - 1) / 2
depth_plane = (
    depth_init
    + sx_val * xs.unsqueeze(0) * pixel_size_mm
    + sy_val * ys.unsqueeze(1) * pixel_size_mm
)
depth_norm = (depth_plane - z_min) / (z_max - z_min)
depth_norm = depth_norm.clamp(eps, 1 - eps)
d_init = torch.logit(depth_norm)
depth_param = d_init.clone().requires_grad_()
depth_optimizer = torch.optim.Adam([depth_param], lr=args.lr)
albedo_optimizer = torch.optim.Adam([albedo_param], lr=args.lr)
z_mm_tensor = torch.tensor(z_mm, device=DEVICE).view(Nz,1,1)
# Forward differentiable renderer
def render(depth_p, albedo_p, gaussian_kernel):
    depth_norm = torch.sigmoid(depth_p)
    depth_map  = depth_norm * (z_max - z_min) + z_min
    depth_map = F.conv2d(
        depth_map.unsqueeze(0).unsqueeze(0),
        gaussian_kernel,
        padding=gaussian_kernel.shape[-1] // 2
    )[0, 0]
    albedo_map = eroded*(0.5*0+2*0.5*torch.sigmoid(albedo_p))
    ks = gaussian_kernel.shape[-1]
    albedo_map = F.conv2d(
        albedo_map.unsqueeze(0).unsqueeze(0),
        gaussian_kernel,
        padding=ks // 2
    )[0, 0]*eroded
    D    = depth_map.unsqueeze(0)
    diff = torch.abs(D - z_mm_tensor)
    mask = torch.softmax(-diff / args.tau, dim=0)
    vol = albedo_map.unsqueeze(0) * mask
    Ny, Nx = vol.shape[-2:]
    pad_top    = Ny // 2
    pad_bottom = Ny - pad_top
    pad_left   = Nx // 2
    pad_right  = Nx - pad_left
    pad = (pad_left, pad_right, pad_top, pad_bottom)
    sim_imgs = []
    for psf_k in psf_k_list:
        vol_pad = F.pad(vol, pad)
        obj_k = torch.fft.fftn(vol_pad, dim=(-2, -1))
        img_k = (obj_k * psf_k).sum(dim=0)
        sim_full = torch.fft.ifftn(img_k, dim=(-2, -1))
        sim_shift = torch.fft.fftshift(sim_full, dim=(-2, -1)).real
        sim = sim_shift[
            pad_top : pad_top + Ny,
            pad_left: pad_left + Nx
        ]
        sim_imgs.append(sim)
    return sim_imgs, depth_map, albedo_map
# Optimization loop
for it in range(1, args.iter+1):
    depth_optimizer.zero_grad()
    albedo_optimizer.zero_grad()
    sim, depth_map, albedo_map = render(depth_param, albedo_param, gaussian_kernel)
    p1 = sim[0] / sim[0].sum()
    p2 = sim[1] / sim[1].sum()
    diff1 = (p1 - meas_imgs[0])
    diff2 = (p2 - meas_imgs[1])
    loss_data_depth  = 0.5 * diff1.pow(2).mean() * 1e11
    loss_data_albedo = 0.5 * diff2.pow(2).mean() * 1e11 *0.1
    mask_vis = (albedo_map > 0.2 * albedo_map.max()).detach()
    loss_plane = args.plane_reg_weight * plane_residual_loss(depth_map, mask_vis)
    loss_tv_albedo  = args.tv_albedo_weight  * tv_norm(albedo_map)
    loss_depth_total  = loss_data_depth  + loss_plane
    loss_albedo_total = loss_data_albedo + loss_tv_albedo
    loss_total = loss_depth_total + loss_albedo_total
    loss_total.backward()
    depth_optimizer.step()
    albedo_optimizer.step()
    if it % 1000 == 0 or it == 1:
        torch.save(depth_map.detach().cpu(),  "recovered_depth_mm.pt")
        torch.save(albedo_map.detach().cpu(), "recovered_albedo.pt")
        torch.save(sim[0].detach().cpu(), "sim1_image.pt")
        torch.save(sim[1].detach().cpu(), "sim2_image.pt")