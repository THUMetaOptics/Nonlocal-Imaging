import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from mayavi import mlab
from scipy.ndimage import gaussian_filter, zoom
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from tvtk.api import tvtk
from PIL import Image
dir_name = './'
filename_depth  = os.path.join(dir_name, "recovered_depth_mm.pt")
filename_albedo = os.path.join(dir_name, "recovered_albedo.pt")

dz_mm = (11.0-1.0)/30
z_min = 1.0
z_max = 11.0
dx_mm = 0.005

colors = [
    "#3c5474", "#31698a", "#1a7e9b", "#0094a6",
    "#00a9aa", "#16b8a7", "#3cc7a0", "#61d494",
    "#86df89", "#abe97f", "#d1f176", "#f9f871"
]
cmap_depth = LinearSegmentedColormap.from_list("custom_cmap_depth", colors, N=256)

colors_intensity = [
    "#0a204f", "#003460", "#00476d", "#005a76",
    "#006d7c", "#277f86", "#42908f", "#5ba298",
    "#7fb8ad", "#a2cec3", "#c4e4db", "#e6fbf4"
]
cmap_intensity = LinearSegmentedColormap.from_list("custom_cmap_int", colors_intensity, N=256)
# 3D surface preview of H
def visualize_surface_from_depth(depth_map, albedo_mask, dx_mm):
    depth_map =np.flip(depth_map, axis=0)
    albedo_mask = np.flip(albedo_mask, axis=0)
    depth_smooth = depth_map
    mask = (albedo_mask == 1)
    depth_smooth = depth_smooth.copy()
    depth_smooth[~mask] = np.nan

    nx, ny = depth_smooth.shape
    x = np.linspace(-nx * dx_mm / 2, nx * dx_mm / 2, nx)
    y = np.linspace(-ny * dx_mm / 2, ny * dx_mm / 2, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    fig = mlab.figure(size=(1000, 1000), bgcolor=(0.96, 0.96, 0.96))
    surf = mlab.surf(X, Y, depth_smooth, warp_scale=1.0)

    renderer = fig.scene.renderer
    try:
        renderer.remove_all_lights()
    except AttributeError:
        n = renderer.lights.number_of_items
        for _ in range(n):
            light = renderer.lights.get_item_as_object(0)
            renderer.remove_light(light)
    light_dirs = [(1,1,1), (-1,1,1), (1,-1,1), (1,1,-1),
                  (-1,-1,1), (-1,1,-1), (1,-1,-1), (-1,-1,-1)]
    for pos in light_dirs:
        light = tvtk.Light()
        light.light_type = 'scene_light'
        light.position = pos
        light.focal_point = (0, 0, 0)
        light.intensity = 0.125
        renderer.add_light(light)

    rgba = (cmap_depth(np.linspace(0, 1, 256)) * 255).astype(np.uint8)
    lut_mgr = surf.module_manager.scalar_lut_manager
    lut_mgr.lut.table = rgba
    lut_mgr.data_range = (1, 11)
    mlab.outline(surf, extent=[x.min(), x.max(), y.min(), y.max(), 3, 8],
                 color=(0, 0, 0), line_width=1.0)
    mlab.show()

def _find_runs_bool(arr_bool):
    arr = np.asarray(arr_bool, dtype=np.bool_)
    n = arr.size
    if n == 0:
        return []
    d = np.diff(arr.astype(np.int8))
    starts = np.where(d == 1)[0] + 1
    ends   = np.where(d == -1)[0] + 1
    if arr[0]:
        starts = np.r_[0, starts]
    if arr[-1]:
        ends = np.r_[ends, n]
    runs = [(int(s), int(e-1)) for s, e in zip(starts, ends)]
    return runs

# Determine center position of E
def center_position(albedo_mask, save_debug=True):
    Ny, Nx = albedo_mask.shape
    x0 = 314
    y0 = 254
    row = albedo_mask[y0, :].astype(bool)
    row_runs = _find_runs_bool(row)
    run_idx = None
    for i, (xs, xe) in enumerate(row_runs):
        if xs <= x0 <= xe:
            run_idx = i
            break
    xs, xe = row_runs[run_idx]
    col = albedo_mask[:, x0].astype(bool)
    col_runs = _find_runs_bool(col)
    col_idx = None
    for j, (ys, ye) in enumerate(col_runs):
        if ys <= y0 <= ye:
            col_idx = j
            break
    ys, ye = col_runs[col_idx]
    y_mid_f = 0.5 * (ys + ye)
    y_mid = int(np.rint(y_mid_f))
    center_x = x0
    center_y = int(np.clip(y_mid, 0, Ny-1))
    if save_debug:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(albedo_mask, cmap='gray', origin='upper')
        ax.axhline(y=y0, linestyle='--', linewidth=1)
        ax.axvline(x=x0, linestyle='--', linewidth=1)
        ax.plot([xs, xe], [y0, y0], linewidth=4)
        ax.plot([x0, x0], [ys, ye], linewidth=4)
        ax.plot([x0], [y0], marker='o', markersize=6)
        ax.plot([center_x], [center_y], marker='x', markersize=8)
        ax.set_title(f"click=({x0},{y0}), row_runs={len(row_runs)}, col_runs={len(col_runs)}\n"
                     f"row_sel=[{xs},{xe}], col_sel=[{ys},{ye}], center=({center_x},{center_y})")
        plt.tight_layout()
        plt.savefig("center_position.png", dpi=300)
        plt.close(fig)
    return center_x, center_y, x0, y0, row_runs, col_runs

# Determine the ground‑truth plane passing through the center, given a known 60° tilt angle.
def fit_true_plane_fixed_tilt(xx, yy, depth_map, mask, dx_mm, tilt_deg, center_x, center_y, z0,
                              n_phi=7200, phi_batch=256, dtype=np.float32):
    theta = np.deg2rad(float(tilt_deg))
    s = dx_mm * np.tan(theta)
    valid = mask.astype(bool) & np.isfinite(depth_map)
    if not np.any(valid):
        raise ValueError("No valid pixels for plane fitting (mask empty).")
    xi = (xx[valid] - center_x).astype(dtype, copy=False)
    yi = (yy[valid] - center_y).astype(dtype, copy=False)
    yv = (depth_map[valid] - z0).astype(dtype, copy=False)
    phis = np.linspace(0.0, 2.0*np.pi, int(n_phi), endpoint=False, dtype=np.float64)
    best_cost = np.inf
    k0 = 0
    for start in range(0, len(phis), phi_batch):
        end = min(start + phi_batch, len(phis))
        phi_blk = phis[start:end]
        cph = np.cos(phi_blk).astype(dtype, copy=False)
        sph = np.sin(phi_blk).astype(dtype, copy=False)
        t_blk = s * (np.outer(xi, cph) + np.outer(yi, sph))
        resid_blk = yv[:, None] - t_blk
        cost_blk = np.sum(np.abs(resid_blk), axis=0)
        j_local = int(np.argmin(cost_blk))
        cost_local = float(cost_blk[j_local])
        if cost_local < best_cost:
            best_cost = cost_local
            k0 = start + j_local
    phi_best = float(phis[k0])
    a_true = float(s * np.cos(phi_best))
    b_true = float(s * np.sin(phi_best))
    c_true = float(z0 - a_true * center_x - b_true * center_y)
    return a_true, b_true, c_true, phi_best

if __name__ == "__main__":
    depth_map  = torch.load(filename_depth, map_location="cpu").numpy()
    albedo_map = torch.load(filename_albedo, map_location="cpu").numpy()
    albedo_map = albedo_map / max(1e-12, np.max(albedo_map))
    albedo_mask = (albedo_map > 0.1)
    plt.imsave("recovered_albedo_map.png", albedo_map, cmap=cmap_intensity)
    rgb = (cmap_intensity(albedo_map)[..., :3] * 255).astype(np.uint8)
    Image.fromarray(rgb).save("recovered_albedo_map.tif")
    plt.figure()
    plt.imshow(albedo_mask, cmap='gray')
    plt.xticks([]); plt.yticks([])
    plt.savefig("albedo_mask.png", dpi=300, bbox_inches='tight')
    plt.savefig("albedo_mask.tif", dpi=300, bbox_inches='tight')
    plt.close()
    def _save_sim_image(sim_tensor, basename: str, cmap_used):
        sim_arr = sim_tensor.numpy() if isinstance(sim_tensor, torch.Tensor) else sim_tensor
        sim_norm = sim_arr.astype(np.float32)
        if sim_norm.max() > 0: sim_norm /= sim_norm.max()
        plt.imsave(f"{basename}.png", sim_norm, cmap=cmap_used)
        rgb = (cmap_used(sim_norm)[..., :3] * 255).astype(np.uint8)
        Image.fromarray(rgb).save(f"{basename}.tif")
    for name in ("sim1_image.pt", "sim2_image.pt"):
        f = os.path.join(dir_name, name)
        arr = torch.load(f, map_location="cpu")
        _save_sim_image(arr, name.replace(".pt",""), cmap_intensity)
    Ny, Nx = depth_map.shape
    ys = np.arange(Ny)
    xs = np.arange(Nx)
    xx, yy = np.meshgrid(xs, ys)
    masked_depth = np.ma.masked_where(~albedo_mask, depth_map)
    cmap1 = cmap_depth.copy(); cmap1.set_bad(color='white')
    plt.figure(figsize=(8, 6))
    plt.imshow(masked_depth, cmap=cmap1, vmin=1, vmax=11)
    plt.xticks([]); plt.yticks([])
    plt.savefig("recovered_depth_map.png", dpi=300, bbox_inches='tight')
    plt.savefig("recovered_depth_map.tif", dpi=300, bbox_inches='tight')
    plt.close()
    z_grid = np.arange(z_min, z_max + dz_mm, dz_mm)
    visualize_surface_from_depth(depth_map, albedo_mask, dx_mm)
    center_x, center_y, x0, y0, row_runs, col_runs = center_position(albedo_mask, save_debug=True)
    z0 = 6.21 # Ground-truth value for the central height
    tilt_deg = 60.0
    a_true, b_true, c_true, phi_true = fit_true_plane_fixed_tilt(
        xx=xx, yy=yy, depth_map=depth_map, mask=albedo_mask,
        dx_mm=dx_mm, tilt_deg=tilt_deg,
        center_x=center_x, center_y=center_y, z0=z0
    )
    anglex_true_eff = np.degrees(np.arctan(a_true / dx_mm))
    angley_true_eff = np.degrees(np.arctan(b_true / dx_mm))
    tilt_total_deg = np.degrees(np.arctan(np.sqrt((a_true/dx_mm)**2 + (b_true/dx_mm)**2)))
    depth_true = a_true * xx + b_true * yy + c_true
    error_map = np.abs(depth_map - depth_true)
    valid_mask = albedo_mask & np.isfinite(error_map)
    rel_error_map = np.abs(depth_map - depth_true) / np.maximum(np.abs(depth_true), 1e-9)
    mean_rel_error = float(np.mean(rel_error_map[valid_mask]))
    print(f"The average of pointwise relative differences between reconstructed and ground truth depth values is {mean_rel_error*100:.2f}%")
