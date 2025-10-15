import torch
import torch.fft
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import torch.optim as optim
import torch.nn.functional as F
from scipy.ndimage import median_filter
from scipy.ndimage import grey_erosion
from matplotlib.colors import LinearSegmentedColormap
colors = [
    "#0a204f", "#003460", "#00476d", "#005a76",
    "#006d7c", "#277f86", "#42908f", "#5ba298",
    "#7fb8ad", "#a2cec3", "#c4e4db", "#e6fbf4"
]
cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)
# Generate an array with a specified size
def create_sphere_slice_phase(layer_idx, total_slices,
                              sphere_radius, delta_n, wavelength,
                              grid_extent, grid_size,z_volumn):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    N = grid_size
    L = grid_extent
    dz = z_volumn / total_slices
    z_min_layer = -z_volumn/2+ dz/2 + layer_idx     * dz -dz/2
    z_max_layer = -z_volumn/2+ dz/2+ layer_idx * dz +dz/2

    x = torch.linspace(-L/2, L/2-L/N, N, device=device)+7*L/N
    y = torch.linspace(-L/2, L/2-L/N, N, device=device)+7*L/N
    X, Y = torch.meshgrid(x, y, indexing='ij')
    Rxy = torch.sqrt(X**2 + Y**2)
    h = torch.sqrt(torch.clamp(sphere_radius**2 - Rxy**2, min=0.0))
    z_min_valid = torch.maximum(z_min_layer*torch.ones_like(h), -h)
    z_max_valid = torch.minimum(z_max_layer*torch.ones_like(h), +h)
    thickness = torch.clamp(z_max_valid - z_min_valid, min=0.0)
    k = 2 * np.pi / wavelength
    phase_slice = k * delta_n * thickness
    U_slice = torch.exp(1j * phase_slice).type(torch.complex64)
    return U_slice
# Generate the modulation function for a given slice.
def create_cube_slice_phase_overridable(layer_idx, total_slices,
                                        sphere_radius, delta_n, wavelength,
                                        grid_extent, grid_size,
                                        override_phase=None,
                                        center_pixels=100):
    U_slice = torch.ones(grid_size, grid_size, dtype=torch.complex64, device='cuda')
    if override_phase is not None:
        current_phase = torch.angle(U_slice)
        N = grid_size
        half = center_pixels // 2
        center = N // 2
        i0 = center - half
        j0 = center - half
        current_phase[i0:i0+center_pixels, j0:j0+center_pixels] = override_phase
        U_slice = torch.exp(1j * current_phase)
    return U_slice
# Perform angular spectrum propagation in the medium.
def angular_spectrum_propagate(U, wavelength, z, grid_extent, H1):
    U_shifted = torch.fft.fftshift(U, dim=(-2, -1))
    U_fft = torch.fft.fft2(U_shifted)
    U_fft = torch.fft.fftshift(U_fft, dim=(-2, -1))
    U_prop_fft = U_fft * H1
    U_ifftshift = torch.fft.ifftshift(U_prop_fft, dim=(-2, -1))
    U_ifft = torch.fft.ifft2(U_ifftshift)
    U_propagated = torch.fft.fftshift(U_ifft, dim=(-2, -1))
    return U_propagated
# Perform angular spectrum propagation in the air.
def ASM_propagate(U, wavelength, z, grid_extent, U_pad, paddingnum2, H2):
    N = U.shape[-2]
    L = grid_extent
    dx = L / N
    k = 2 * np.pi / wavelength
    up_factor = paddingnum2
    Npad = up_factor * N
    U_pad = torch.zeros((U.shape[0], Npad, Npad), dtype=torch.complex64, device=U.device)
    i0 = (Npad - N) // 2
    j0 = (Npad - N) // 2
    U_pad[:, i0:i0+N, j0:j0+N] = U
    U_shifted = torch.fft.fftshift(U_pad, dim=(-2, -1))
    U_fft = torch.fft.fft2(U_shifted)
    U_fft = torch.fft.fftshift(U_fft, dim=(-2, -1))
    Upad_prop = U_fft * H2
    U_ifftshift = torch.fft.ifftshift(Upad_prop, dim=(-2, -1))
    Upad_out = torch.fft.ifft2(U_ifftshift)
    Upad_out = torch.fft.ifftshift(Upad_out, dim=(-2, -1))
    U_out = Upad_out
    return U_out
# Perform full multi-layer wave propagation.
def wpm_layers(U_in, wavelength, z_total, grid_extent,
                 sphere_radius, delta_n, grid_size, paddingnum, z_volumn,
                 U_in_bg, sphere_slice_list, total_slices, paddingnum2, H1, H2, U_bg_layers,z_mid,H3):
    z_step = z_volumn / (total_slices)
    U = U_in.clone()
    sidenum = 200
    cmin = sidenum
    cmax = grid_size-sidenum
    mid_layer = total_slices // 2
    # The position of the aperture
    aperture_offset = int(round(56e-6 / z_step))
    aperture_layer_idx = mid_layer - aperture_offset
    N = U.shape[-1]
    device = U.device
    for layer_idx in range(total_slices):
        U = U * sphere_slice_list[layer_idx]
        if layer_idx < total_slices - 1:
            U = angular_spectrum_propagate(U, wavelength, z_step, grid_extent, H1)
            # Apply a circular aperture truncation with 1 mm diameter.
            if layer_idx == aperture_layer_idx:
                x = torch.linspace(-grid_extent/2,
                                   grid_extent/2 - grid_extent/N,
                                   N, device=device)
                y = torch.linspace(-grid_extent/2,
                                   grid_extent/2 - grid_extent/N,
                                   N, device=device)
                X, Y = torch.meshgrid(x, y, indexing='ij')
                mask = (X**2 + Y**2 <= (0.5*1e-3)**2).to(U.dtype)[None, :, :]
                U = U * mask
        else:
            U = ASM_propagate(
                U,
                wavelength,
                z_total,
                grid_extent,
                U_in_bg,
                paddingnum2,
                H2
            )
    return U

# Generate a tilted plane wave
def add_tilt_only(wavelength,
                  grid_extent,
                  tilt_angle_deg,
                  azimuth_angle_deg,
                  grid_size,
                  z=0.0,
                  n_m=1,
                  aperture_radius=0.5e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B = tilt_angle_deg.shape[0]
    N = grid_size
    L = grid_extent

    x = torch.linspace(-L/2, L/2 - L/N, N, device=device)
    y = torch.linspace(-L/2, L/2 - L/N, N, device=device)
    X, Y = torch.meshgrid(x, y, indexing='ij')  
    k = 2 * np.pi / wavelength
    tilt_rad    = torch.deg2rad(tilt_angle_deg)  
    azimuth_rad = torch.deg2rad(azimuth_angle_deg)  
    kx = k * torch.sin(tilt_rad) * torch.cos(azimuth_rad)
    ky = k * torch.sin(tilt_rad) * torch.sin(azimuth_rad)
    kz = torch.sqrt(torch.clamp((n_m * k)**2 - (kx**2 + ky**2), min=0.0))
    kx = kx[:, None, None]
    ky = ky[:, None, None]
    kz = kz[:, None, None]
    Xb = X.unsqueeze(0)  
    Yb = Y.unsqueeze(0)  
    phase = kx * Xb + ky * Yb + kz * z   
    U_tilt = torch.exp(1j * phase).to(torch.complex64)  
    radius = aperture_radius
    aperture_mask = (Xb**2 + Yb**2 <= radius**2)
    aperture_mask = aperture_mask.to(U_tilt.dtype)
    U_tilt_aperture = U_tilt * aperture_mask
    return U_tilt_aperture

def compute_intensity(U):
    return torch.abs(U)**2
import os
def load_preprocessed_images_and_masks(data_dir, tilt_angles, grid_size_out, dx,
                                       apply_highpass=True, highpass_ratio=0.005):
    import os
    import numpy as np
    import torch
    import torch.nn.functional as F

    cropped_files = sorted([f for f in os.listdir(data_dir) if f.endswith('_cropped.npy')],
                           key=lambda x: float(x.split('_')[1]))
    mask_files = sorted([f for f in os.listdir(data_dir) if f.endswith('_mask.npy')],
                        key=lambda x: float(x.split('_')[1]))
    angle_values = [0.0, 5.0, 6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0,17.0,18.0,19.0,20.0]
    selected_indices = []
    for t in tilt_angles:
        diffs = [abs(t - w) for w in angle_values]
        idx = diffs.index(min(diffs))
        selected_indices.append(idx)
    ref_images_interp = []
    masks_interp = []
    origin_images_interp = []
    grid_H, grid_W = grid_size_out
    for idx in selected_indices:
        print(cropped_files[idx], end=' ')
        cropped_path = os.path.join(data_dir, cropped_files[idx])
        mask_path = os.path.join(data_dir, mask_files[idx])

        img = np.load(cropped_path)
        mask = np.load(mask_path)
        H_raw, W_raw = img.shape

        H_target = round(H_raw * 5.0e-6 / dx)
        W_target = round(W_raw * 5.0e-6 / dx)

        img_tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)

        img_interp = F.interpolate(img_tensor, size=(H_target, W_target), mode='bicubic', align_corners=False)
        mask_interp = F.interpolate(mask_tensor, size=(H_target, W_target), mode='nearest')

        img_interp = img_interp.squeeze(0).squeeze(0)
        mask_interp = mask_interp.squeeze(0).squeeze(0)

        if apply_highpass:
            origin_interp = img_interp.clone()
            img_interp = img_interp

        pad_top = (grid_H - H_target) // 2
        pad_bottom = grid_H - H_target - pad_top
        pad_left = (grid_W - W_target) // 2
        pad_right = grid_W - W_target - pad_left
        padding = (pad_left, pad_right, pad_top, pad_bottom)

        img_padded = F.pad(img_interp.unsqueeze(0).unsqueeze(0), padding, mode='constant', value=0)
        mask_padded = F.pad(mask_interp.unsqueeze(0).unsqueeze(0), padding, mode='constant', value=0)
        origin_padded = F.pad(origin_interp.unsqueeze(0).unsqueeze(0), padding, mode='constant', value=0)

        ref_images_interp.append(img_padded.squeeze(0).squeeze(0).cpu())
        masks_interp.append(mask_padded.squeeze(0).squeeze(0).cpu())
        origin_images_interp.append(origin_padded.squeeze(0).squeeze(0).cpu())

    return ref_images_interp, masks_interp, origin_images_interp

def main_optimization_example():
    def compute_tv_norm(
            phase_params: torch.Tensor,
            dx: float,
            z_step: float,
            eps: float = 1e-8
    ) -> torch.Tensor:
        dz = phase_params[1:, :, :] - phase_params[:-1, :, :]
        tv_z = torch.sum(torch.sqrt(dz * dz + eps * eps)) * z_step/50

        dy = phase_params[:, 1:, :] - phase_params[:, :-1, :]
        tv_y = torch.sum(torch.sqrt(dy * dy + eps * eps)) * dx

        dx_ = phase_params[:, :, 1:] - phase_params[:, :, :-1]
        tv_x = torch.sum(torch.sqrt(dx_ * dx_ + eps * eps)) * dx

        return tv_z + tv_y + tv_x

    def compute_loss_per_item(
            I_sim_list,
            ref_images,
            masks,
            epoch=0,
            weights=None,
            sim_outside_vals=None,
            highpass_ratio=0.005
        ):
        assert len(I_sim_list) == len(ref_images) == len(masks)

        if epoch == 0:
            sim_outside_vals = []

        loss_list = []

        for idx, ((angle, sim_tensor), ref_tensor, mask_tensor) in enumerate(
            zip(I_sim_list, ref_images, masks)
        ):
            sim_filtered = sim_tensor

            if epoch == 0:
                out = sim_filtered.detach().clone()
                out[mask_tensor.bool()] = 0.0
                sim_outside_vals.append(out)

            ref_filtered = ref_tensor.clone()
            ref_filtered[~mask_tensor.bool()] = sim_outside_vals[idx][~mask_tensor.bool()]

            sim_inside = sim_filtered * mask_tensor
            ref_inside = ref_filtered * mask_tensor
            sim_inside_norm = sim_inside / (sim_inside.norm() + 1e-12)
            ref_inside_norm = ref_inside / (ref_inside.norm() + 1e-12)
            inside_loss = F.mse_loss(sim_inside_norm, ref_inside_norm)

            mask_inv = ~mask_tensor.bool()
            sim_outside = sim_filtered * mask_inv
            ref_outside = ref_filtered * mask_inv
            denom_out = ref_outside.norm() + 1e-12
            sim_out_norm = sim_outside / denom_out
            ref_out_norm = ref_outside / denom_out
            outside_loss = F.mse_loss(sim_out_norm, ref_out_norm)

            loss_list.append(inside_loss + 0*outside_loss)

        loss_tensor = torch.stack(loss_list)

        if epoch == 0:
            normalized = loss_tensor / loss_tensor.max()
            new_weights = 1.0 / (normalized + 1e-12)
            new_weights = new_weights / new_weights.sum()
            total_loss = (loss_tensor * new_weights).sum()
            return total_loss * 1e14, new_weights, sim_outside_vals

        else:
            w = torch.tensor(weights, device=loss_tensor.device, dtype=loss_tensor.dtype)
            w = w / (w.sum() + 1e-12)
            total_loss = (loss_tensor * w).sum()
            return total_loss * 1e14, None, sim_outside_vals
    #Forward simulation over tilt and azimuth angles, splitting azimuths into two batches for parallel GPU computation.
    def forward_simulation(phase_params, H1, H2, tilt_angles, grid_size_out,
                           azimuth_angles, ref_image_torch, z_mid, H3):
        sphere_slice_list = []
        for layer_idx in range(total_slices):
            override_phase_layer = phase_params[layer_idx, :, :]
            mapped_phase = 4*target_phase * torch.sigmoid(override_phase_layer)-0.003*target_phase
            mapped_phase_blurred = mapped_phase
            U_slice = create_cube_slice_phase_overridable(
                layer_idx=layer_idx,
                total_slices=total_slices,
                sphere_radius=sphere_radius,
                delta_n=delta_n,
                wavelength=wavelength,
                grid_extent=grid_extent,
                grid_size=grid_size,
                override_phase=mapped_phase_blurred,
                center_pixels=center_pixels,
            )
            sphere_slice_list.append(U_slice)
        def process_azimuth_batch(az_batch):
            # Number of azimuths and tilts
            A = len(az_batch)
            B = len(tilt_angles)
            # Create angle grids
            tilt_t = torch.tensor(tilt_angles, device=device, dtype=torch.float32)
            az_t   = torch.tensor(az_batch,    device=device, dtype=torch.float32)
            tilt_grid = tilt_t.unsqueeze(0).expand(A, B).reshape(-1)
            az_grid   = az_t.unsqueeze(1).expand(A, B).reshape(-1)
            # Compute input fields
            U_in = add_tilt_only(
                wavelength=wavelength,
                grid_extent=grid_extent,
                tilt_angle_deg=tilt_grid,
                azimuth_angle_deg=az_grid,
                grid_size=grid_size,
                aperture_radius=1.2 * 5e-3
            )
            U_in_bg = add_tilt_only(
                wavelength=wavelength,
                grid_extent=grid_extent * paddingnum2,
                tilt_angle_deg=tilt_grid,
                azimuth_angle_deg=az_grid,
                grid_size=int(grid_size * paddingnum2),
                z=z_volumn * (total_slices-1) / total_slices,
                n_m=n_m
            )
            U_bg_layers = []
            for j in range(total_slices - 1):
                z_j = z_volumn * (j+1) / total_slices
                U_bg_layers.append(
                    add_tilt_only(
                        wavelength=wavelength,
                        grid_extent=grid_extent,
                        tilt_angle_deg=tilt_grid,
                        azimuth_angle_deg=az_grid,
                        grid_size=grid_size,
                        z=z_j,
                        n_m=n_m
                    )
                )
            # Multi-layer WPM propagation
            U_out = wpm_layers(
                U_in=U_in,
                wavelength=wavelength,
                z_total=z,
                grid_extent=grid_extent,
                sphere_radius=sphere_radius,
                delta_n=delta_n,
                grid_size=grid_size,
                paddingnum=paddingnum,
                z_volumn=z_volumn,
                U_in_bg=U_in_bg,
                sphere_slice_list=sphere_slice_list,
                total_slices=total_slices,
                paddingnum2=paddingnum2,
                H1=H1,
                H2=H2,
                U_bg_layers=U_bg_layers,
                z_mid=z_mid,
                H3=H3
            )
            # Compute intensity and sum over azimuth dimension
            I_out = compute_intensity(U_out)
            I_out = I_out.view(A, B, grid_size_out, grid_size_out)
            return I_out.sum(dim=0)

        # Split azimuth angles into two batches
        A = len(azimuth_angles)
        mid = A // 2
        az_batch1 = azimuth_angles[:mid]
        az_batch2 = azimuth_angles[mid:]

        # Process each batch
        intensity1 = process_azimuth_batch(az_batch1)
        intensity2 = process_azimuth_batch(az_batch2)
        intensity_sum_batch = intensity1 + intensity2

        final_images = []
        final_images_np = []
        for i, tilt in enumerate(tilt_angles):
            img = intensity_sum_batch[i]
            start = (img.shape[0] - grid_size_out) // 2
            crop_img = img[start:start+grid_size_out, start:start+grid_size_out]
            final_images.append((tilt, crop_img))
            final_images_np.append((tilt, crop_img.detach().cpu().numpy()))

        return final_images, final_images_np

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    grid_size    = int(800)
    grid_extent  = (1500e-6)
    sphere_radius = 50e-6
    delta_n      = 0.003
    wavelength   = 1600e-9
    z            = 1.347e-3
    n_m = 1.561

    z_volumn     = 800e-6
    total_slices = 60
    tilt_angles_total     = [0.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0,17.0,18.0,19.0]
    paddingnum = 800/400
    paddingnum2 = 2
    N = grid_size
    L = grid_extent
    dx = L / N
    z_step = z_volumn / (total_slices)

    fx = torch.linspace(-1/(2*dx), 1/(2*dx) - 1/dx/N, N, device=device)
    fy = torch.linspace(-1/(2*dx), 1/(2*dx) - 1/dx/N, N, device=device)
    FX, FY = torch.meshgrid(fx, fy)
    k = 2 * np.pi / wavelength*n_m
    kx = 2 * np.pi * FX
    ky = 2 * np.pi * FY
    kz = torch.sqrt(torch.clamp(k**2 - (kx**2 + ky**2), min=0.0+0j))
    H1 = torch.exp(1j * kz * z_step).type(torch.complex64)
    z_mid = 400e-6
    H3 = torch.exp(1j * kz * z_mid).type(torch.complex64)
    k = 2 * np.pi / wavelength
    dx = grid_extent / grid_size
    N_pad = int(paddingnum2 * grid_size)
    fx = torch.linspace(-1/(2*dx), 1/(2*dx) - 1/dx/N_pad, N_pad, device=device)
    fy = torch.linspace(-1/(2*dx), 1/(2*dx) - 1/dx/N_pad, N_pad, device=device)
    FX, FY = torch.meshgrid(fx, fy)
    kx = 2 * np.pi * FX
    ky = 2 * np.pi * FY
    kz = torch.sqrt(torch.clamp(k**2 - (kx**2 + ky**2), min=0.0+0j))
    H2 = torch.exp(1j * kz * z).type(torch.complex64)
    center_pixels = 800

    target_phase = delta_n * z_step / wavelength * 2 * np.pi
    ref_image_cpu, masks_cpu, origin_images_interp_cpu = load_preprocessed_images_and_masks('./diff_images', tilt_angles_total, (int(grid_size*paddingnum2),int(grid_size*paddingnum2)),dx)
    initial_phase = []
    for layer_idx in range(total_slices):
        U_slice = create_sphere_slice_phase(
            layer_idx=layer_idx,
            total_slices=total_slices,
            sphere_radius=sphere_radius,
            delta_n=delta_n,
            wavelength=wavelength,
            grid_extent=grid_extent,
            grid_size=grid_size,
            z_volumn=z_volumn
        )
        phase_slice = torch.angle(U_slice).float()  # shape: (H, W)
        H, W = phase_slice.shape
        h0 = H // 2 - center_pixels // 2
        h1 = H // 2 + center_pixels // 2
        w0 = W // 2 - center_pixels // 2
        w1 = W // 2 + center_pixels // 2
        phase_center = phase_slice[h0:h1, w0:w1]

        eps = -1e-6
        clipped = torch.clamp(torch.zeros_like(phase_center)+0.003*target_phase, min=eps, max=4*target_phase - eps)/4
        inv_sigmoid = torch.log(clipped / (target_phase - clipped))  # σ⁻¹(y) = log(y / (1 - y))
        initial_phase.append(inv_sigmoid)

        initial_phase_tensor = torch.stack(initial_phase, dim=0)  # shape: (D, H, W)
        phase_params = torch.nn.Parameter(initial_phase_tensor.to(device))


    optimizer = optim.Adam([phase_params], lr=0.15)

    num_epochs = 4201
    num_save = 50
    grid_size_out = int(grid_size * paddingnum2)
    num_azimuth = 120
    azimuth_angles = np.linspace(0, 360, num_azimuth, endpoint=False)

    os.makedirs("loss_plots", exist_ok=True)

    data_loss_history = []
    tv_loss_history = []
    l1_loss_history = []
    spa_loss_history = []
    total_loss_history = []
    lambda_tv=37500000
    lambda_l1 =0.00125
    lambda_spa=0.01125
    batch_size = 1
    is_single_check = 0
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        grad_accum = torch.zeros_like(phase_params)
        num_batches = 0
        if epoch % num_save == 0:
            I_sim_np_all =[]
        for i in tqdm(range(0, len(tilt_angles_total), batch_size)):
            tilt_angles = tilt_angles_total[i:i + batch_size]
            ref_image_torch = [img.to(device) for img in ref_image_cpu[i:i + batch_size]]
            masks = [mask.to(device) for mask in masks_cpu[i:i + batch_size]]
            I_sim, I_sim_np = forward_simulation(phase_params, H1, H2, tilt_angles, grid_size_out, azimuth_angles, ref_image_torch,z_mid,H3)
            if epoch % num_save == 0:
                for i_batch in range(len(I_sim)):
                    I_sim_np_all.append(I_sim_np[i_batch])
            if epoch==0:
                data_loss, fixed_weights, sim_outside_vals = compute_loss_per_item(I_sim,ref_image_torch,masks,epoch=0)
            else:
                data_loss, _, sim_outside_vals = compute_loss_per_item(I_sim,ref_image_torch,masks,epoch=epoch,
        weights=fixed_weights,
        sim_outside_vals=sim_outside_vals
    )
            mapped_phase_all = 4 * target_phase * torch.sigmoid(phase_params) - 0.003 * target_phase
            total_value = torch.sum(torch.abs(mapped_phase_all))+1e-8
            tv_loss = compute_tv_norm(mapped_phase_all, dx=dx, z_step=z_step)
            v_abs = torch.abs(mapped_phase_all)
            eps = 1e-4
            normed_v = v_abs / (v_abs.sum(dim=0, keepdim=True) + eps)
            sum_sq = (normed_v ** 2).sum(dim=0)
            Z = v_abs.size(0)
            spa_loss = torch.sum((1.0 / Z) - sum_sq)
            l1_loss = torch.sum(torch.abs(mapped_phase_all))
            total_loss = data_loss + lambda_tv * tv_loss + lambda_l1 * l1_loss + lambda_spa * spa_loss
            total_loss.backward()
        optimizer.step()
        data_loss_history.append(data_loss.item())
        tv_loss_history.append(tv_loss.item())
        l1_loss_history.append(l1_loss.item())
        spa_loss_history.append(0 if epoch <= 2 else spa_loss.item())
        total_loss_history.append(total_loss.item())
        if (epoch) % num_save == 0:
            with torch.no_grad():
                processed_params = [
                    ((4*target_phase * torch.sigmoid(phase)-0.003*target_phase) / z_step * wavelength / 2/np.pi).detach().cpu().numpy()
                    for phase in phase_params
                ]
                np.save('delta_n_free_optimized.npy', np.array(processed_params))
                print(f"Epoch [{epoch+1}/{num_epochs}] - Data Loss: {data_loss.item():.4e}, TV Loss: {lambda_tv * tv_loss.item():.4e}, Total Loss: {total_loss.item():.4e}, L1 Loss: {lambda_l1 *l1_loss.item():.4e},spa loss:{lambda_spa *spa_loss.item():.4e}")
                data_loss_history_array = np.array(data_loss_history)
                tv_loss_history_array = np.array(tv_loss_history)
                l1_loss_history_array = np.array(l1_loss_history)
                spa_loss_history_array = np.array(spa_loss_history)
                total_loss_history_array = np.array(total_loss_history)
        if (epoch) % 1 == 0 and epoch!=0:
            with torch.no_grad():
                mapped_phase_3d = 4.0 * target_phase * torch.sigmoid(phase_params) - 0.003 * target_phase
                delta_n_3d = mapped_phase_3d.detach().cpu().numpy() * (wavelength/(2*np.pi*z_step))
                from scipy.ndimage import gaussian_filter
                delta_n_3d = np.clip(delta_n_3d, a_min=0, a_max=None)
                delta_n_3d_smoothed = delta_n_3d
                np.save('delta_n_filted_optimized.npy', np.array(delta_n_3d_smoothed))
                mapped_phase_smoothed = delta_n_3d_smoothed*(2*np.pi*z_step)/wavelength
                mapped_phase_torch = torch.from_numpy(mapped_phase_smoothed).float().to(phase_params.device)
                ratio = (mapped_phase_torch + 0.003*target_phase) / (4.0*target_phase + 1e-12)
                ratio_clamped = torch.clamp(ratio, 1e-6, 1 - 1e-6)
                new_phase_params = torch.log(ratio_clamped) - torch.log(1 - ratio_clamped)
                phase_params.data[:] = new_phase_params

if __name__ == "__main__":
    main_optimization_example()