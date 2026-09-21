"""Source-faithful RELION scoring operands shared by coarse and fine search."""

from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from recovar.em.dense_single_volume.helpers.dataset_indexing import original_indices_for_local
from recovar.em.dense_single_volume.runtime_options import current_environment


def relion_translation_angles_f32(translations, image_shape):
    """Return RELION fine-score ``(tx, ty)`` radians with host rounding."""

    image_size = int(image_shape[0])
    if image_size <= 0:
        raise ValueError(f"image_shape must be positive, got {image_shape}")
    translations_f64 = np.asarray(translations, dtype=np.float64)
    if translations_f64.ndim != 2 or translations_f64.shape[1] != 2:
        raise ValueError("RELION score translations must have shape (T, 2), got " f"{translations_f64.shape}")
    return np.asarray(
        -2.0 * np.pi * translations_f64 / float(image_size),
        dtype=np.float32,
    )


def relion_translation_angles_f64(translations, image_shape):
    """Return RELION double-ACC ``(tx, ty)`` translation radians."""

    image_size = int(image_shape[0])
    if image_size <= 0:
        raise ValueError(f"image_shape must be positive, got {image_shape}")
    translations_f64 = np.asarray(translations, dtype=np.float64)
    if translations_f64.ndim != 2 or translations_f64.shape[1] != 2:
        raise ValueError("RELION score translations must have shape (T, 2), got " f"{translations_f64.shape}")
    return -2.0 * np.pi * translations_f64 / float(image_size)


def relion_cuda_fine_full_to_compact_lookup(image_shape, current_size, compact_indices):
    """Map RELION's full current-size packed pixel order to compact score rows."""

    image_size = int(image_shape[0])
    current_size = image_size if current_size is None else int(current_size)
    original_half_width = int(image_shape[1]) // 2 + 1
    current_half_width = current_size // 2 + 1
    compact_indices = np.asarray(compact_indices, dtype=np.int64).reshape(-1)
    centered_rows = compact_indices // original_half_width
    columns = compact_indices % original_half_width
    ky = centered_rows - image_size // 2
    fftw_rows = np.where(ky < 0, ky + current_size, ky)
    if np.any(fftw_rows < 0) or np.any(fftw_rows >= current_size):
        raise ValueError("compact score indices contain rows outside the RELION current-size crop")
    if np.any(columns < 0) or np.any(columns >= current_half_width):
        raise ValueError("compact score indices contain columns outside the RELION current-size crop")
    relion_flat_indices = fftw_rows * current_half_width + columns
    if np.unique(relion_flat_indices).size != relion_flat_indices.size:
        raise ValueError("compact score indices do not map uniquely into RELION's current-size layout")
    lookup = np.full(current_size * current_half_width, -1, dtype=np.int32)
    lookup[relion_flat_indices] = np.arange(compact_indices.size, dtype=np.int32)
    return lookup


def relion_cuda_corr_img_from_rfloat_ctf(
    inverse_noise,
    ctf_rfloat,
    scale=None,
    *,
    output_dtype=jnp.float32,
):
    """Form XFLOAT ``corr_img`` after RELION's RFLOAT CTF square."""

    output_dtype = jnp.dtype(output_dtype)
    if output_dtype not in (jnp.dtype(jnp.float32), jnp.dtype(jnp.float64)):
        raise TypeError(f"output_dtype must be float32 or float64, got {output_dtype}")
    inverse_noise_rfloat = jnp.asarray(inverse_noise, dtype=output_dtype).astype(jnp.float64)
    ctf_rfloat = jnp.asarray(ctf_rfloat, dtype=jnp.float64)
    ctf_squared_rfloat = jax.lax.optimization_barrier(ctf_rfloat * ctf_rfloat)
    corr_img = jax.lax.optimization_barrier(inverse_noise_rfloat * ctf_squared_rfloat).astype(output_dtype)
    if scale is not None:
        scale = jnp.asarray(scale, dtype=output_dtype)
        scale_squared = jax.lax.optimization_barrier(scale * scale)
        corr_img = corr_img * scale_squared
    return corr_img


def relion_cuda_corr_img_from_native_noise_variance(
    noise_variance,
    ctf_rfloat,
    image_shape,
    scale=None,
    *,
    output_dtype=jnp.float32,
):
    """Form score-unit ``corr_img`` with RELION's native-FFT cast order."""

    output_dtype = jnp.dtype(output_dtype)
    if output_dtype not in (jnp.dtype(jnp.float32), jnp.dtype(jnp.float64)):
        raise TypeError(f"output_dtype must be float32 or float64, got {output_dtype}")
    image_size = int(image_shape[0])
    if tuple(image_shape) != (image_size, image_size):
        raise ValueError(f"RELION corr_img requires a square image, got {image_shape}")
    native_fourier_scale_rfloat = jnp.asarray(image_size**4, dtype=jnp.float64)
    native_variance = jnp.asarray(noise_variance, dtype=jnp.float64) / native_fourier_scale_rfloat
    native_inverse_noise = jnp.reciprocal(native_variance).astype(output_dtype)
    native_corr_img = relion_cuda_corr_img_from_rfloat_ctf(
        native_inverse_noise,
        ctf_rfloat,
        scale,
        output_dtype=output_dtype,
    )
    return (native_corr_img.astype(jnp.float64) / native_fourier_scale_rfloat).astype(output_dtype)


def relion_cuda_pixel_correction_from_rfloat_ctf(
    scale,
    ctf_rfloat,
    *,
    output_dtype=jnp.float32,
):
    """Form RELION's XFLOAT score-image correction from an RFLOAT CTF."""

    output_dtype = jnp.dtype(output_dtype)
    if output_dtype not in (jnp.dtype(jnp.float32), jnp.dtype(jnp.float64)):
        raise TypeError(f"output_dtype must be float32 or float64, got {output_dtype}")
    scale = jnp.asarray(scale, dtype=output_dtype)
    ctf_rfloat = jnp.asarray(ctf_rfloat, dtype=jnp.float64)
    pixel_correction = jax.lax.optimization_barrier(jnp.reciprocal(scale))
    corrected = jax.lax.optimization_barrier(pixel_correction.astype(jnp.float64) / ctf_rfloat).astype(output_dtype)
    return jnp.where(jnp.abs(ctf_rfloat) > 1e-8, corrected, pixel_correction)


_RELION_CUDA_POWERCLASS_BLOCK_SIZE = 128


@partial(jax.jit, static_argnames=("image_shape", "current_size"))
def relion_cuda_powerclass_highres_xi2_half(
    processed_score_half,
    *,
    image_shape,
    current_size,
):
    """Reproduce the class-power high-resolution image tail used by fine diff2."""

    image_height = int(image_shape[0])
    image_width = int(image_shape[1])
    if image_height != image_width:
        raise ValueError(f"RELION powerClass parity requires square images, got {image_shape}")
    half_width = image_width // 2 + 1
    processed_score_half = jnp.asarray(processed_score_half)
    complex_dtype = jnp.complex128 if processed_score_half.dtype == jnp.dtype(jnp.complex128) else jnp.complex64
    real_dtype = jnp.float64 if complex_dtype == jnp.complex128 else jnp.float32
    processed_score_half = processed_score_half.astype(complex_dtype)
    if processed_score_half.ndim != 2 or processed_score_half.shape[-1] != image_height * half_width:
        raise ValueError(
            "RELION powerClass input must be flattened centred rfft images, got "
            f"{processed_score_half.shape} for image_shape={image_shape}"
        )
    if current_size is None:
        current_size = image_width
    resolution_limit = int(current_size) // 2 + 1

    relion_image = jnp.roll(
        processed_score_half.reshape((-1, image_height, half_width)),
        -(image_height // 2),
        axis=1,
    ).reshape((processed_score_half.shape[0], -1))
    relion_image = relion_image / jnp.asarray(image_height * image_width, dtype=real_dtype)

    rows = np.arange(image_height, dtype=np.int32)[:, None]
    columns = np.arange(half_width, dtype=np.int32)[None, :]
    signed_rows = np.where(rows < half_width, rows, rows - image_height)
    radius_squared = columns * columns + signed_rows * signed_rows
    shell_real_dtype = np.float64 if real_dtype == jnp.float64 else np.float32
    shell = np.rint(np.sqrt(radius_squared.astype(shell_real_dtype))).astype(np.int32)
    valid = (
        (shell > 0) & (shell < half_width) & ~((columns == 0) & (signed_rows < 0)) & (shell >= resolution_limit)
    ).reshape(-1)

    power = relion_image.real * relion_image.real
    power = jax.lax.optimization_barrier(power)
    power = power + relion_image.imag * relion_image.imag
    power = jnp.where(jnp.asarray(valid)[None, :], power, jnp.asarray(0.0, dtype=real_dtype))

    block_size = _RELION_CUDA_POWERCLASS_BLOCK_SIZE
    n_blocks = (power.shape[-1] + block_size - 1) // block_size
    power = jnp.pad(power, ((0, 0), (0, n_blocks * block_size - power.shape[-1])))
    block_lanes = power.reshape((power.shape[0], n_blocks, block_size))
    for width in (64, 32, 16, 8, 4, 2, 1):
        block_lanes = block_lanes[..., :width] + block_lanes[..., width : 2 * width]
        block_lanes = jax.lax.optimization_barrier(block_lanes)
    block_sums = block_lanes[..., 0]

    def add_block(block_index, total):
        total = total + block_sums[:, block_index]
        return jax.lax.optimization_barrier(total)

    highres_xi2 = jax.lax.fori_loop(
        0,
        n_blocks,
        add_block,
        jnp.zeros((processed_score_half.shape[0],), dtype=real_dtype),
    )
    return highres_xi2 * jnp.asarray(0.5, dtype=real_dtype)


_RELION_EXACT_CTF_SOURCE_CACHE: dict[tuple[str, tuple[int, int]], dict] = {}


def _star_column(table, name: str):
    for candidate in (name, f"_{name}"):
        if candidate in table.columns:
            return table[candidate]
    raise ValueError(f"RELION source STAR column {name} is missing")


def _relion_exact_ctf_source_star(experiment_dataset) -> Path:
    source_star = current_environment().get("RECOVAR_K1_RELION_EXACT_CTF_STAR", "").strip()
    if not source_star:
        dataset_source = getattr(experiment_dataset, "particles_file", None)
        if dataset_source and Path(dataset_source).suffix.lower() == ".star":
            source_star = str(dataset_source)
    if not source_star:
        raise ValueError("exact RELION operands require a STAR-backed dataset or " "RECOVAR_K1_RELION_EXACT_CTF_STAR")
    return Path(source_star).expanduser().resolve()


def relion_exact_ctf_half_from_source_star(
    experiment_dataset,
    image_indices,
    image_shape,
):
    """Evaluate source-precision SPA CTFs with RELION's scalar implementation."""

    source_path = _relion_exact_ctf_source_star(experiment_dataset)
    cache_key = (str(source_path), tuple(int(size) for size in image_shape))
    cache = _RELION_EXACT_CTF_SOURCE_CACHE.get(cache_key)
    if cache is None:
        from recovar.data_io.starfile import read_star
        from recovar.relion_bind import _relion_bind_core as relion_bind

        particles, optics = read_star(str(source_path))
        if optics is None:
            raise ValueError(f"RELION source STAR has no optics table: {source_path}")
        optics_ids = np.asarray(_star_column(optics, "rlnOpticsGroup"), dtype=np.int64)
        if np.unique(optics_ids).size != optics_ids.size:
            raise ValueError(f"RELION source STAR has duplicate optics groups: {source_path}")
        cache = {
            "particles": particles,
            "optics": {int(group): optics.iloc[row] for row, group in enumerate(optics_ids)},
            "relion_bind": relion_bind,
            "images": {},
        }
        _RELION_EXACT_CTF_SOURCE_CACHE[cache_key] = cache

    original_indices = original_indices_for_local(
        experiment_dataset,
        np.asarray(image_indices, dtype=np.int64),
    )
    image_h, image_w = (int(size) for size in image_shape)
    if image_h != image_w:
        raise ValueError("exact RELION CTF replay currently requires square images")
    ctf_rows = []
    for original_index in original_indices:
        original_index = int(original_index)
        cached_image = cache["images"].get(original_index)
        if cached_image is None:
            particle = cache["particles"].iloc[original_index]
            optics_group = int(
                particle["rlnOpticsGroup"] if "rlnOpticsGroup" in particle else particle["_rlnOpticsGroup"]
            )
            optics = cache["optics"][optics_group]

            def particle_value(name: str) -> float:
                return float(particle[name] if name in particle else particle[f"_{name}"])

            def optics_value(name: str) -> float:
                return float(optics[name] if name in optics else optics[f"_{name}"])

            native = np.asarray(
                cache["relion_bind"].get_ctf_image(
                    particle_value("rlnDefocusU"),
                    particle_value("rlnDefocusV"),
                    particle_value("rlnDefocusAngle"),
                    optics_value("rlnVoltage"),
                    optics_value("rlnSphericalAberration"),
                    optics_value("rlnAmplitudeContrast"),
                    0.0,
                    optics_value("rlnImagePixelSize"),
                    image_w,
                    image_h,
                    False,
                    False,
                    False,
                    particle_value("rlnPhaseShift"),
                    1.0,
                ),
                dtype=np.float64,
            )
            cached_image = (-np.fft.fftshift(native, axes=0)).reshape(-1)
            cache["images"][original_index] = cached_image
        ctf_rows.append(cached_image)
    return jnp.asarray(np.stack(ctf_rows, axis=0), dtype=jnp.float64)
