# -*- coding: utf-8 -*-
"""
Análisis espectral (PSD) y correlación intercanal para MI (Left/Right)

Uso:
    python eda.py                     # si estás en la carpeta con left_imag/, right_imag/
    python eda.py --data-root /ruta/a/DATOS_BCI
    python eda.py --output-dir mi_reports  # especificar directorio de salida

Salidas (en directorio 'reports/' por defecto):
  - psd_avg.png                      : PSD promedio (8–30 Hz) con bandas μ/β sombreadas
  - corr_heatmap.png                 : Mapa de calor de correlación intercanal
  - psd_bandpower_per_channel.csv    : Potencia media por banda y picos por canal
  - corr_region_summary.txt          : Resumen de correlaciones por regiones/patrones
"""
from __future__ import annotations

import argparse
import os
import re
from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from mne.io import read_raw_eeglab
from scipy.signal import welch

# ----------------------------
# Parámetros
# ----------------------------
LOW_BAND, HIGH_BAND = 8.0, 30.0
MU_BAND = (10.0, 12.0)
BETA_BAND = (18.0, 26.0)
EXPECTED_TRIAL_SEC = 9.0

CANDIDATE_DIRS: Dict[str, Tuple[str, str]] = {
    "left_imag":  ("left",  "imag"),
    "right_imag": ("right", "imag"),
}

# Regiones aproximadas por prefijos 10–20
REGION_PREFIXES = {
    "F" : ("Fp", "AF", "F"),
    "FC": ("FC",),
    "C" : ("C", "Cz"),
    "CP": ("CP",),
    "P" : ("P",),
    "PO": ("PO",),
    "O" : ("O",),
}

# ----------------------------
# Utilidades
# ----------------------------
def subject_from_fname(fname: str) -> str:
    m = re.search(r"(S\d{3})", os.path.basename(fname))
    return m.group(1) if m else os.path.basename(fname)

def try_read_epochs(fname: str) -> mne.BaseEpochs:
    # 1) Si ya viene epocado
    try:
        ep = mne.read_epochs_eeglab(fname, verbose="ERROR")
        _ = ep.get_data()
        return ep
    except Exception:
        pass
    # 2) Continuo -> ventaneo simple de 9s
    raw = read_raw_eeglab(fname, preload=True, verbose="ERROR")
    try:
        raw.filter(LOW_BAND, HIGH_BAND, verbose="ERROR")
    except Exception:
        pass
    sfreq = float(raw.info["sfreq"])
    n_win = int(np.floor(raw.times[-1] / EXPECTED_TRIAL_SEC))
    if n_win < 1:
        data = np.expand_dims(raw.get_data(), axis=0)          # (1, ch, t)
        return mne.EpochsArray(data, raw.info, tmin=0.0, verbose="ERROR")
    picks = mne.pick_types(raw.info, eeg=True, meg=False, stim=False, eog=False)
    samps = int(EXPECTED_TRIAL_SEC * sfreq)
    data_list: List[np.ndarray] = []
    for i in range(n_win):
        s, e = i * samps, (i + 1) * samps
        if e <= raw.n_times:
            data_list.append(np.expand_dims(raw.get_data(picks=picks)[:, s:e], axis=0))
    data = np.concatenate(data_list, axis=0)                   # (epochs, ch, t)
    info = mne.create_info([raw.ch_names[p] for p in picks], sfreq, ch_types="eeg")
    return mne.EpochsArray(data, info, tmin=0.0, verbose="ERROR")

def pick_region_indices(ch_names: List[str], region_prefixes: Tuple[str, ...]) -> List[int]:
    idx = []
    for i, name in enumerate(ch_names):
        nm = name.upper().replace('Z', 'Z')  # neutral
        if nm.startswith(region_prefixes):
            idx.append(i)
        else:
            # prefijos múltiples
            for pre in region_prefixes:
                if nm.startswith(pre):
                    idx.append(i); break
    return sorted(set(idx))

def band_mask(freqs: np.ndarray, fmin: float, fmax: float) -> np.ndarray:
    return (freqs >= fmin) & (freqs <= fmax)

# ----------------------------
# PSD y correlación
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(Path(__file__).resolve().parent),
        help="Ruta con left_imag/ y right_imag/ (por defecto, carpeta del script).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="Directorio para guardar los reportes (por defecto, 'reports').",
    )
    args = parser.parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"DATA_ROOT = {data_root}")
    print(f"OUTPUT_DIR = {output_dir.resolve()}")

    # 1) Cargar epochs de imaginación (left/right)
    epochs_list: List[mne.BaseEpochs] = []
    for dirname in ["left_imag", "right_imag"]:
        dpath = data_root / dirname
        if not dpath.is_dir():
            print(f"[!] No existe {dpath}, omito.")
            continue
        for set_path in sorted(glob(str(dpath / "*.set"))):
            try:
                ep = try_read_epochs(set_path)
                epochs_list.append(ep)
            except Exception as e:
                print(f"[!] Fallo leyendo {set_path}: {e}")

    if not epochs_list:
        print("[X] No se cargaron epochs.")
        return

    # Concatenar en eje de epochs
    # (asegura mismo canal/orden)
    base_info = epochs_list[0].info
    ch_names = epochs_list[0].ch_names
    sfreq = float(epochs_list[0].info["sfreq"])
    for ep in epochs_list[1:]:
        assert ep.ch_names == ch_names, "Los órdenes de canales difieren entre archivos."
        assert int(ep.info["sfreq"]) == int(sfreq), "sfreq inconsistente."
    X = np.concatenate([ep.get_data() for ep in epochs_list], axis=0)  # (N, ch, T)

    # 2) PSD por canal (Welch)
    # Aplanamos epochs en el eje del tiempo concatenando a lo largo de epochs (por canal)
    N, C, T = X.shape
    data_concat = X.transpose(1, 0, 2).reshape(C, N * T)      # (ch, N*T)

    nperseg = int(sfreq * 2.0)     # 2 s por segmento
    noverlap = int(nperseg * 0.5)  # 50% overlap
    freqs, psd = welch(data_concat, fs=sfreq, nperseg=nperseg, noverlap=noverlap, axis=1)  # freqs: (F,), psd: (ch, F)

    # Limitar a 8–30 Hz
    mask_830 = band_mask(freqs, LOW_BAND, HIGH_BAND)
    freqs_filt = freqs[mask_830]
    psd_filt = psd[:, mask_830]    # (ch, F')

    # Potencia media por banda μ/β y picos
    mu_mask = band_mask(freqs_filt, *MU_BAND)
    be_mask = band_mask(freqs_filt, *BETA_BAND)

    mu_power = psd_filt[:, mu_mask].mean(axis=1)
    be_power = psd_filt[:, be_mask].mean(axis=1)

    # picos (frecuencia y valor) por canal en μ y β
    def peak_in_band(power_row: np.ndarray, faxis: np.ndarray, mask: np.ndarray) -> Tuple[float, float]:
        if not mask.any():
            return (np.nan, np.nan)
        p = power_row[mask]
        f = faxis[mask]
        idx = int(np.argmax(p))
        return float(f[idx]), float(p[idx])

    mu_peaks = [peak_in_band(psd_filt[i], freqs_filt, mu_mask) for i in range(C)]
    be_peaks = [peak_in_band(psd_filt[i], freqs_filt, be_mask) for i in range(C)]

    # Guardar CSV de bandas y picos
    df = pd.DataFrame({
        "channel": ch_names,
        "mu_power": mu_power,
        "beta_power": be_power,
        "mu_peak_hz": [p[0] for p in mu_peaks],
        "mu_peak_val": [p[1] for p in mu_peaks],
        "beta_peak_hz": [p[0] for p in be_peaks],
        "beta_peak_val": [p[1] for p in be_peaks],
    })
    out_csv = output_dir / "psd_bandpower_per_channel.csv"
    df.to_csv(out_csv, index=False)
    print(f"[OK] Guardado: {out_csv.resolve()}")

    # 3) Gráfico PSD promedio con bandas μ/β
    psd_mean = psd_filt.mean(axis=0)  # promedio sobre canales
    plt.figure()
    plt.plot(freqs_filt, psd_mean)
    plt.axvspan(MU_BAND[0], MU_BAND[1], alpha=0.2, label="μ (10–12 Hz)")
    plt.axvspan(BETA_BAND[0], BETA_BAND[1], alpha=0.2, label="β (18–26 Hz)")
    plt.title("PSD promedio (8–30 Hz)")
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("PSD (V²/Hz)")
    plt.legend()
    plt.tight_layout()
    out_png = output_dir / "psd_avg.png"
    plt.savefig(out_png, dpi=140)
    plt.close()
    print(f"[OK] Guardado: {out_png.resolve()}")

    # 4) Correlación intercanal (Pearson) con datos 8–30 Hz
    # Usamos los mismos datos concatenados por canal (data_concat) pero ya filtrados por banda
    # Para “filtrar” por banda en el tiempo necesitaríamos un filtro temporal;
    # como usamos Welch para PSD, para correlación usamos los datos temporales originales (ya prefiltrados 8–30 en Raw si fue continuo).
    # Si los epochs ya venían filtrados en origen, seguimos consistentes.

    # Reconstituimos señal 8–30 Hz por canal uniendo epochs (ya prefiltrados si eran continuos)
    # Si venían epocados prefiltrados: asumimos 8–30 del preprocesamiento de origen.
    data_for_corr = data_concat  # (ch, N*T)
    corr = np.corrcoef(data_for_corr)  # (ch, ch)

    # Heatmap
    plt.figure(figsize=(7, 6))
    plt.imshow(corr, vmin=-1, vmax=1, aspect='auto')
    plt.title("Correlación intercanal")
    plt.xlabel("Canal")
    plt.ylabel("Canal")
    plt.colorbar(label="r")
    plt.tight_layout()
    out_heatmap = output_dir / "corr_heatmap.png"
    plt.savefig(out_heatmap, dpi=140)
    plt.close()
    print(f"[OK] Guardado: {out_heatmap.resolve()}")

    # 5) Resumen por regiones (promedios de r)
    # Índices por región
    ch_upper = [c.upper() for c in ch_names]
    region_indices: Dict[str, List[int]] = {}
    for reg, prefixes in REGION_PREFIXES.items():
        idxs = []
        for i, c in enumerate(ch_upper):
            if any(c.startswith(p) for p in prefixes):
                idxs.append(i)
        region_indices[reg] = idxs

    def mean_block(A: np.ndarray, rows: List[int], cols: List[int]) -> float:
        if not rows or not cols:
            return np.nan
        sub = A[np.ix_(rows, cols)]
        return float(np.nanmean(sub))

    # pares de interés
    pairs = [
        ("F", "C"), ("FC", "C"), ("C", "CP"), ("C", "P"), ("F", "P"), ("P", "O")
    ]
    lines = []
    lines.append("=== Resumen de correlaciones por región ===")
    global_mean = float(np.nanmean(corr))
    lines.append(f"Global mean r: {global_mean:.3f}")
    for a, b in pairs:
        ra, rb = region_indices.get(a, []), region_indices.get(b, [])
        rmean = mean_block(corr, ra, rb)
        lines.append(f"{a}-{b}: mean r = {rmean:.3f}  (n_a={len(ra)}, n_b={len(rb)})")

    # intra-región (opcional)
    for a in ["F", "C", "P", "O"]:
        ra = region_indices.get(a, [])
        rmean = mean_block(corr, ra, ra)
        lines.append(f"{a}-{a}: mean r = {rmean:.3f}  (n={len(ra)})")

    out_txt = output_dir / "corr_region_summary.txt"
    out_txt.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] Guardado: {out_txt.resolve()}")

    # Print breve en consola
    print("\n".join(lines))


if __name__ == "__main__":
    main()
