"""
modules/file_ingestion.py
Multi-format petroleum data ingestion engine.
Supported: xlsx, xls, csv, las, grdecl, npy/npz, dcm/dcom, jpg/jpeg/png/img, nic, txt/dat
"""

import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

# ── Optional library guards ───────────────────────────────────────────────────
try:
    import lasio
    HAS_LASIO = True
except ImportError:
    HAS_LASIO = False

try:
    import pydicom
    HAS_DICOM = True
except ImportError:
    HAS_DICOM = False

try:
    import imageio.v3 as iio
    from PIL import Image as PILImage
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

from config import CSV_MAP, LAS_MAP


# ─────────────────────────────────────────────────────────────────────────────
#  DATA BUNDLE  — standardised output container
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class DataBundle:
    source:       str                       = "unknown"
    filename:     str                       = ""
    file_type:    str                       = ""
    well_name:    str                       = ""
    # Data tables
    reservoir:    Optional[pd.DataFrame]    = None   # per-zone / per-sample rock props
    completions:  Optional[pd.DataFrame]    = None   # perforation / completion design
    production:   Optional[pd.DataFrame]    = None   # time-series production history
    well_logs:    Optional[pd.DataFrame]    = None   # depth-indexed log curves
    grid:         Optional[dict]            = None   # 3D grid (GRDECL / npy)
    images:       list                      = field(default_factory=list)  # PIL Images
    raw:          Any                       = None
    metadata:     dict                      = field(default_factory=dict)
    warnings:     list                      = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"Source : {self.source} | File : {self.filename}"]
        if self.well_name:
            lines.append(f"Well   : {self.well_name}")
        for tbl, name in [(self.reservoir,"Reservoir"), (self.completions,"Completions"),
                          (self.production,"Production"), (self.well_logs,"Well Logs")]:
            if tbl is not None and len(tbl) > 0:
                lines.append(f"{name:12}: {tbl.shape[0]} rows × {tbl.shape[1]} cols")
        if self.grid:
            lines.append(f"Grid   : {self.grid.get('dims','?')}")
        if self.images:
            lines.append(f"Images : {len(self.images)} loaded")
        return "\n".join(lines)

    def all_data(self) -> Optional[pd.DataFrame]:
        """Merge available tabular data into one DataFrame for modelling."""
        frames = [f for f in [self.reservoir, self.completions, self.well_logs]
                  if f is not None and len(f) > 0]
        if not frames:
            return None
        if len(frames) == 1:
            return frames[0]
        base = frames[0].copy()
        for f in frames[1:]:
            for col in f.columns:
                if col not in base.columns:
                    if len(f) == len(base):
                        base[col] = f[col].values
                    else:
                        base[col] = f.iloc[0][col] if len(f) > 0 else np.nan
        return base


# ─────────────────────────────────────────────────────────────────────────────
#  COLUMN NORMALISER
# ─────────────────────────────────────────────────────────────────────────────
def _normalise_columns(df: pd.DataFrame, col_map: dict) -> pd.DataFrame:
    """Rename known aliases to standard column names (case-insensitive)."""
    upper_map = {k.upper(): v for k, v in col_map.items()}
    rename = {}
    for col in df.columns:
        upper = col.strip().upper()
        if upper in upper_map:
            rename[col] = upper_map[upper]
    if rename:
        df = df.rename(columns=rename)
    return df


def _classify_columns(df: pd.DataFrame) -> dict:
    """Guess which columns belong to reservoir / completion / production."""
    res_kw  = {"permeability","porosity","net_pay","pressure","temperature",
               "viscosity","api","gor","compressibility","depth"}
    perf_kw = {"spf","phasing","pen_depth","hole_dia","underbalance",
               "cluster","stages","perforation"}
    prod_kw = {"q_oil","q_water","q_gas","water_cut","bhfp","gor","cum",
               "rate","time","date"}

    def score(col, kw_set):
        col_l = col.lower().replace("_","")
        return sum(1 for kw in kw_set if kw.replace("_","") in col_l)

    res_cols  = [c for c in df.columns if score(c, res_kw)  > 0]
    perf_cols = [c for c in df.columns if score(c, perf_kw) > 0]
    prod_cols = [c for c in df.columns if score(c, prod_kw) > 0]
    return {"reservoir": res_cols, "completions": perf_cols, "production": prod_cols}


# ─────────────────────────────────────────────────────────────────────────────
#  INGESTION ENGINE
# ─────────────────────────────────────────────────────────────────────────────
class FileIngestionEngine:

    # ── public entry point ────────────────────────────────────────────────────
    def ingest(self, file_obj, filename: str) -> DataBundle:
        ext  = Path(filename).suffix.lower()
        name = Path(filename).stem
        readers = {
            ".csv":    self._read_csv,
            ".txt":    self._read_csv,
            ".dat":    self._read_csv,
            ".xlsx":   self._read_excel,
            ".xls":    self._read_excel,
            ".las":    self._read_las,
            ".grdecl": self._read_grdecl,
            ".npy":    self._read_npy,
            ".npz":    self._read_npz,
            ".dcm":    self._read_dicom,
            ".dcom":   self._read_dicom,
            ".jpg":    self._read_image,
            ".jpeg":   self._read_image,
            ".png":    self._read_image,
            ".img":    self._read_raw_image,
            ".nic":    self._read_nexus,
        }
        reader = readers.get(ext, self._read_csv)
        try:
            bundle = reader(file_obj, filename)
            bundle.file_type = ext.lstrip(".")
            bundle.filename  = filename
            if not bundle.well_name:
                bundle.well_name = name   # fall back to the filename stem
            return bundle
        except Exception as exc:
            b = DataBundle(source="error", filename=filename)
            b.warnings.append(f"Failed to read {filename}: {exc}")
            return b

    # ═════════════════════════════════════════════════════════════════════════
    # CSV / TXT / DAT
    # ═════════════════════════════════════════════════════════════════════════
    def _read_csv(self, file_obj, filename: str) -> DataBundle:
        raw_bytes = file_obj.read()
        # Try multiple delimiters
        for sep in [",", "\t", ";", " ", "|"]:
            try:
                df = pd.read_csv(io.BytesIO(raw_bytes), sep=sep,
                                 comment="#", skipinitialspace=True,
                                 skip_blank_lines=True, engine="python")
                if df.shape[1] > 1:
                    break
            except Exception:
                continue
        df = _normalise_columns(df, CSV_MAP)
        cats = _classify_columns(df)
        b = DataBundle(source="csv")
        b.reservoir   = df[cats["reservoir"]].copy()   if cats["reservoir"]  else df
        b.completions = df[cats["completions"]].copy() if cats["completions"] else None
        b.production  = df[cats["production"]].copy()  if cats["production"]  else None
        b.raw         = df
        b.metadata    = {"rows": len(df), "cols": df.shape[1], "delimiter": sep}
        return b

    # ═════════════════════════════════════════════════════════════════════════
    # EXCEL (.xlsx / .xls)
    # ═════════════════════════════════════════════════════════════════════════
    def _read_excel(self, file_obj, filename: str) -> DataBundle:
        raw_bytes = file_obj.read()
        xl  = pd.ExcelFile(io.BytesIO(raw_bytes))
        b   = DataBundle(source="excel")
        sheets = {}
        for sname in xl.sheet_names:
            try:
                df = pd.read_excel(io.BytesIO(raw_bytes), sheet_name=sname,
                                   header=0, skiprows=0)
                df = _normalise_columns(df, CSV_MAP)
                df.dropna(how="all", inplace=True)
                sheets[sname] = df
            except Exception as e:
                b.warnings.append(f"Sheet '{sname}' skipped: {e}")

        b.metadata = {"sheets": list(sheets.keys())}
        # Heuristic sheet assignment
        for sname, df in sheets.items():
            sl = sname.lower()
            if any(k in sl for k in ["reservoir","rock","poro","perm","formation"]):
                b.reservoir = df
            elif any(k in sl for k in ["perf","complet","well","design"]):
                b.completions = df
            elif any(k in sl for k in ["prod","rate","history","time","decline"]):
                b.production = df
            elif any(k in sl for k in ["log","depth","curve","las"]):
                b.well_logs = df
            elif b.reservoir is None:
                b.reservoir = df
        if b.reservoir is None and sheets:
            b.reservoir = next(iter(sheets.values()))
        b.raw = sheets
        return b

    # ═════════════════════════════════════════════════════════════════════════
    # LAS WELL LOG  (CWLS LAS 2.0 / 3.0)
    # ═════════════════════════════════════════════════════════════════════════
    def _read_las(self, file_obj, filename: str) -> DataBundle:
        raw_bytes = file_obj.read()
        b = DataBundle(source="las")

        if HAS_LASIO:
            try:
                las  = lasio.read(io.StringIO(raw_bytes.decode("utf-8","ignore")))
                df   = las.df().reset_index()
                df   = _normalise_columns(df, LAS_MAP)
                b.well_name = getattr(las.well.get("WELL"), "value", filename)
                b.well_logs = df
                b.metadata  = {
                    "curves":   [c.mnemonic for c in las.curves],
                    "well":     str(b.well_name),
                    "start_depth": float(las.well.get("STRT", {"value": 0}).value),
                    "stop_depth":  float(las.well.get("STOP", {"value": 0}).value),
                    "step":        float(las.well.get("STEP", {"value": 0}).value),
                }
                # Derive reservoir properties from log
                b.reservoir = self._logs_to_reservoir(df)
                b.raw = las
                return b
            except Exception as e:
                b.warnings.append(f"lasio failed, falling back to basic parser: {e}")

        # ── Basic LAS 2.0 fallback parser ────────────────────────────────────
        text = raw_bytes.decode("utf-8", "ignore")
        b.well_logs = self._parse_las_basic(text)
        b.well_name = filename
        b.reservoir = self._logs_to_reservoir(b.well_logs)
        return b

    def _parse_las_basic(self, text: str) -> pd.DataFrame:
        """Minimal LAS 2.0 data-section reader."""
        lines  = text.splitlines()
        data_start = None
        headers    = []
        for i, line in enumerate(lines):
            if line.strip().startswith("~A"):
                data_start = i + 1
                break
            if line.strip().startswith("~C"):
                # Collect curve names in subsequent non-~ lines
                for j in range(i+1, len(lines)):
                    l = lines[j].strip()
                    if l.startswith("~"):
                        break
                    if l and not l.startswith("#"):
                        mnem = l.split(".")[0].strip()
                        if mnem:
                            headers.append(mnem)
        if data_start is None:
            return pd.DataFrame()
        rows = []
        for line in lines[data_start:]:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                row = [float(v) for v in line.split()]
                rows.append(row)
            except ValueError:
                pass
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows, columns=headers[:len(rows[0])] if headers else None)
        return _normalise_columns(df, LAS_MAP)

    def _logs_to_reservoir(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derive reservoir properties from well log curves (Wyllie, Archie)."""
        if df is None or len(df) == 0:
            return pd.DataFrame()
        res = {}
        # Porosity from density log: φ = (ρma - ρb) / (ρma - ρfl)
        if "RHOB_gcc" in df.columns:
            rho_ma, rho_fl = 2.65, 1.00
            res["porosity"] = ((rho_ma - df["RHOB_gcc"]) /
                               (rho_ma - rho_fl)).clip(0.05, 0.45)
        # Porosity from neutron
        elif "NPHI_pu" in df.columns:
            res["porosity"] = (df["NPHI_pu"] / 100).clip(0.05, 0.45)

        # Water saturation from Archie: Sw = (a/(Rt*φ^m))^(1/n), a=1, m=2, n=2
        if "Rt_ohmm" in df.columns and "porosity" in res:
            Rw = 0.05  # formation water resistivity (Ω·m), typical
            phi = res["porosity"]
            res["Sw_frac"] = np.sqrt(Rw / (df["Rt_ohmm"] * phi**2)).clip(0.1, 1.0)
            res["So_frac"] = 1 - res["Sw_frac"]

        # GR-based Vshale
        if "GR_api" in df.columns:
            gr_clean, gr_shale = df["GR_api"].quantile(0.05), df["GR_api"].quantile(0.95)
            res["Vshale"] = ((df["GR_api"] - gr_clean) /
                             (gr_shale - gr_clean)).clip(0, 1)

        # Young's modulus from sonic (simplified):  E ≈ ρV²
        if "DT_usft" in df.columns and "RHOB_gcc" in df.columns:
            Vp = 1e6 / df["DT_usft"]  # ft/s → ft/s
            E_dyn_psi = df["RHOB_gcc"] * 62.4 * Vp**2 / 1000  # simplified
            res["youngs_modulus_GPa"] = (E_dyn_psi * 6.895e-9).clip(1, 100)

        if "depth_ft" in df.columns:
            res["depth_ft"] = df["depth_ft"]

        return pd.DataFrame(res)

    # ═════════════════════════════════════════════════════════════════════════
    # GRDECL  (Eclipse / INTERSECT corner-point grid)
    # ═════════════════════════════════════════════════════════════════════════
    def _read_grdecl(self, file_obj, filename: str) -> DataBundle:
        text = file_obj.read().decode("utf-8", "ignore")
        b    = DataBundle(source="grdecl")

        def _parse_kw(keyword):
            pattern = rf'(?:^|\s){keyword}\s*(.*?)\s*/'
            m = re.search(pattern, text, re.DOTALL | re.IGNORECASE | re.MULTILINE)
            if not m:
                return None
            vals_str = m.group(1).strip()
            vals = []
            for tok in vals_str.split():
                if '*' in tok:
                    parts = tok.split('*')
                    try:
                        n, v = int(parts[0]), float(parts[1])
                        vals.extend([v]*n)
                    except (ValueError, IndexError):
                        pass
                else:
                    try:
                        vals.append(float(tok))
                    except ValueError:
                        pass
            return np.array(vals)

        # Grid dimensions
        sg = _parse_kw("SPECGRID")
        if sg is not None and len(sg) >= 3:
            NI, NJ, NK = int(sg[0]), int(sg[1]), int(sg[2])
        else:
            NI = NJ = NK = 10  # default fallback

        kw_data = {}
        for kw in ["ACTNUM","PORO","PERMX","PERMY","PERMZ","SWAT","SOIL","SGAS",
                   "PRESSURE","PVTNUM","SATNUM"]:
            arr = _parse_kw(kw)
            if arr is not None:
                kw_data[kw] = arr

        b.grid = {"dims": (NI, NJ, NK), "data": kw_data,
                  "n_cells": NI * NJ * NK}

        # Flatten to reservoir DataFrame
        n = NI * NJ * NK
        res_rows = {}
        for kw, arr in kw_data.items():
            if len(arr) == n:
                res_rows[kw.lower()] = arr
        if res_rows:
            df_res = pd.DataFrame(res_rows)
            # Apply ACTNUM filter
            if "actnum" in df_res.columns:
                df_res = df_res[df_res["actnum"] > 0].reset_index(drop=True)
            df_res = _normalise_columns(df_res, {
                "poro":"porosity","permx":"permeability_mD",
                "swat":"Sw_frac","pressure":"initial_pressure_psi"
            })
            b.reservoir = df_res

        b.metadata = {"NI":NI,"NJ":NJ,"NK":NK,"keywords":list(kw_data.keys())}
        b.raw = kw_data
        return b

    # ═════════════════════════════════════════════════════════════════════════
    # NUMPY  (.npy / .npz)
    # ═════════════════════════════════════════════════════════════════════════
    def _read_npy(self, file_obj, filename: str) -> DataBundle:
        raw = file_obj.read()
        arr = np.load(io.BytesIO(raw), allow_pickle=True)
        return self._npy_to_bundle(arr, filename)

    def _read_npz(self, file_obj, filename: str) -> DataBundle:
        raw  = file_obj.read()
        npz  = np.load(io.BytesIO(raw), allow_pickle=True)
        b    = DataBundle(source="npz")
        frames = {}
        for key in npz.files:
            arr = npz[key]
            if arr.ndim == 2:
                frames[key] = pd.DataFrame(arr, columns=[f"{key}_{i}" for i in range(arr.shape[1])])
        if frames:
            b.reservoir = next(iter(frames.values()))
        b.grid    = {"arrays": {k: npz[k] for k in npz.files}}
        b.raw     = npz
        b.metadata = {"keys": list(npz.files)}
        return b

    def _npy_to_bundle(self, arr, filename):
        b = DataBundle(source="npy")
        if arr.ndim == 1:
            b.reservoir = pd.DataFrame({"value": arr})
            b.metadata  = {"shape": arr.shape, "interpretation": "1-D log or time series"}
        elif arr.ndim == 2:
            b.reservoir = pd.DataFrame(arr, columns=[f"col_{i}" for i in range(arr.shape[1])])
            b.metadata  = {"shape": arr.shape, "interpretation": "2-D table"}
        elif arr.ndim == 3:
            b.grid = {"array": arr, "dims": arr.shape}
            # Flatten to DataFrame for modelling
            flat = arr.reshape(-1, arr.shape[-1]) if arr.ndim == 3 and arr.shape[2] < 20 else arr.ravel().reshape(-1,1)
            b.reservoir = pd.DataFrame(flat, columns=[f"feature_{i}" for i in range(flat.shape[1])])
            b.metadata  = {"shape": arr.shape, "interpretation": "3-D volume (seismic/grid)"}
        b.raw = arr
        return b

    # ═════════════════════════════════════════════════════════════════════════
    # DICOM  (.dcm / .dcom)  — CT core scans
    # ═════════════════════════════════════════════════════════════════════════
    def _read_dicom(self, file_obj, filename: str) -> DataBundle:
        raw = file_obj.read()
        b   = DataBundle(source="dicom")

        if not HAS_DICOM:
            b.warnings.append("pydicom not installed — CT scan pixel data unavailable")
            return b

        ds  = pydicom.dcmread(io.BytesIO(raw), force=True)
        b.metadata = {
            "patient_id":    getattr(ds, "PatientID", "unknown"),
            "study_date":    getattr(ds, "StudyDate", ""),
            "modality":      getattr(ds, "Modality", "CT"),
            "rows":          getattr(ds, "Rows", 0),
            "cols":          getattr(ds, "Columns", 0),
            "slice_thick":   getattr(ds, "SliceThickness", 1.0),
        }
        if hasattr(ds, "pixel_array"):
            pix = ds.pixel_array.astype(float)
            # Hounsfield units
            slope     = float(getattr(ds, "RescaleSlope", 1))
            intercept = float(getattr(ds, "RescaleIntercept", 0))
            hu        = pix * slope + intercept

            b.metadata["HU_min"]  = float(hu.min())
            b.metadata["HU_max"]  = float(hu.max())
            b.metadata["HU_mean"] = float(hu.mean())

            # Convert HU → porosity (limestone transform: HU = -1000 + 1000*φ roughly)
            # More realistic: φ = (HU_matrix - HU_measured)/(HU_matrix - HU_fluid)
            HU_matrix = 2710   # calcite/dolomite
            HU_fluid  = -1000  # air (dry core)  or 0 (water-saturated)
            phi_ct    = ((HU_matrix - hu) / (HU_matrix - HU_fluid)).clip(0, 0.50)

            b.reservoir = pd.DataFrame({
                "HU":       hu.ravel(),
                "porosity": phi_ct.ravel(),
                "depth_mm": np.arange(len(hu.ravel())) * float(b.metadata["slice_thick"]),
            })
            # Store image
            pix_norm = ((hu - hu.min()) / max(hu.max() - hu.min(), 1) * 255).astype(np.uint8)
            b.images.append(pix_norm)

        b.raw = ds
        return b

    # ═════════════════════════════════════════════════════════════════════════
    # IMAGE files  (.jpg / .jpeg / .png)
    # ═════════════════════════════════════════════════════════════════════════
    def _read_image(self, file_obj, filename: str) -> DataBundle:
        raw = file_obj.read()
        b   = DataBundle(source="image")

        if not HAS_PIL:
            b.warnings.append("Pillow not installed — image processing unavailable")
            return b

        try:
            img = PILImage.open(io.BytesIO(raw)).convert("L")  # grayscale
            arr = np.array(img, dtype=np.float32)
            width, height = img.width, img.height
        except Exception as pil_exc:
            # v3: fall back to imageio for formats Pillow can't decode
            # (e.g. some multi-page TIFF / scientific formats) — imageio
            # was imported in v2 and never actually used anywhere.
            try:
                arr2 = iio.imread(io.BytesIO(raw))
                if arr2.ndim == 3:
                    arr2 = arr2.mean(axis=2)   # collapse to grayscale
                arr = arr2.astype(np.float32)
                height, width = arr.shape[:2]
                b.warnings.append(f"Pillow could not read this image ({pil_exc}); used imageio fallback")
            except Exception as io_exc:
                b.warnings.append(f"Could not decode image with Pillow or imageio: {io_exc}")
                return b

        b.metadata = {"width": width, "height": height,
                      "mode": "grayscale", "mean_intensity": float(arr.mean())}
        b.images   = [arr]

        # Interpret horizontal rows as depth slices, columns as lateral position
        # Row-mean intensity ≈ proxy for porosity variation with depth
        row_mean = arr.mean(axis=1)
        phi_proxy = (255 - row_mean) / 255 * 0.35 + 0.05  # dark=low porosity
        b.reservoir = pd.DataFrame({
            "pixel_row":    np.arange(len(row_mean)),
            "mean_intensity": row_mean,
            "porosity_proxy": phi_proxy,
        })
        b.raw = arr
        return b

    # ═════════════════════════════════════════════════════════════════════════
    # RAW BINARY IMAGE  (.img)
    # ═════════════════════════════════════════════════════════════════════════
    def _read_raw_image(self, file_obj, filename: str) -> DataBundle:
        raw  = file_obj.read()
        b    = DataBundle(source="raw_image")
        # Assume little-endian float32 or uint16
        try:
            arr = np.frombuffer(raw, dtype=np.float32)
            n   = len(arr)
            side= int(np.sqrt(n))
            if side**2 == n:
                arr = arr.reshape(side, side)
            b.metadata  = {"size_bytes": len(raw), "shape": arr.shape}
            b.reservoir = pd.DataFrame({"value": arr.ravel()[:10000]})
            b.images    = [arr]
        except Exception as e:
            b.warnings.append(f"Raw image parse failed: {e}")
        return b

    # ═════════════════════════════════════════════════════════════════════════
    # NEXUS RESERVOIR SIMULATOR  (.nic)
    # ═════════════════════════════════════════════════════════════════════════
    def _read_nexus(self, file_obj, filename: str) -> DataBundle:
        text = file_obj.read().decode("utf-8", "ignore")
        b    = DataBundle(source="nexus")

        # Parse RESERVOIR block
        res_block  = re.search(r'RESERVOIR\s+(.*?)(?=RESERVOIR\s|WELL\s|RECURRENT\s|$)',
                               text, re.DOTALL | re.IGNORECASE)
        well_block = re.search(r'WELL\s+(.*?)(?=RESERVOIR\s|WELL\s|RECURRENT\s|$)',
                               text, re.DOTALL | re.IGNORECASE)

        kv_pattern = r'(\w+)\s*=\s*([^\n!]+)'

        def parse_kv(block_text):
            d = {}
            for m in re.finditer(kv_pattern, block_text):
                key, val = m.group(1).strip(), m.group(2).strip()
                try:
                    d[key] = float(val.split()[0])
                except (ValueError, IndexError):
                    d[key] = val
            return d

        res_kv  = parse_kv(res_block.group(1))  if res_block  else {}
        well_kv = parse_kv(well_block.group(1)) if well_block else {}

        b.reservoir   = pd.DataFrame([res_kv])  if res_kv  else None
        b.completions = pd.DataFrame([well_kv]) if well_kv else None
        b.metadata    = {"reservoir_keys": list(res_kv.keys()),
                         "well_keys": list(well_kv.keys())}
        b.raw = text
        return b
