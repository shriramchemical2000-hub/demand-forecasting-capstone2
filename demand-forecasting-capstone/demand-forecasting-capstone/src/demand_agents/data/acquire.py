"""Stage 1 - acquire the raw data. Nothing here is committed to git; everything lands in ``data/raw/``.

Works in Colab and locally. Kaggle needs credentials (see ``kaggle_ready``); NOAA needs none.
"""
import io
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

M5_COMPETITION = "m5-forecasting-accuracy"
M5_FILES = ("calendar.csv", "sales_train_evaluation.csv", "sell_prices.csv")
WALMART_DATASET = "yasserh/walmart-dataset"
GHCN_URL = "https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/access/{station}.csv"
# One representative airport station per M5 state (M5 does not reveal store locations).
DEFAULT_STATIONS = {"CA": "USW00023174", "TX": "USW00012960", "WI": "USW00014839"}


def kaggle_ready() -> bool:
    """True if Kaggle credentials are available (new token env var / token file, or the legacy kaggle.json)."""
    home = Path.home() / ".kaggle"
    return bool(os.environ.get("KAGGLE_API_TOKEN")) or (home / "kaggle.json").exists() or (home / "access_token").exists() \
        or bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))


def _kaggle(args: list[str]) -> None:
    exe = shutil.which("kaggle")
    if exe is None:
        raise RuntimeError("The kaggle package is not installed: run  pip install kaggle")
    if not kaggle_ready():
        raise RuntimeError(
            "No Kaggle credentials found. On kaggle.com open Settings -> API and either generate a token "
            "(then set the KAGGLE_API_TOKEN environment variable) or create a legacy key (kaggle.json)."
        )
    subprocess.run([exe, *args], check=True)


def download_m5(dest="data/raw/m5", force: bool = False) -> Path:
    """M5 Forecasting - Accuracy. You must have accepted the competition rules on kaggle.com first."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    if not force and all((dest / f).exists() for f in M5_FILES):
        return dest
    _kaggle(["competitions", "download", "-c", M5_COMPETITION, "-p", str(dest)])
    zip_path = dest / f"{M5_COMPETITION}.zip"
    if zip_path.exists():
        shutil.unpack_archive(str(zip_path), str(dest))
    missing = [f for f in M5_FILES if not (dest / f).exists()]
    if missing:
        raise RuntimeError(f"M5 download finished but these files are missing: {missing}")
    return dest


def download_walmart(dest="data/raw/walmart", force: bool = False) -> Path:
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    if not force and (dest / "Walmart.csv").exists():
        return dest
    _kaggle(["datasets", "download", "-d", WALMART_DATASET, "-p", str(dest), "--unzip"])
    if not (dest / "Walmart.csv").exists():
        raise RuntimeError("Walmart download finished but Walmart.csv is missing")
    return dest


def _download_text(url: str, retries: int = 3) -> str:
    last = None
    for attempt in range(1, retries + 1):
        try:
            req = Request(url, headers={"User-Agent": "Mozilla/5.0 (capstone-project data fetch)"})
            return urlopen(req, timeout=180).read().decode()
        except (HTTPError, URLError, TimeoutError) as e:
            last = e
            print(f"  attempt {attempt}/{retries} failed: {e}")
            time.sleep(2 * attempt)
    raise RuntimeError(
        f"Could not download {url} ({last}). Open the address in a browser, save the CSV to data/raw/noaa/<STATION>.csv and rerun."
    )


def download_noaa(dest="data/raw/noaa", stations: dict | None = None, force: bool = False) -> Path:
    """Raw GHCN-Daily station files, one CSV per station, named ``<STATE>_<STATION>.csv``."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for state, station in (stations or DEFAULT_STATIONS).items():
        f = dest / f"{state}_{station}.csv"
        if f.exists() and not force:
            continue
        print(f"Downloading NOAA station {station} for {state} ...")
        f.write_text(_download_text(GHCN_URL.format(station=station)))
    return dest


def parse_ghcn_daily(csv_text: str, state_id: str) -> pd.DataFrame:
    """GHCN-Daily CSV -> DataFrame[date, state_id, temp], temp = mean of TMAX and TMIN in degrees C."""
    raw = pd.read_csv(io.StringIO(csv_text), usecols=lambda c: c in ("DATE", "TMAX", "TMIN"))
    raw["date"] = pd.to_datetime(raw["DATE"])
    tmax, tmin = raw["TMAX"].astype(float), raw["TMIN"].astype(float)
    if max(tmax.abs().median(), tmin.abs().median()) > 60:  # raw GHCN units are tenths of a degree C
        tmax, tmin = tmax / 10.0, tmin / 10.0
    out = pd.DataFrame({"date": raw["date"], "state_id": state_id, "temp": (tmax + tmin) / 2.0})
    return out.dropna(subset=["temp"]).sort_values("date").reset_index(drop=True)


def read_noaa_raw(raw_dir) -> pd.DataFrame:
    """All raw station files in ``raw_dir`` -> one table (date, state_id, temp). The state comes from the file name."""
    frames = []
    for f in sorted(Path(raw_dir).glob("*.csv")):
        state = f.stem.split("_")[0]
        frames.append(parse_ghcn_daily(f.read_text(), state))
    if not frames:
        raise FileNotFoundError(f"No NOAA station files in {raw_dir}")
    return pd.concat(frames, ignore_index=True)


def download(dataset: str, root="data/raw") -> list[Path]:
    root = Path(root)
    out = []
    if dataset in ("m5", "all"):
        out.append(download_m5(root / "m5"))
    if dataset in ("walmart", "all"):
        out.append(download_walmart(root / "walmart"))
    if dataset in ("noaa", "all"):
        out.append(download_noaa(root / "noaa"))
    return out
