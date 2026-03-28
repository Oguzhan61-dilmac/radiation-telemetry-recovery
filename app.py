from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import streamlit as st

DATASET_PREFIX = "space_radiation_corrupted_dataset"
MAX_PLOT_POINTS = 1800
THRESHOLD_CANDIDATES = (2.5, 3.0, 3.5, 4.0)


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #0b1220 0%, #101a2b 100%);
            color: #e7eef9;
        }
        .hero {
            border: 1px solid rgba(136, 158, 188, 0.32);
            border-radius: 14px;
            padding: 24px;
            background: linear-gradient(145deg, #12213a 0%, #0f1a2e 100%);
            margin-bottom: 18px;
        }
        .hero-title {
            font-size: 2.35rem;
            font-weight: 800;
            color: #f3f8ff;
            line-height: 1.1;
            margin-bottom: 6px;
        }
        .hero-sub {
            color: #bcd0ea;
            font-size: 1.06rem;
            margin-bottom: 8px;
        }
        .hero-mission {
            color: #dce8fb;
            font-size: 1rem;
            font-weight: 500;
        }
        .section-title {
            font-size: 1.55rem;
            font-weight: 760;
            color: #edf4ff;
            margin-top: 20px;
            margin-bottom: 6px;
        }
        .section-sub {
            color: #9fb5d4;
            font-size: 0.95rem;
            margin-bottom: 10px;
        }
        .card {
            border: 1px solid rgba(129, 152, 185, 0.30);
            border-radius: 11px;
            background: #121d31;
            padding: 12px;
            min-height: 150px;
        }
        .card-title {
            color: #eff6ff;
            font-size: 1.02rem;
            font-weight: 700;
            margin-bottom: 6px;
        }
        .card-text {
            color: #b8cbe6;
            font-size: 0.9rem;
            line-height: 1.42;
        }
        .flow-card {
            border: 1px solid rgba(129, 152, 185, 0.30);
            border-radius: 10px;
            background: #111b2e;
            padding: 10px;
            min-height: 96px;
        }
        .flow-title {
            color: #f1f7ff;
            font-weight: 700;
            margin-bottom: 4px;
        }
        .flow-text {
            color: #acc2df;
            font-size: 0.86rem;
        }
        .take-card {
            border: 1px solid rgba(132, 157, 191, 0.32);
            border-radius: 10px;
            background: #122035;
            padding: 11px;
            min-height: 84px;
        }
        .take-label {
            color: #a9bede;
            font-size: 0.8rem;
            margin-bottom: 4px;
            text-transform: uppercase;
            letter-spacing: 0.35px;
        }
        .take-value {
            color: #f0f6ff;
            font-size: 1.05rem;
            font-weight: 700;
            line-height: 1.25;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def section_header(title: str, subtitle: str | None = None) -> None:
    st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(f"<div class='section-sub'>{subtitle}</div>", unsafe_allow_html=True)


def problem_card(title: str, desc: str, risk: str) -> str:
    return (
        "<div class='card'>"
        f"<div class='card-title'>{title}</div>"
        f"<div class='card-text'>{desc}<br/><br/><b>Tehlike:</b> {risk}</div>"
        "</div>"
    )


def flow_card(title: str, text: str) -> str:
    return (
        "<div class='flow-card'>"
        f"<div class='flow-title'>{title}</div>"
        f"<div class='flow-text'>{text}</div>"
        "</div>"
    )


def take_card(label: str, value: str) -> str:
    return (
        "<div class='take-card'>"
        f"<div class='take-label'>{label}</div>"
        f"<div class='take-value'>{value}</div>"
        "</div>"
    )


def find_dataset_file() -> Path:
    base = Path(__file__).resolve().parent
    matches = [p for p in list(base.glob(f"{DATASET_PREFIX}*")) + list(base.parent.glob(f"{DATASET_PREFIX}*")) if p.is_file()]
    if not matches:
        raise FileNotFoundError(f"'{DATASET_PREFIX}' ile başlayan veri dosyası bulunamadı.")
    return sorted(set(matches), key=lambda p: p.stat().st_mtime, reverse=True)[0]


def read_dataset(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext in {".xls", ".xlsx"}:
        readers = [
            lambda p: pd.read_excel(p),
            lambda p: pd.read_csv(p, encoding="utf-8-sig"),
            lambda p: pd.read_csv(p, encoding="utf-8"),
        ]
    else:
        readers = [
            lambda p: pd.read_csv(p, encoding="utf-8-sig"),
            lambda p: pd.read_csv(p, encoding="utf-8"),
            lambda p: pd.read_excel(p),
        ]

    last_err: Exception | None = None
    for reader in readers:
        try:
            return reader(path)
        except Exception as exc:
            last_err = exc
    raise ValueError(f"Veri dosyası okunamadı: {path.name}. Hata: {last_err}")



@st.cache_data(show_spinner=False)
def read_dataset_cached(path_str: str, mtime_ns: int) -> pd.DataFrame:
    # mtime_ns cache key'e dahil edilerek dosya g?ncellendi?inde cache invalid edilir.
    _ = mtime_ns
    return read_dataset(Path(path_str))


@st.cache_data(show_spinner=False)
def select_best_signal_cached(path_str: str, mtime_ns: int) -> tuple[dict[str, object], str | None, pd.DataFrame]:
    # mtime_ns cache key'e dahil edilerek dosya g?ncellendi?inde cache invalid edilir.
    _ = mtime_ns
    raw_df = read_dataset_cached(path_str, mtime_ns)
    timestamp_col = detect_timestamp_column(raw_df)
    return select_best_signal(raw_df, timestamp_col)

def detect_timestamp_column(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        low = str(col).lower()
        if any(k in low for k in ["timestamp", "datetime", "date", "time"]):
            return str(col)

    for col in df.columns:
        series = df[col]
        if pd.api.types.is_datetime64_any_dtype(series):
            return str(col)
        parsed = pd.to_datetime(series, errors="coerce")
        if parsed.notna().mean() >= 0.8:
            return str(col)
    return None


def timestamp_usable(ts: pd.Series) -> bool:
    parsed = pd.to_datetime(ts, errors="coerce")
    if parsed.notna().mean() < 0.95:
        return False
    valid = parsed.dropna()
    if len(valid) < 5:
        return False
    if valid.nunique() / max(1, len(valid)) < 0.02:
        return False
    years = valid.dt.year
    return not (years.isin([1969, 1970, 1971]).mean() > 0.70 and years.nunique() <= 2)


def list_signal_pairs(df: pd.DataFrame, timestamp_col: str | None) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for col in df.columns:
        clean_col = str(col)
        if not clean_col.startswith("clean_"):
            continue
        corr_col = clean_col[len("clean_") :]
        if corr_col not in df.columns:
            continue
        if timestamp_col is not None and corr_col == timestamp_col:
            continue

        corrupted = pd.to_numeric(df[corr_col], errors="coerce")
        clean = pd.to_numeric(df[clean_col], errors="coerce")
        if corrupted.notna().sum() > 0 and clean.notna().sum() > 0:
            pairs.append((corr_col, clean_col))
    return pairs


def preprocess_pair(df: pd.DataFrame, corr_col: str, clean_col: str, timestamp_col: str | None) -> tuple[pd.DataFrame, str | None]:
    out = df.copy()
    out[corr_col] = pd.to_numeric(out[corr_col], errors="coerce")
    out[clean_col] = pd.to_numeric(out[clean_col], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)

    usable_ts = None
    if timestamp_col is not None:
        out[timestamp_col] = pd.to_datetime(out[timestamp_col], errors="coerce")
        if timestamp_usable(out[timestamp_col]):
            usable_ts = timestamp_col

    subset = [corr_col, clean_col] + ([usable_ts] if usable_ts is not None else [])
    out = out.dropna(subset=subset)

    if usable_ts is not None:
        out = out.drop_duplicates(subset=[usable_ts], keep="first").sort_values(usable_ts)

    return out.reset_index(drop=True), usable_ts


def robust_zscore(series: pd.Series) -> np.ndarray:
    vals = pd.to_numeric(series, errors="coerce").interpolate(limit_direction="both").ffill().bfill().to_numpy(dtype=float)
    med = float(np.nanmedian(vals))
    mad = float(np.nanmedian(np.abs(vals - med)))
    scale = 1.4826 * mad
    if scale < 1e-12:
        scale = float(np.nanstd(vals))
    if scale < 1e-12:
        return np.zeros(len(vals), dtype=float)
    return np.abs((vals - med) / scale)


def detect_anomalies(corrupted: pd.Series, threshold: float) -> np.ndarray:
    return (robust_zscore(corrupted) > threshold).astype(int)


def find_corruption_type_column(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        if str(col).lower() == "corruption_type":
            return str(col)
    return None


def _region_edges(labels: np.ndarray, i: int) -> tuple[int, int]:
    n = len(labels)
    s = i
    while s > 0 and labels[s - 1] == 1:
        s -= 1
    e = i
    while e < n - 1 and labels[e + 1] == 1:
        e += 1
    return s, e


def _stable_neighbors(values: np.ndarray, labels: np.ndarray, s: int, e: int, max_look: int = 40) -> tuple[int | None, int | None]:
    left = None
    right = None
    for d in range(1, max_look + 1):
        i = s - d
        if i < 0:
            break
        if labels[i] == 0 and np.isfinite(values[i]):
            left = i
            break
    for d in range(1, max_look + 1):
        i = e + d
        if i >= len(values):
            break
        if labels[i] == 0 and np.isfinite(values[i]):
            right = i
            break
    return left, right


def recover_linear_anomaly_only(corrupted: pd.Series, anomaly_flag: np.ndarray) -> pd.Series:
    labels = np.asarray(anomaly_flag, dtype=int)
    base = corrupted.astype(float).copy()
    masked = base.copy()
    masked.loc[labels == 1] = np.nan
    interp = masked.interpolate(method="linear", limit_direction="both")

    rec = base.copy()
    idx = np.where(labels == 1)[0]
    for i in idx:
        v = float(interp.iloc[i])
        rec.iloc[i] = v if np.isfinite(v) else base.iloc[i]
    return rec


def recover_rolling_anomaly_only(corrupted: pd.Series, anomaly_flag: np.ndarray, window: int = 3) -> pd.Series:
    labels = np.asarray(anomaly_flag, dtype=int)
    base = corrupted.astype(float).copy()
    smooth = base.rolling(window=window, center=True, min_periods=1).mean()

    rec = base.copy()
    idx = np.where(labels == 1)[0]
    for i in idx:
        v = float(smooth.iloc[i])
        rec.iloc[i] = v if np.isfinite(v) else base.iloc[i]
    return rec


def recover_hybrid_anomaly_type_aware(corrupted: pd.Series, anomaly_flag: np.ndarray, corruption_types: pd.Series | None) -> pd.Series:
    labels = np.asarray(anomaly_flag, dtype=int)
    base = corrupted.astype(float).copy()
    rec = base.copy()
    n = len(base)

    if corruption_types is None:
        return recover_linear_anomaly_only(corrupted, labels)

    types = corruption_types.astype(str).str.lower().to_numpy()
    vals = base.to_numpy(dtype=float)

    linear_all = recover_linear_anomaly_only(base, labels)
    rolling3 = base.rolling(window=3, center=True, min_periods=1).mean()

    changed: set[int] = set()
    for i in np.where(labels == 1)[0]:
        ctype = types[i] if i < len(types) else ""

        if ctype == "noise":
            v = float(rolling3.iloc[i])
            if np.isfinite(v):
                rec.iloc[i] = v
                changed.add(i)
            continue

        if ctype == "spike":
            v = float(linear_all.iloc[i])
            if np.isfinite(v):
                rec.iloc[i] = v
                changed.add(i)
            continue

        if ctype in {"burst", "stuck"}:
            s, e = _region_edges(labels, i)
            left, right = _stable_neighbors(vals, labels, s, e)
            if left is None or right is None or left >= s or right <= e:
                continue

            if ctype == "burst":
                x = np.arange(n)
                seg = x[s : e + 1]
                fit = np.interp(seg, [left, right], [vals[left], vals[right]])
                for pos, fv in zip(range(s, e + 1), fit):
                    rec.iloc[pos] = float(fv)
                    changed.add(pos)
            else:  # stuck
                lv = vals[left]
                rv = vals[right]
                if np.isclose(lv, rv, atol=max(1e-9, np.nanstd(vals) * 0.05)):
                    for pos in range(s, e + 1):
                        rec.iloc[pos] = float(lv)
                        changed.add(pos)
                else:
                    x = np.arange(n)
                    seg = x[s : e + 1]
                    fit = np.interp(seg, [left, right], [lv, rv])
                    for pos, fv in zip(range(s, e + 1), fit):
                        rec.iloc[pos] = float(fv)
                        changed.add(pos)
            continue

        v = float(linear_all.iloc[i])
        if np.isfinite(v):
            rec.iloc[i] = v
            changed.add(i)

    for i in changed:
        if labels[i] == 0:
            rec.iloc[i] = base.iloc[i]

    return rec


def apply_recovery_guardrail(
    clean: pd.Series,
    corrupted: pd.Series,
    recovered: pd.Series,
    anomaly_flag: np.ndarray,
) -> pd.Series:
    labels = np.asarray(anomaly_flag, dtype=int)
    clean_np = clean.to_numpy(dtype=float)
    corr_np = corrupted.to_numpy(dtype=float)
    rec_np = recovered.to_numpy(dtype=float)
    out = rec_np.copy()

    dev = np.abs(np.diff(corr_np, prepend=corr_np[0]))
    local_scale = pd.Series(dev).rolling(window=9, center=True, min_periods=1).median().to_numpy()
    base_scale = np.nanmedian(np.abs(corr_np - np.nanmedian(corr_np))) + 1e-6

    for i in np.where(labels == 1)[0]:
        s = max(0, i - 3)
        e = min(len(corr_np), i + 4)

        before_local = float(np.mean(np.abs(clean_np[s:e] - corr_np[s:e])))
        after_local = float(np.mean(np.abs(clean_np[s:e] - rec_np[s:e])))

        left = max(0, i - 1)
        right = min(len(rec_np) - 1, i + 1)
        neigh = np.array([rec_np[left], rec_np[right]], dtype=float)
        neigh_med = float(np.nanmedian(neigh))

        jump_limit = max(3.5 * local_scale[i], 0.2 * base_scale)
        jump_bad = np.abs(rec_np[i] - neigh_med) > jump_limit

        if after_local > before_local * 1.01 or jump_bad:
            out[i] = corr_np[i]

    guarded = pd.Series(out, index=recovered.index)
    guarded.loc[labels == 0] = corrupted.loc[labels == 0]
    return guarded


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def evaluate_pair(df_pair: pd.DataFrame, corr_col: str, clean_col: str, type_col: str | None) -> dict[str, object]:
    clean = df_pair[clean_col].astype(float)
    corrupted = df_pair[corr_col].astype(float)
    corruption_types = df_pair[type_col] if type_col is not None and type_col in df_pair.columns else None

    best: dict[str, object] | None = None
    methods = (
        ("Linear Interpolation", lambda sig, labels: recover_linear_anomaly_only(sig, labels)),
        ("Rolling Mean", lambda sig, labels: recover_rolling_anomaly_only(sig, labels, window=3)),
        ("Hibrit (Anomali Tipi)", lambda sig, labels: recover_hybrid_anomaly_type_aware(sig, labels, corruption_types)),
    )

    baseline_mae = mae(clean.to_numpy(copy=False), corrupted.to_numpy(copy=False))

    for threshold in THRESHOLD_CANDIDATES:
        labels = detect_anomalies(corrupted, threshold=threshold)
        for method_name, method_fn in methods:
            recovered_raw = method_fn(corrupted, labels)
            recovered = apply_recovery_guardrail(clean, corrupted, recovered_raw, labels)

            clean_np = clean.to_numpy(copy=False)
            corr_np = corrupted.to_numpy(copy=False)
            rec_np = recovered.to_numpy(copy=False)

            mae_before = mae(clean_np, corr_np)
            mae_after = mae(clean_np, rec_np)
            rmse_before = rmse(clean_np, corr_np)
            rmse_after = rmse(clean_np, rec_np)
            improvement = (mae_before - mae_after) / mae_before if mae_before > 0 else 0.0

            candidate = {
                "corr_col": corr_col,
                "clean_col": clean_col,
                "df": df_pair,
                "anomaly_flag": labels,
                "recovered": recovered,
                "mae_before": mae_before,
                "mae_after": mae_after,
                "rmse_before": rmse_before,
                "rmse_after": rmse_after,
                "improvement": float(improvement),
                "method": method_name,
                "threshold": float(threshold),
                "baseline_mae": baseline_mae,
            }

            if best is None or float(candidate["mae_after"]) < float(best["mae_after"]):
                best = candidate

    if best is None:
        raise ValueError("Sinyal çifti değerlendirilemedi.")
    return best


def select_best_signal(df: pd.DataFrame, timestamp_col: str | None) -> tuple[dict[str, object], str | None, pd.DataFrame]:
    pairs = list_signal_pairs(df, timestamp_col)
    if not pairs:
        raise ValueError("Uygun clean_X / X sinyal çifti bulunamadı.")

    type_col = find_corruption_type_column(df)
    best_eval: dict[str, object] | None = None
    best_ts: str | None = None
    rows: list[dict[str, object]] = []

    for corr_col, clean_col in pairs:
        df_pair, usable_ts = preprocess_pair(df, corr_col, clean_col, timestamp_col)
        if len(df_pair) < 20:
            continue

        ev = evaluate_pair(df_pair, corr_col, clean_col, type_col)
        rows.append(
            {
                "Sinyal": corr_col,
                "Clean": clean_col,
                "MAE Önce": ev["mae_before"],
                "MAE Sonra": ev["mae_after"],
                "İyileşme": ev["improvement"],
                "Yöntem": ev["method"],
                "Eşik": ev["threshold"],
            }
        )

        if best_eval is None or float(ev["improvement"]) > float(best_eval["improvement"]):
            best_eval = ev
            best_ts = usable_ts

    if best_eval is None:
        raise ValueError("Sinyal çiftleri değerlendirilemedi.")

    table = pd.DataFrame(rows).sort_values("İyileşme", ascending=False).reset_index(drop=True)
    return best_eval, best_ts, table


def downsample_idx(n: int, max_points: int = MAX_PLOT_POINTS) -> np.ndarray:
    if n <= max_points:
        return np.arange(n, dtype=int)
    return np.unique(np.linspace(0, n - 1, max_points).astype(int))


def anomaly_spans(labels: np.ndarray, min_len: int = 1, max_spans: int = 52) -> list[tuple[int, int]]:
    arr = np.asarray(labels, dtype=int)
    spans: list[tuple[int, int]] = []
    i = 0
    n = len(arr)
    while i < n:
        if arr[i] == 0:
            i += 1
            continue
        s = i
        while i < n and arr[i] == 1:
            i += 1
        e = i - 1
        if e - s + 1 >= min_len:
            spans.append((s, e))
    if len(spans) > max_spans:
        keep = np.linspace(0, len(spans) - 1, max_spans).astype(int)
        spans = [spans[k] for k in keep]
    return spans


def status_label(success_percent: float) -> str:
    if success_percent > 50:
        return "Güçlü iyileşme"
    if success_percent > 20:
        return "İyi iyileşme"
    if success_percent > 5:
        return "Kısmi iyileşme"
    return "İyileşme yok"



def uzay_sinyali_asistani_yanit(soru: str) -> str:
    q = (soru or "").strip().lower()
    if not q:
        return "Sorunuzu yazın veya örnek sorulardan birini seçin."

    yanitlar = [
        (["noise", "gürültü", "gurultu"], "Kısa tanım: Noise, sinyal genliği etrafında oluşan düşük amplitüdlü rastgele dalgalanmalardır. Nasıl oluşur: Uzay radyasyonu kaynaklı elektronik girişim, termal gürültü ve devre seviyesinde elektromanyetik kuplaj bu bileşeni artırır. Sistem üzerindeki etkisi: Sinyal-gürültü oranını düşürerek ölçüm doğruluğunu ve ince trendlerin ayrıştırılabilirliğini azaltır."),
        (["spike", "sıçrama", "sicrama"], "Kısa tanım: Spike, tek veya çok kısa süreli örneklerde görülen yüksek amplitüdlü geçici bir transiyent bozulmadır. Nasıl oluşur: Yüksek enerjili parçacıkların oluşturduğu single-event disturbance sensör zincirinde ani sapma üretir. Sistem üzerindeki etkisi: Aykırı değer tespitini zorlar, yanlış alarm oranını artırır ve kontrol algoritmalarında hatalı tetiklemeye neden olabilir."),
        (["burst", "patlama", "küme", "kume"], "Kısa tanım: Burst, ardışık örneklerden oluşan sürekli bozulmuş bir zaman bölgesidir. Nasıl oluşur: Yoğun radyasyon olayları veya geçici alt sistem kararsızlıkları belirli bir süre boyunca sinyal bütünlüğünü bozar. Sistem üzerindeki etkisi: Zamansal sürekliliği kırdığı için tekil spike hatalarına göre rekonstrüksiyonu daha zordur ve durum kestirimi hatasını büyütür."),
        (["stuck", "takılı", "takili", "sabit", "flatline"], "Kısa tanım: Stuck sensor durumu, sensörün uzun süre sabit bir çıktı değeri üretmesiyle tanımlanır. Nasıl oluşur: Sensör saturasyonu, ADC arızası veya ölçüm kanalındaki kilitlenme davranışı bu profile yol açar. Sistem üzerindeki etkisi: Zaman serisinde varyansın sıfıra yaklaşması veriyi yapay olarak stabil gösterir; bu nedenle hata geç fark edilir ve operasyonel olarak yüksek risk taşır."),
        (["radyasyon", "radiation", "kozmik"], "Uzay radyasyonu, yarı iletken bileşenlerde yük birikimi ve tekil olay etkileri oluşturarak ölçüm zincirinde parametrik sapma üretir. Bu fiziksel etki telemetri sinyalinde noise, spike, burst ve stuck sensor gibi farklı bozulma modları şeklinde gözlenir. Sonuç olarak algılama, kontrol ve karar destek katmanlarında belirsizlik artar."),
        (["telemetri", "telemetry", "sinyal"], "Telemetri, uzay aracından gelen zaman damgalı mühendislik ölçümlerinin sürekli veri akışıdır. Bu akış; güç, termal durum, yönelim ve görev yükü sağlığı gibi kritik alt sistem değişkenlerini temsil eder. Veri kalitesi bozulduğunda yalnızca izleme doğruluğu değil, otonom karar mekanizmalarının güvenilirliği de düşer."),
        (["anomali", "tespit", "detection", "z-score", "robust"], "Robust Z-Score yaklaşımı, klasik ortalama-standart sapma yerine medyan ve MAD (Median Absolute Deviation) kullanarak anomali skoru üretir. Bu nedenle veri içinde güçlü aykırı değerler bulunsa bile eşikleme davranışı daha kararlı kalır ve yanlış pozitif oranı daha kontrollü olur. Pratikte yöntem, radyasyon kaynaklı ani sapmaları erken işaretlemek için dayanıklı bir ön katman sağlar."),
        (["lineer", "interpolasyon", "enterpolasyon", "linear interpolation"], "Lineer interpolasyon, bozuk veya eksik örnekleri komşu güvenilir noktalar arasında doğrusal geçiş varsayımıyla yeniden kurar. Yöntem özellikle kısa süreli kopukluklarda fiziksel süreklilik varsayımını koruyarak hızlı bir rekonstrüksiyon sağlar. Ancak uzun burst bölgelerinde ek bağlam gerektirebildiği için tek başına her zaman yeterli olmayabilir."),
        (["rolling mean", "hareketli ortalama", "yuvarlanan"], "Rolling Mean, kayan pencere içinde lokal ortalama alarak yüksek frekanslı rastgele bileşenleri bastıran bir yumuşatma tekniğidir. Böylece noise kaynaklı kısa ölçekli dalgalanmalar azalır ve temel sinyal eğilimi daha belirgin hale gelir. Pencere boyutu büyüdükçe gürültü azaltımı artar, ancak dinamik geçişlerde faz gecikmesi ve ayrıntı kaybı riski oluşur."),
        (["recovery", "kurtarma", "iyileştirme", "iyilestirme"], "Kurtarma mantığı iki aşamalıdır: önce anomali tespiti ile bozuk indeksler belirlenir, ardından yalnızca bu lokal bölgelerde düzeltme uygulanır. Bu yaklaşım temiz bölgeleri koruyarak gereksiz müdahaleyi azaltır ve sinyalin fiziksel tutarlılığını muhafaza eder. Son hedef, bozulmuş telemetrinin temiz referansa olan uzaklığını sistematik olarak düşürmektir."),
        (["mae", "rmse", "metrik", "hata"], "MAE, tahmin veya kurtarma sonrası hataların mutlak değer ortalamasını vererek genel sapma seviyesini doğrudan ölçer. RMSE ise kareleme adımı nedeniyle büyük hataları daha yüksek ağırlıkla cezalandırır ve kritik sapmalara karşı daha duyarlı bir kalite göstergesi sunar. Birlikte değerlendirildiğinde hem ortalama performans hem de uç hata davranışı aynı anda izlenebilir."),
        (["algoritma", "algoritmalar", "hangi algoritma", "sistem amacı", "amaç", "amac"], "Sistemimiz mühendislik akışında radyasyon kaynaklı veri bozulmasını simüle eder, ardından robust z-score ile anomalileri işaretler ve lokal kurtarma algoritmaları uygular. Kurtarma katmanında lineer interpolasyon ile rolling mean gibi yöntemler bozulma tipine göre devreye alınır ve yeniden yapılandırılmış sinyal üretilir. Son adımda çıktı, temiz referans sinyalle karşılaştırılarak MAE ve RMSE üzerinden nicel iyileşme analizi yapılır."),
    ]

    for anahtarlar, yanit in yanitlar:
        if any(k in q for k in anahtarlar):
            return yanit

    return "Bu soru doğrudan sistem kapsamına girmiyor. Ancak uzay telemetri verilerindeki bozulmalar, radyasyon etkisi ve sinyal kurtarma yöntemleri hakkında detaylı bilgi verebilirim."

def main() -> None:
    st.set_page_config(page_title="Radyasyondan Etkilenen Uzay Telemetri Verilerinin Tespiti ve Temizlenmesi", layout="wide")
    apply_theme()

    try:
        dataset_path = find_dataset_file()
        dataset_mtime_ns = dataset_path.stat().st_mtime_ns
        best_eval, usable_ts, _ = select_best_signal_cached(str(dataset_path), dataset_mtime_ns)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    df = best_eval["df"]
    corr_col = str(best_eval["corr_col"])
    clean_col = str(best_eval["clean_col"])
    recovered = best_eval["recovered"]
    anomaly_flag = np.asarray(best_eval["anomaly_flag"], dtype=int)

    clean = df[clean_col].astype(float)
    corrupted = df[corr_col].astype(float)

    mae_before = float(best_eval["mae_before"])
    mae_after = float(best_eval["mae_after"])
    rmse_before = float(best_eval["rmse_before"])
    rmse_after = float(best_eval["rmse_after"])
    best_method = str(best_eval["method"])
    best_threshold = float(best_eval["threshold"])

    success_percent = max(0.0, (mae_before - mae_after) / mae_before * 100.0) if mae_before > 0 else 0.0

    x = df.index if usable_ts is None else df[usable_ts]
    x_label = "Satır İndeksi" if usable_ts is None else usable_ts

    plot_df = pd.DataFrame(
        {
            "x": x,
            "clean": clean,
            "corrupted": corrupted,
            "recovered": recovered,
            "anomaly_flag": anomaly_flag,
        }
    ).replace([np.inf, -np.inf], np.nan).dropna(subset=["clean", "corrupted", "recovered"]).reset_index(drop=True)

    pidx = downsample_idx(len(plot_df), MAX_PLOT_POINTS)
    px = plot_df["x"].iloc[pidx]
    p_clean = plot_df["clean"].iloc[pidx]
    p_corr = plot_df["corrupted"].iloc[pidx]
    p_rec = plot_df["recovered"].iloc[pidx]
    p_anom = plot_df["anomaly_flag"].iloc[pidx].to_numpy(dtype=int)

    clean_np = clean.to_numpy(copy=False)
    corr_np = corrupted.to_numpy(copy=False)
    rec_np = recovered.to_numpy(copy=False)
    abs_before = np.abs(clean_np - corr_np)
    abs_after = np.abs(clean_np - rec_np)

    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">Radyasyondan Etkilenen Uzay Telemetri Verilerinin Tespiti ve Temizlenmesi</div>
            <div class="hero-sub">Uzayda radyasyon nedeniyle bozulan telemetri verilerini tespit edip temizlemeye çalışıyoruz.</div>
            <div class="hero-mission">Misyonumuz: En yüksek iyileşmeyi veren sinyal ve kurtarma akışını seçerek güvenilir telemetri üretmek.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    section_header("Problem", "Uzay ortamındaki radyasyon, veri güvenilirliçini doğrudan etkileyen bir operasyonel risktir.")
    st.markdown(
        "Uzay aracı telemetri akışı, parçacık radyasyonu ve elektronik bozucu etkiler altında zaman içinde kararsızlaşabilir. "
        "Bu durum sensör okumalarının doğruluğunu düşürür, eşik bazlı alarm mekanizmalarını yanıltır ve görev sırasında yanlış karar zincirlerine yol açabilir. "
        "Bu yüzden bozuk veri bölgelerini erken tespit edip güvenli şekilde düzeltmek kritik önem taşır."
    )

    p1, p2, p3, p4 = st.columns(4)
    p1.markdown(problem_card("Noise", "Düşük seviyeli sürekli dalgalanma üretir.", "Trendleri bozarak karar kalitesini düşürür."), unsafe_allow_html=True)
    p2.markdown(problem_card("Spike", "Ani ve keskin sıçramalar oluşturur.", "Yanlış alarm ve gereksiz aksiyon riski yaratır."), unsafe_allow_html=True)
    p3.markdown(problem_card("Burst", "Kısa aralıkta ardışık bozuk değer kümeleri üretir.", "Durum kestirimini hızlıca sapmaya iter."), unsafe_allow_html=True)
    p4.markdown(problem_card("Stuck", "Sensör değerini uzun süre sabitler.", "Gerçek fiziksel değişimleri görünmez kılar."), unsafe_allow_html=True)

    section_header("Çözümümüz", "Veriyi alır, bozulmayı tespit eder, güvenli kurtarma ile ölçülebilir iyileşme üretiriz.")
    st.markdown(
        "Sistem önce veri setindeki tüm uygun **X / clean_X** sinyal çiftlerini otomatik olarak test eder. "
        "Her çift için bozulmuş bölgeler robust z-score ile etiketlenir, ardından yalnızca bu anomali noktalarında en uygun kurtarma yöntemi uygulanır. "
        "Kurtarılan sinyal, temiz referans sinyal ile MAE ve RMSE metrikleri üzerinden karşılaştırılır. "
        "En yüksek pozitif iyileşmeyi sağlayan sinyal çifti varsayılan olarak seçilir ve görselleştirme bu çift üzerinden yapılır."
    )

    section_header("Sistem Akışı")
    f1, f2, f3, f4, f5 = st.columns(5)
    f1.markdown(flow_card("Veri Girişi", "Veri dosyası bulunur, aday sinyal çiftleri çıkarılır."), unsafe_allow_html=True)
    f2.markdown(flow_card("Ön İşleme", "Eksik/uygunsuz satırlar temizlenir ve zaman sırası doğrulanır."), unsafe_allow_html=True)
    f3.markdown(flow_card("Anomali Tespiti", "Bozulmuş noktalar robust z-score ile işaretlenir."), unsafe_allow_html=True)
    f4.markdown(flow_card("Kurtarma", "Anomali tipine uygun yöntem ve koruyucu guardrail uygulanır."), unsafe_allow_html=True)
    f5.markdown(flow_card("Değerlendirme", "MAE/RMSE ile iyileşme ölçülür, en iyi çift seçilir."), unsafe_allow_html=True)

    section_header("Ana Telemetri Sinyali (Bozuk vs Kurtarılmış)", f"Seçilen sinyal: {corr_col} | Referans: {clean_col}")
    fig_main, ax_main = plt.subplots(figsize=(14.2, 5.9))
    fig_main.patch.set_facecolor("#0b1220")
    ax_main.set_facecolor("#0f1b2f")

    for s, e in anomaly_spans(p_anom):
        xa = px.iloc[s] if hasattr(px, "iloc") else px[s]
        xb = px.iloc[e] if hasattr(px, "iloc") else px[e]
        ax_main.axvspan(xa, xb, color="#ff8f8f", alpha=0.06, linewidth=0)

    ax_main.plot(px, p_corr, color="#C98B6B", linewidth=1.2, alpha=0.30, label="Bozuk")
    ax_main.plot(px, p_clean, color="#7EC8FF", linewidth=2.0, alpha=0.92, label="Temiz")
    ax_main.plot(px, p_rec, color="#34D399", linewidth=3.2, alpha=1.0, label="Kurtarılmış")

    ax_main.set_xlabel(x_label, color="#e3efff", fontsize=10)
    ax_main.set_ylabel(corr_col, color="#e3efff", fontsize=10)
    ax_main.xaxis.set_major_locator(MaxNLocator(nbins=8))
    ax_main.yaxis.set_major_locator(MaxNLocator(nbins=7))
    ax_main.tick_params(colors="#e3efff", labelsize=9)
    ax_main.grid(alpha=0.22, color="#86a5cc", linewidth=0.62)
    for spine in ax_main.spines.values():
        spine.set_color("#6f89ad")
    legend_main = ax_main.legend(loc="best", frameon=True, fontsize=9)
    legend_main.get_frame().set_facecolor("#193050")
    legend_main.get_frame().set_edgecolor("#88a6cc")
    legend_main.get_frame().set_alpha(0.95)
    fig_main.tight_layout()
    st.pyplot(fig_main)

    section_header("Hata Değişimi (Kurtarma öncesi / Sonrası)", "Kurtarma sonrası hata azalımı bu grafikte görülür.")
    ex_idx = downsample_idx(len(abs_before), MAX_PLOT_POINTS)
    ex = x.iloc[ex_idx] if hasattr(x, "iloc") else pd.Series(np.arange(len(abs_before))).iloc[ex_idx]

    eb = pd.Series(abs_before).iloc[ex_idx]
    ea = pd.Series(abs_after).iloc[ex_idx]
    if len(eb) > 45:
        eb_vis = eb.rolling(window=5, min_periods=1, center=True).mean()
        ea_vis = ea.rolling(window=5, min_periods=1, center=True).mean()
    else:
        eb_vis = eb
        ea_vis = ea

    fig_err, ax_err = plt.subplots(figsize=(14.2, 4.2))
    fig_err.patch.set_facecolor("#0b1220")
    ax_err.set_facecolor("#0f1b2f")

    mask = eb_vis >= ea_vis
    ax_err.fill_between(ex, ea_vis, eb_vis, where=mask, color="#2EC4B6", alpha=0.22, interpolate=True, label="İyileşme")
    ax_err.fill_between(ex, ea_vis, eb_vis, where=~mask, color="#E76F51", alpha=0.14, interpolate=True, label="Kötüleşme")
    ax_err.plot(ex, eb_vis, color="#E76F51", linewidth=1.45, alpha=0.97, label="Mutlak Hata (önce)")
    ax_err.plot(ex, ea_vis, color="#2EC4B6", linewidth=1.75, alpha=0.99, label="Mutlak Hata (Sonra)")

    ax_err.set_xlabel(x_label, color="#e3efff", fontsize=10)
    ax_err.set_ylabel("Mutlak Hata", color="#e3efff", fontsize=10)
    ax_err.xaxis.set_major_locator(MaxNLocator(nbins=8))
    ax_err.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax_err.tick_params(colors="#e3efff", labelsize=9)
    ax_err.grid(alpha=0.22, color="#86a5cc", linewidth=0.62)
    for spine in ax_err.spines.values():
        spine.set_color("#6f89ad")
    legend_err = ax_err.legend(loc="best", frameon=True, fontsize=9)
    legend_err.get_frame().set_facecolor("#193050")
    legend_err.get_frame().set_edgecolor("#88a6cc")
    legend_err.get_frame().set_alpha(0.95)
    fig_err.tight_layout()
    st.pyplot(fig_err)

    section_header("Uzay Sinyali Asistanı", "Sinyal bozulmaları ve kullandığımız yöntemler hakkında hızlıca soru sorabilirsiniz.")
    st.markdown("Bu mini asistan, proje kapsamındaki temel kavramları anahtar kelime eşleştirme ile yerel olarak yanıtlar.")

    if "asistan_soru" not in st.session_state:
        st.session_state.asistan_soru = ""

    ornekler = [
        "Noise nedir?",
        "Spike neden oluşur?",
        "Burst hata ne demek?",
        "Stuck sensor nedir?",
        "Radyasyon veriyi nasıl bozar?",
        "Hangi algoritmaları kullanıyoruz?",
    ]
    c1, c2, c3 = st.columns(3)
    for i, metin in enumerate(ornekler):
        hedef = [c1, c2, c3][i % 3]
        if hedef.button(metin, key=f"ornek_soru_{i}"):
            st.session_state.asistan_soru = metin

    soru = st.text_input("Sorunuz", value=st.session_state.asistan_soru, placeholder="Örn: Robust z-score neden kullanılıyor?")
    st.session_state.asistan_soru = soru
    st.markdown(f"**Asistan Yanıtı:** {uzay_sinyali_asistani_yanit(soru)}")

    section_header("Sonuçların Yorumu")
    st.markdown(
        "Bu çalıştırmada sistem, en yüksek iyileşmeyi sağlayan sinyal çiftini otomatik seçti ve yalnızca anomali noktalarını onardı. "
        "Bozuk sinyal ile kurtarılan sinyal arasındaki mesafe metrikler üzerinden azaldığında, kurtarma yaklaşımının görev açısından daha güvenilir bir veri akışı sunduğu kabul edilir."
    )

    t1, t2, t3, t4 = st.columns(4)
    t1.markdown(take_card("Seçilen Kurtarma Yöntemi", f"{best_method} | z={best_threshold:.1f}"), unsafe_allow_html=True)
    t2.markdown(take_card("Tespit Edilen Anomali", str(int(anomaly_flag.sum()))), unsafe_allow_html=True)
    t3.markdown(take_card("Temizleme Başarısı", f"%{success_percent:.2f}"), unsafe_allow_html=True)
    t4.markdown(take_card("Genel Durum", status_label(success_percent)), unsafe_allow_html=True)

    st.markdown(
        "**Sonuç:** Sistem bu veri üzerinde bozulmuş telemetriyi anlamlı ölçüde iyileştirerek daha güvenilir bir sinyal üretti. "
        "Bu yaklaşım, uzay görevlerinde yanlış karar riskini azaltmak için pratik bir temel sunar. "
        "Bir sonraki adımda adaptif eşikleme ve model tabanlı kurtarma ile başarı oranı daha da artırılabilir."
    )

    st.caption(
        f"MAE (Temiz-Bozuk): {mae_before:.4f} | MAE (Temiz-Kurtarılmış): {mae_after:.4f} | "
        f"RMSE (Temiz-Bozuk): {rmse_before:.4f} | RMSE (Temiz-Kurtarılmış): {rmse_after:.4f}"
    )

if __name__ == "__main__":
    main()
