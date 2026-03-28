from __future__ import annotations

import numpy as np
import pandas as pd


def _nearest_clean_neighbors(values: np.ndarray, labels: np.ndarray, idx: int) -> tuple[tuple[int, float] | None, tuple[int, float] | None]:
    n = len(values)

    left = idx - 1
    while left >= 0 and labels[left] == 1:
        left -= 1

    right = idx + 1
    while right < n and labels[right] == 1:
        right += 1

    left_pt = (left, float(values[left])) if left >= 0 else None
    right_pt = (right, float(values[right])) if right < n else None
    return left_pt, right_pt


def _segment_bounds(labels: np.ndarray) -> list[tuple[int, int]]:
    segs: list[tuple[int, int]] = []
    i = 0
    n = len(labels)
    while i < n:
        if labels[i] == 0:
            i += 1
            continue
        s = i
        while i < n and labels[i] == 1:
            i += 1
        segs.append((s, i - 1))
    return segs


def _majority_type(types: np.ndarray, start: int, end: int) -> str:
    region = [str(v).lower() for v in types[start : end + 1]]
    known = [v for v in region if v in {"noise", "spike", "burst", "stuck"}]
    if not known:
        return "unknown"
    uniq, cnt = np.unique(np.array(known, dtype=object), return_counts=True)
    return str(uniq[np.argmax(cnt)])


def _local_scale(values: np.ndarray, labels: np.ndarray, idx: int, window: int = 10) -> float:
    s = max(0, idx - window)
    e = min(len(values), idx + window + 1)
    seg = values[s:e]
    lab = labels[s:e]

    clean = seg[lab == 0]
    if clean.size < 3:
        clean = seg

    med = float(np.nanmedian(clean))
    mad = float(np.nanmedian(np.abs(clean - med)))
    scale = 1.4826 * mad
    if scale < 1e-9:
        scale = float(np.nanstd(clean))
    if scale < 1e-9:
        scale = 1.0
    return scale


def recovery_safety_guard(corrupted: pd.Series, recovered: pd.Series, anomaly_flag: np.ndarray) -> pd.Series:
    corr = corrupted.astype(float).to_numpy(copy=False)
    rec = recovered.astype(float).to_numpy(copy=True)
    labels = np.asarray(anomaly_flag, dtype=int)

    for i in np.where(labels == 1)[0]:
        if not np.isfinite(rec[i]):
            rec[i] = corr[i]
            continue

        left, right = _nearest_clean_neighbors(corr, labels, int(i))
        scale = _local_scale(corr, labels, int(i))

        if left is None and right is None:
            rec[i] = corr[i]
            continue

        if left is not None and right is not None:
            lval = left[1]
            rval = right[1]
            lo = min(lval, rval) - 0.35 * scale
            hi = max(lval, rval) + 0.35 * scale

            if rec[i] < lo or rec[i] > hi:
                rec[i] = corr[i]
                continue

            old_cost = abs(corr[i] - lval) + abs(corr[i] - rval)
            new_cost = abs(rec[i] - lval) + abs(rec[i] - rval)
            if old_cost > 1e-9 and new_cost > old_cost * 1.28:
                rec[i] = corr[i]
                continue
        else:
            side_val = left[1] if left is not None else right[1]
            if abs(rec[i] - side_val) > 3.2 * scale:
                rec[i] = corr[i]
                continue

    return pd.Series(rec, index=corrupted.index, name="recovered")


def recover_linear_simple(corrupted: pd.Series, anomaly_flag: np.ndarray) -> pd.Series:
    labels = np.asarray(anomaly_flag, dtype=int)
    out = corrupted.astype(float).copy()
    out.loc[labels == 1] = np.nan
    out = out.interpolate(method="linear", limit_direction="both")
    out = recovery_safety_guard(corrupted, out, labels)
    return out


def recover_type_aware(corrupted: pd.Series, anomaly_flag: np.ndarray, anomaly_types: np.ndarray) -> pd.Series:
    labels = np.asarray(anomaly_flag, dtype=int)
    types = np.asarray(anomaly_types, dtype=object)

    base = corrupted.astype(float).to_numpy(copy=True)
    original = corrupted.astype(float).to_numpy(copy=False)

    # noise için hafif yumuşatma
    rolling3 = pd.Series(original).rolling(window=3, center=True, min_periods=1).mean().to_numpy()

    for s, e in _segment_bounds(labels):
        seg_type = _majority_type(types, s, e)
        left, right = _nearest_clean_neighbors(original, labels, s)
        if right is None:
            _, right = _nearest_clean_neighbors(original, labels, e)

        if seg_type == "noise":
            for i in range(s, e + 1):
                # aşırı düzeltme yapmadan hafif düzelt
                base[i] = 0.80 * original[i] + 0.20 * rolling3[i]

        elif seg_type == "spike":
            for i in range(s, e + 1):
                lpt, rpt = _nearest_clean_neighbors(original, labels, i)
                if lpt is not None and rpt is not None and rpt[0] > lpt[0]:
                    t = (i - lpt[0]) / max(1, (rpt[0] - lpt[0]))
                    base[i] = (1.0 - t) * lpt[1] + t * rpt[1]
                else:
                    base[i] = original[i]

        elif seg_type == "burst":
            seg_len = e - s + 1
            if left is not None and right is not None and right[0] > left[0]:
                # pencere bazlı lineer süreklilik
                for i in range(s, e + 1):
                    t = (i - left[0]) / max(1, (right[0] - left[0]))
                    lin = (1.0 - t) * left[1] + t * right[1]
                    base[i] = lin

                # uzun burst için lokal regresyon karışımı
                if seg_len >= 6:
                    ws = max(0, s - 16)
                    we = min(len(original), e + 17)
                    x = np.arange(ws, we)
                    y = original[ws:we]
                    good = labels[ws:we] == 0
                    if np.sum(good) >= 4:
                        coef = np.polyfit(x[good], y[good], deg=1)
                        pred = coef[0] * np.arange(s, e + 1) + coef[1]
                        base[s : e + 1] = 0.55 * base[s : e + 1] + 0.45 * pred
            else:
                base[s : e + 1] = original[s : e + 1]

        elif seg_type == "stuck":
            if left is not None and right is not None:
                lval, rval = left[1], right[1]
                span = abs(rval - lval)
                if span < 1e-9:
                    # gerçek plato geçişini koru
                    base[s : e + 1] = lval
                else:
                    # adım geçişini bozmadan sınırlı lineer geçiş
                    for i in range(s, e + 1):
                        t = (i - s + 1) / (e - s + 2)
                        base[i] = (1.0 - t) * lval + t * rval
            else:
                base[s : e + 1] = original[s : e + 1]

        else:
            # bilinmeyen tiplerde konservatif fallback
            for i in range(s, e + 1):
                lpt, rpt = _nearest_clean_neighbors(original, labels, i)
                if lpt is not None and rpt is not None:
                    t = (i - lpt[0]) / max(1, (rpt[0] - lpt[0]))
                    base[i] = (1.0 - t) * lpt[1] + t * rpt[1]
                else:
                    base[i] = original[i]

    out = pd.Series(base, index=corrupted.index, name="recovered")
    out = recovery_safety_guard(corrupted, out, labels)

    # normal noktaları kesinlikle değiştirme
    final_vals = out.to_numpy(copy=True)
    final_vals[labels == 0] = original[labels == 0]
    return pd.Series(final_vals, index=corrupted.index, name="recovered")
