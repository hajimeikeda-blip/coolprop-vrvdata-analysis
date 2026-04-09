"""
PH線図生成スクリプト: RXYP335FC (GAUSS連携データ)

- Combined parquetを読み込み、冷凍サイクルをPH線図上にプロット
- Plotlyを使ったアニメーション付きインタラクティブHTMLを出力

レイアウト:
  左列 row1: 温度時系列         (高め)
  左列 row2: 圧縮機回転数 (RPS) (低め)
  左列 row3: EV開度 (EVM%等)    (低め)
  左列 row4: FanSt / 制御ﾓｰﾄﾞ / 外ｻｰﾓON中 (低め)
  右列 row1-3 (rowspan=3): P-h 線図
  右列 row4: 温度センサー大小関係バリデーション表
  ※ secondary_y は使わない（アニメーション時にフリッカーするため）
"""

import os
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import CoolProp.CoolProp as CP
import yaml

# ─────────────────────────────────────────────
# パス設定
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "Gauss", "20260407_RXYP335FC")
PARQUET_PATH = os.path.join(DATA_DIR, "Combined_RAM260407184816.parquet")
CIRCUIT_YAML_PATH = os.path.join(DATA_DIR, "RXYP335FC_circuit.yml")
JAPANESE_NAMES_PATH = os.path.join(DATA_DIR, "RXYP335FC_Japanese_names.json")
OUTPUT_HTML = os.path.join(BASE_DIR, "ph_interactive_Gauss_335FC.html")

FLUID = "R410A"

# 4行×2列レイアウトの軸参照（非None セルの順に連番）
# (1,1)→x/y, (1,2 rowspan=3)→x2/y2, (2,1)→x3/y3, (3,1)→x4/y4, (4,1)→x5/y5, (4,2)→table
PH_XREF = "x2"
PH_YREF = "y2"

# ─────────────────────────────────────────────
# 温度センサーバリデーション設定
# ─────────────────────────────────────────────
# データカラム名
VAL_SENSORS = ["Tdi", "Tcg", "Tb", "Tf", "Tm", "Teg", "Tsh", "Ts", "Tsc", "In1_TH3", "In1_TH2"]
# 表示用短縮名
VAL_LABELS  = ["Tdi", "Tcg", "Tb", "Tf", "Tm", "Teg", "Tsh", "Ts", "Tsc", "Th3", "Th2"]

# EXPECTED_MATRIX[i][j] = row_i センサー vs col_j センサーの期待大小関係
# "大": row_i > col_j が期待値, "小": row_i < col_j, "保留": 未確定, "-": 同一
EXPECTED_MATRIX = [
    # Tdi  Tcg   Tb   Tf   Tm    Teg   Tsh   Ts    Tsc   Th3   Th2
    ["-",  "大", "大", "大", "大", "大",  "大",  "大",  "大",  "大",  "大"],  # Tdi
    ["小",  "-", "大", "大", "大", "大",  "大",  "大",  "大",  "大",  "大"],  # Tcg
    ["小", "小",  "-", "大", "大", "大",  "大",  "大",  "大",  "大",  "大"],  # Tb
    ["小", "小", "小",  "-", "大", "大",  "大",  "大",  "大",  "大",  "大"],  # Tf
    ["小", "小", "小", "小",  "-", "保留","小",  "大",  "保留","大",  "大"],  # Tm
    ["小", "小", "小", "小", "保留", "-", "保留","小",  "大",  "小",  "大"],  # Teg
    ["小", "小", "小", "小", "大", "保留",  "-", "保留","小",  "保留","大"],  # Tsh
    ["小", "小", "小", "小", "小", "大",  "保留",  "-", "保留","保留","大"],  # Ts
    ["小", "小", "小", "小", "保留","小",  "大", "保留",  "-", "保留","大"],  # Tsc
    ["小", "小", "小", "小", "小", "大",  "保留","保留","保留",  "-", "大"],  # Th3
    ["小", "小", "小", "小", "小", "小",  "小",  "小",  "小",  "小",  "-"],  # Th2
]

# 色定義
COLOR_OK      = "#c8e6c9"  # 緑: 期待通り
COLOR_NG      = "#ffcdd2"  # 赤: 期待と不一致
COLOR_PENDING = "#fff9c4"  # 黄: 保留
COLOR_HEADER  = "#546e7a"  # ヘッダー背景
COLOR_ROWLBL  = "#eceff1"  # 行ラベル背景
COLOR_DIAG    = "#f5f5f5"  # 対角


def compute_table_data(row_dict: dict) -> tuple[list, list]:
    """
    現在の行データから大小比較テーブルを計算する。
    戻り値: (col_values, col_colors) — go.Table の cells.values / cells.fill.color 形式
    各リストは列ごとのリスト（最初の列 = 行ラベル）
    """
    n = len(VAL_SENSORS)
    col_values = [VAL_LABELS]               # 1列目: 行ラベル
    col_colors = [[COLOR_ROWLBL] * n]       # 1列目の色

    for j in range(n):
        vals, colors = [], []
        for i in range(n):
            exp = EXPECTED_MATRIX[i][j]
            if exp == "-":
                vals.append("—")
                colors.append(COLOR_DIAG)
                continue

            vi = row_dict.get(VAL_SENSORS[i])
            vj = row_dict.get(VAL_SENSORS[j])

            if vi is None or vj is None or pd.isna(vi) or pd.isna(vj):
                vals.append("?")
                colors.append(COLOR_DIAG)
                continue

            actual = "大" if vi > vj else "小"

            if exp == "保留":
                vals.append(actual)
                colors.append(COLOR_PENDING)
            elif exp == actual:
                vals.append(actual)
                colors.append(COLOR_OK)
            else:
                vals.append(actual)
                colors.append(COLOR_NG)

        col_values.append(vals)
        col_colors.append(colors)

    return col_values, col_colors


def make_initial_table() -> go.Table:
    n = len(VAL_SENSORS)
    return go.Table(
        columnwidth=[2.2] + [1.0] * n,
        header=dict(
            values=[""] + VAL_LABELS,
            fill_color=COLOR_HEADER,
            font=dict(size=9, color="white"),
            align="center",
            height=20,
        ),
        cells=dict(
            values=[VAL_LABELS] + [["-"] * n] * n,
            fill=dict(color=[[COLOR_ROWLBL] * n] + [[COLOR_DIAG] * n] * n),
            font=dict(size=9, color="black"),
            align="center",
            height=20,
        ),
    )


# ─────────────────────────────────────────────
# PH線図ヘルパー関数
# ─────────────────────────────────────────────
def get_h(T_C, P_MPa, fluid):
    """温度[℃]と圧力[MPa]からエンタルピー[kJ/kg]を計算する。飽和域へのフォールバックあり。"""
    if pd.isna(T_C) or pd.isna(P_MPa):
        return None
    T_K = T_C + 273.15
    P_Pa = P_MPa * 1e6
    try:
        return CP.PropsSI("H", "T", T_K, "P", P_Pa, fluid) / 1000.0
    except ValueError:
        try:
            T_sat_K = CP.PropsSI("T", "P", P_Pa, "Q", 0, fluid)
            q = 0 if T_K < T_sat_K else 1
            return CP.PropsSI("H", "P", P_Pa, "Q", q, fluid) / 1000.0
        except Exception:
            return None


def add_ph_background(fig, fluid, row, col, xref, yref):
    """飽和線と等温線をサブプロットの指定セルに追加する。"""
    Tc = CP.PropsSI("Tcrit", fluid)
    T_min = 223.15
    T_range = np.linspace(T_min, Tc - 0.1, 100)

    h_l, p_l, h_v, p_v = [], [], [], []
    for T in T_range:
        h_l.append(CP.PropsSI("H", "T", T, "Q", 0, fluid) / 1000.0)
        p_l.append(CP.PropsSI("P", "T", T, "Q", 0, fluid) / 1e6)
        h_v.append(CP.PropsSI("H", "T", T, "Q", 1, fluid) / 1000.0)
        p_v.append(CP.PropsSI("P", "T", T, "Q", 1, fluid) / 1e6)

    fig.add_trace(go.Scatter(x=h_l, y=p_l, mode="lines",
                             line=dict(color="black", width=2),
                             showlegend=False, hoverinfo="skip"), row=row, col=col)
    fig.add_trace(go.Scatter(x=h_v, y=p_v, mode="lines",
                             line=dict(color="black", width=2),
                             showlegend=False, hoverinfo="skip"), row=row, col=col)

    T_celsius_list = [0, 20, 40, 60, 80, 100]
    p_range = np.geomspace(0.5e6, 5.0e6, 50)

    for Tc_val in T_celsius_list:
        T_K = Tc_val + 273.15
        h_iso, p_iso = [], []
        for p in p_range:
            try:
                h_iso.append(CP.PropsSI("H", "T", T_K, "P", p, fluid) / 1000.0)
                p_iso.append(p / 1e6)
            except Exception:
                continue
        if not h_iso:
            continue
        fig.add_trace(go.Scatter(x=h_iso, y=p_iso, mode="lines",
                                 line=dict(color="green", width=1, dash="dot"),
                                 opacity=0.3, showlegend=False, hoverinfo="skip"),
                      row=row, col=col)
        fig.add_annotation(x=h_iso[-1], y=p_iso[-1], text=f"{Tc_val}°C",
                           showarrow=False, font=dict(color="green", size=9),
                           xanchor="left", xref=xref, yref=yref)


def build_cycle_traces(row_data, config, fluid, xaxis, yaxis):
    """
    1行分のデータから冷凍サイクルのトレースリストを生成する。

    ph_flag に応じたエンタルピー決定ロジック:
      None        : T, P から CoolProp で計算（飽和域フォールバックあり）
      "Tsc"       : T, P から CoolProp で計算。二相状態の場合はノードをスキップ。
      "In1_HexIn" : ph_flag="Tsc" ノードのエンタルピーを流用（等エンタルピー膨張）。
    """
    traces = []
    points_coords = {}  # node_id -> (h [kJ/kg], p [MPa])

    # ph_flag="Tsc" を持つノードIDを先に探す（In1_HexIn が依存する）
    tsc_node_id = next(
        (nid for nid, ninfo in config["nodes"].items() if ninfo.get("ph_flag") == "Tsc"),
        None,
    )

    for node_id, info in config["nodes"].items():
        t_col   = info.get("temp")
        p_col   = info.get("press_ref")
        ph_flag = info.get("ph_flag")
        label   = info.get("label") or node_id

        # ── エンタルピー・圧力の決定 ──────────────────────────
        if ph_flag == "Tsc":
            # T, P から単相域のみ計算。二相状態はスキップ
            t_val = row_data.get(t_col)
            p_val = row_data.get(p_col)
            if t_val is None or p_val is None or pd.isna(t_val) or pd.isna(p_val):
                continue
            try:
                h = CP.PropsSI("H", "T", t_val + 273.15, "P", p_val * 1e6, fluid) / 1000.0
            except ValueError:
                continue  # 二相状態 → スキップ
            points_coords[node_id] = (h, p_val)

        elif ph_flag == "In1_HexIn":
            # Tsc ノードの h を流用し、圧力は自身の LMPa を使う（等エンタルピー膨張）
            if tsc_node_id is None or tsc_node_id not in points_coords:
                continue
            h_tsc, _ = points_coords[tsc_node_id]
            p_val = row_data.get(p_col)
            if p_val is None or pd.isna(p_val):
                continue
            points_coords[node_id] = (h_tsc, p_val)

        else:
            # 通常: T, P から CoolProp で計算（飽和域フォールバックあり）
            t_val = row_data.get(t_col)
            p_val = row_data.get(p_col)
            if t_val is None or p_val is None or pd.isna(t_val) or pd.isna(p_val):
                continue
            h = get_h(t_val, p_val, fluid)
            if h is None:
                continue
            points_coords[node_id] = (h, p_val)

        # ── トレース追加 ────────────────────────────────────────
        h_val, p_val_plot = points_coords[node_id]
        t_display = row_data.get(t_col) if t_col else None
        t_str = f"{t_display:.1f}" if t_display is not None and not pd.isna(t_display) else "N/A"

        traces.append(go.Scatter(
            x=[h_val], y=[p_val_plot],
            mode="markers+text",
            name=label, text=[label], textposition="top center",
            marker=dict(size=10, symbol="circle"),
            customdata=[[t_display if t_display is not None else float("nan"), p_val_plot]],
            hovertemplate=(
                "<b>%{text}</b><br>H: %{x:.1f} kJ/kg<br>"
                f"P: %{{y:.3f}} MPa<br>T: {t_str} °C<extra></extra>"
            ),
            showlegend=False, xaxis=xaxis, yaxis=yaxis,
        ))

    # ── 接続線 ──────────────────────────────────────────────────
    for conn in config.get("connections", []):
        start, end = conn["start"], conn["end"]
        if start not in points_coords or end not in points_coords:
            continue
        h1, p1 = points_coords[start]
        h2, p2 = points_coords[end]
        style = conn.get("style", "-")
        dash = "dash" if "--" in style else ("dot" if ":" in style else "solid")
        traces.append(go.Scatter(
            x=[h1, h2], y=[p1, p2], mode="lines",
            line=dict(color=conn.get("color", "blue"), width=2, dash=dash),
            showlegend=False, hoverinfo="skip", xaxis=xaxis, yaxis=yaxis,
        ))

    return traces


# ─────────────────────────────────────────────
# メイン処理
# ─────────────────────────────────────────────
def main():
    print("Loading data ...")
    df = pd.read_parquet(PARQUET_PATH)
    print(f"  Shape: {df.shape}")

    with open(CIRCUIT_YAML_PATH, "r", encoding="utf-8") as f:
        circuit_dict = yaml.safe_load(f)

    rename_dict = {}
    if os.path.exists(JAPANESE_NAMES_PATH):
        with open(JAPANESE_NAMES_PATH, "r", encoding="utf-8") as f:
            rename_dict = json.load(f)

    df_plot = df.iloc[::30].reset_index()
    print(f"  Animation frames: {len(df_plot)}")

    # ── サブプロット構成 ──
    # 左列 4行（温度 / RPS / EV開度 / FanSt）
    # 右列 row1-3: P-h 線図（rowspan=3）、row4: バリデーション表
    fig = make_subplots(
        rows=4, cols=2,
        column_widths=[0.52, 0.48],
        row_heights=[0.36, 0.13, 0.13, 0.38],
        specs=[
            [{"type": "xy"},    {"rowspan": 3, "type": "xy"}],
            [{"type": "xy"},    None],
            [{"type": "xy"},    None],
            [{"type": "xy"},    {"type": "table"}],
        ],
        subplot_titles=(
            "Temperatures", "P-h Diagram",
            "Compressor RPS", "",
            "EV Openings", "",
            "Fan Status / Control", "",
        ),
        vertical_spacing=0.05,
        horizontal_spacing=0.10,
    )

    # ── 静的トレース: 温度（row=1） ──
    temp_cols = ["Tdi", "Ti", "Tcg", "Tb", "Tf", "Tsh", "Tm", "Ta", "Tsc", "Ts", "Teg", "In1_TH3", "In1_TH2"]
    for c in temp_cols:
        if c in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[c],
                                     name=rename_dict.get(c, c), mode="lines"),
                          row=1, col=1)

    # ── 静的トレース: RPS（row=2） ──
    rps_col = "rps1"
    if rps_col in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[rps_col],
                                 name=rename_dict.get("RPS", "RPS"),
                                 mode="lines", line=dict(color="black")),
                      row=2, col=1)

    # ── 静的トレース: EV開度（row=3） ──
    ev_configs = [
        {"col": "EVM%", "color": "red",   "dash": "dot"},
        {"col": "EVT%", "color": "blue",  "dash": "dash"},
        {"col": "EVJ%", "color": "green", "dash": "dashdot"},
    ]
    for ev in ev_configs:
        c = ev["col"]
        if c in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[c],
                                     name=rename_dict.get(c, c), mode="lines",
                                     line=dict(color=ev["color"], dash=ev["dash"])),
                          row=3, col=1)

    # ── 静的トレース: FanSt / 制御ﾓｰﾄﾞ / 外ｻｰﾓON中（row=4） ──
    for col_name, color in [("FanSt", "purple"), ("制御ﾓｰﾄﾞ", "orange"), ("外ｻｰﾓON中", "steelblue")]:
        if col_name in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[col_name],
                                     name=rename_dict.get(col_name, col_name),
                                     mode="lines", line=dict(color=color)),
                          row=4, col=1)

    # ── 静的トレース: PH線図背景（row=1, col=2） ──
    add_ph_background(fig, FLUID, row=1, col=2, xref=PH_XREF, yref=PH_YREF)

    # 左列 x軸連動
    fig.update_xaxes(matches="x", row=2, col=1)
    fig.update_xaxes(matches="x", row=3, col=1)
    fig.update_xaxes(matches="x", row=4, col=1)

    # 軸ラベル
    fig.update_yaxes(title_text="Temp [℃]",          row=1, col=1)
    fig.update_yaxes(title_text="Speed [rps]",        row=2, col=1)
    fig.update_yaxes(title_text="EV ratio [%/step]",  row=3, col=1)
    fig.update_yaxes(title_text="Fan / Control",      row=4, col=1)

    # scatter 背景のトレース数を記録
    scatter_bg_end = len(fig.data)

    # ── 初期バリデーション表（row=4, col=2）──
    fig.add_trace(make_initial_table(), row=4, col=2)
    table_trace_idx = scatter_bg_end  # テーブルトレースのインデックス

    bg_trace_count = len(fig.data)  # = scatter_bg_end + 1
    print(f"  Background traces: {bg_trace_count} (table at idx {table_trace_idx})")

    # ── フレーム生成 ──
    print("Building animation frames ...")
    frames = []
    scatter_moving_len = 0

    temp_cols_present = [c for c in temp_cols if c in df.columns]
    rename_dict.setdefault("In1_TH3", "In1_TH3(液管)")
    rename_dict.setdefault("In1_TH2", "In1_TH2(ガス管)")
    temp_vals = df[temp_cols_present].dropna()
    y1_min = temp_vals.min().min() - 5
    y1_max = temp_vals.max().max() + 5
    y2_max = df[rps_col].max() + 10 if rps_col in df.columns else 100
    ev_cols_present = [ev["col"] for ev in ev_configs if ev["col"] in df.columns]
    y3_max = df[ev_cols_present].max().max() + 0.2 if ev_cols_present else 1.5
    fan_cols = [c for c in ["FanSt", "制御ﾓｰﾄﾞ", "外ｻｰﾓON中"] if c in df.columns]
    y4_min = df[fan_cols].min().min() - 1 if fan_cols else 0
    y4_max = df[fan_cols].max().max() + 1 if fan_cols else 5

    for _, row in df_plot.iterrows():
        row_dict = row.to_dict()
        ts = row_dict.get("Time", row_dict.get("index", ""))

        # PH線図サイクルトレース
        ph_traces = build_cycle_traces(row_dict, circuit_dict, FLUID,
                                       xaxis=PH_XREF, yaxis=PH_YREF)

        # PH線図内の情報テキスト
        def fmt(v):
            return f"{v:.1f}" if v is not None and not pd.isna(v) else "N/A"

        rps_val   = row_dict.get("rps1")
        sc_val    = (row_dict.get("Tcg") - row_dict.get("Tf")
                     if row_dict.get("Tcg") is not None and row_dict.get("Tf") is not None
                     else None)
        sh_sc_val = (row_dict.get("Tsh") - row_dict.get("Tm")
                     if row_dict.get("Tsh") is not None and row_dict.get("Tm") is not None
                     else None)
        sh_in_val = (row_dict.get("In1_TH3") - row_dict.get("Teg")
                     if row_dict.get("In1_TH3") is not None and row_dict.get("Teg") is not None
                     else None)

        info_text = (
            f"<b>圧縮機回転数</b>: {fmt(rps_val)} rps<br>"
            f"<b>SC</b> = Tcg − Tf: {fmt(sc_val)} ℃<br>"
            f"<b>SH(過冷却)</b> = Tsh − Tm: {fmt(sh_sc_val)} ℃<br>"
            f"<b>SH(室内熱交)</b> = In1_TH3 − Teg: {fmt(sh_in_val)} ℃"
        )
        info_trace = go.Scatter(
            x=[5], y=[4.5], mode="text",
            text=[info_text], textposition="bottom right",
            textfont=dict(size=13, color="darkblue"),
            xaxis=PH_XREF, yaxis=PH_YREF,
            showlegend=False, hoverinfo="skip",
        )

        # カーソル線（左列4行）
        cursor_temp = go.Scatter(x=[ts, ts], y=[y1_min, y1_max], mode="lines",
                                 line=dict(color="rgba(0,0,0,0.4)", width=1, dash="dash"),
                                 xaxis="x", yaxis="y", showlegend=False, hoverinfo="skip")
        cursor_rps  = go.Scatter(x=[ts, ts], y=[0, y2_max], mode="lines",
                                 line=dict(color="rgba(0,0,0,0.4)", width=1, dash="dash"),
                                 xaxis="x3", yaxis="y3", showlegend=False, hoverinfo="skip")
        cursor_ev   = go.Scatter(x=[ts, ts], y=[0, y3_max], mode="lines",
                                 line=dict(color="rgba(0,0,0,0.4)", width=1, dash="dash"),
                                 xaxis="x4", yaxis="y4", showlegend=False, hoverinfo="skip")
        cursor_fan  = go.Scatter(x=[ts, ts], y=[y4_min, y4_max], mode="lines",
                                 line=dict(color="rgba(0,0,0,0.4)", width=1, dash="dash"),
                                 xaxis="x5", yaxis="y5", showlegend=False, hoverinfo="skip")

        moving_scatter = [cursor_temp, cursor_rps, cursor_ev, cursor_fan, info_trace] + ph_traces
        scatter_moving_len = len(moving_scatter)

        # バリデーション表の更新
        col_values, col_colors = compute_table_data(row_dict)
        table_update = go.Table(
            columnwidth=[2.2] + [1.0] * len(VAL_SENSORS),
            header=dict(
                values=[""] + VAL_LABELS,
                fill_color=COLOR_HEADER,
                font=dict(size=9, color="white"),
                align="center", height=20,
            ),
            cells=dict(
                values=col_values,
                fill=dict(color=col_colors),
                font=dict(size=9, color="black"),
                align="center", height=20,
            ),
        )

        # scatter 用スロット + テーブル更新をひとつのフレームにまとめる
        frames.append(go.Frame(
            data=moving_scatter + [table_update],
            name=str(ts),
            traces=(list(range(bg_trace_count, bg_trace_count + scatter_moving_len))
                    + [table_trace_idx]),
        ))

    # ── ダミートレース（scatter 分のプレースホルダー） ──
    for _ in range(scatter_moving_len):
        fig.add_trace(go.Scatter(x=[None], y=[None], showlegend=False))

    fig.frames = frames

    # ── スライダー ──
    sliders = [dict(
        steps=[dict(
            method="animate",
            args=[[f.name], dict(mode="immediate", frame=dict(duration=0, redraw=True))],
            label=str(f.name),
        ) for f in frames],
        active=0,
        currentvalue={"prefix": "Time: "},
        pad={"t": 50},
        x=0, y=0,
    )]

    # ── レイアウト ──
    fig.update_layout(
        sliders=sliders,
        height=1100,
        title_text="RXYP335FC (GAUSS連携) P-h Diagram Analysis（年月日はダミー）",
        template="plotly_white",
        showlegend=True,
        hovermode="closest",
        xaxis2=dict(range=[0, 600], title="Enthalpy [kJ/kg]"),
        yaxis2=dict(type="log", range=[np.log10(0.3), np.log10(5.0)], title="Pressure [MPa]"),
    )

    print(f"Saving HTML to: {OUTPUT_HTML}")
    fig.write_html(OUTPUT_HTML)
    print("Done!")


if __name__ == "__main__":
    main()
