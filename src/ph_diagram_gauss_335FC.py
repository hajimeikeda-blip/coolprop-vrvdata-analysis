"""
PH線図生成スクリプト: RXYP335FC (GAUSS連携データ)

- Combined parquetを読み込み、冷凍サイクルをPH線図上にプロット
- Plotlyを使ったアニメーション付きインタラクティブHTMLを出力

レイアウト:
  左列 row1: 温度時系列
  左列 row2: 圧縮機回転数 (RPS)
  左列 row3: EV開度
  右列 row1-3 (rowspan): P-h 線図
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

# 3行×2列レイアウトの軸参照（非None セルの順に連番）
# (1,1)→x/y, (1,2 rowspan)→x2/y2, (2,1)→x3/y3, (3,1)→x4/y4
PH_XREF = "x2"
PH_YREF = "y2"


# ─────────────────────────────────────────────
# ヘルパー関数
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
    T_min = 223.15  # -50 ℃
    T_range = np.linspace(T_min, Tc - 0.1, 100)

    h_l, p_l, h_v, p_v = [], [], [], []
    for T in T_range:
        h_l.append(CP.PropsSI("H", "T", T, "Q", 0, fluid) / 1000.0)
        p_l.append(CP.PropsSI("P", "T", T, "Q", 0, fluid) / 1e6)
        h_v.append(CP.PropsSI("H", "T", T, "Q", 1, fluid) / 1000.0)
        p_v.append(CP.PropsSI("P", "T", T, "Q", 1, fluid) / 1e6)

    fig.add_trace(
        go.Scatter(x=h_l, y=p_l, mode="lines",
                   line=dict(color="black", width=2),
                   showlegend=False, hoverinfo="skip"),
        row=row, col=col,
    )
    fig.add_trace(
        go.Scatter(x=h_v, y=p_v, mode="lines",
                   line=dict(color="black", width=2),
                   showlegend=False, hoverinfo="skip"),
        row=row, col=col,
    )

    T_celsius_list = [0, 20, 40, 60, 80, 100]
    p_range = np.geomspace(0.5e6, 5.0e6, 50)

    for Tc_val in T_celsius_list:
        T_K = Tc_val + 273.15
        h_iso, p_iso = [], []
        for p in p_range:
            try:
                h = CP.PropsSI("H", "T", T_K, "P", p, fluid) / 1000.0
                h_iso.append(h)
                p_iso.append(p / 1e6)
            except Exception:
                continue

        if not h_iso:
            continue

        fig.add_trace(
            go.Scatter(x=h_iso, y=p_iso, mode="lines",
                       line=dict(color="green", width=1, dash="dot"),
                       opacity=0.3, showlegend=False, hoverinfo="skip"),
            row=row, col=col,
        )
        fig.add_annotation(
            x=h_iso[-1], y=p_iso[-1],
            text=f"{Tc_val}°C",
            showarrow=False,
            font=dict(color="green", size=9),
            xanchor="left",
            xref=xref,
            yref=yref,
        )


def build_cycle_traces(row_data, config, fluid, xaxis, yaxis):
    """
    1行分のデータから冷凍サイクルのトレースリストを生成する。
    xaxis/yaxis はPlotly の軸参照（例: 'x2', 'y2'）。
    """
    traces = []
    points_coords = {}

    for node_id, info in config["nodes"].items():
        t_col = info.get("temp")
        p_col = info.get("press_ref")
        t_val = row_data.get(t_col)
        p_val = row_data.get(p_col)

        if t_val is None or p_val is None:
            continue
        if pd.isna(t_val) or pd.isna(p_val):
            continue

        h = get_h(t_val, p_val, fluid)
        if h is None:
            continue

        points_coords[node_id] = (h, p_val)
        traces.append(
            go.Scatter(
                x=[h], y=[p_val],
                mode="markers+text",
                name=info["label"],
                text=[info["label"]],
                textposition="top center",
                marker=dict(size=10, symbol="circle"),
                customdata=[[t_val, p_val]],
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "H: %{x:.1f} kJ/kg<br>"
                    "P: %{y:.3f} MPa<br>"
                    f"T: {t_val:.1f} °C<extra></extra>"
                ),
                showlegend=False,
                xaxis=xaxis,
                yaxis=yaxis,
            )
        )

    for conn in config.get("connections", []):
        start, end = conn["start"], conn["end"]
        if start not in points_coords or end not in points_coords:
            continue
        h1, p1 = points_coords[start]
        h2, p2 = points_coords[end]

        style = conn.get("style", "-")
        dash = "solid"
        if "--" in style:
            dash = "dash"
        elif ":" in style:
            dash = "dot"

        traces.append(
            go.Scatter(
                x=[h1, h2], y=[p1, p2],
                mode="lines",
                line=dict(color=conn.get("color", "blue"), width=2, dash=dash),
                showlegend=False,
                hoverinfo="skip",
                xaxis=xaxis,
                yaxis=yaxis,
            )
        )

    return traces


# ─────────────────────────────────────────────
# メイン処理
# ─────────────────────────────────────────────
def main():
    # ── データ読み込み ──
    print("Loading data ...")
    df = pd.read_parquet(PARQUET_PATH)
    print(f"  Shape: {df.shape}")

    with open(CIRCUIT_YAML_PATH, "r", encoding="utf-8") as f:
        circuit_dict = yaml.safe_load(f)

    rename_dict = {}
    if os.path.exists(JAPANESE_NAMES_PATH):
        with open(JAPANESE_NAMES_PATH, "r", encoding="utf-8") as f:
            rename_dict = json.load(f)

    # サブサンプリング（30秒ごと）でアニメーションを軽量化
    df_plot = df.iloc[::30].reset_index()
    print(f"  Animation frames: {len(df_plot)}")

    # ── サブプロット構成 ──
    # 左列 3行（温度 / RPS / EV開度）、右列 rowspan=3（PH線図）
    # secondary_y を一切使わないことでアニメーション時のフリッカーを防止
    fig = make_subplots(
        rows=3, cols=2,
        column_widths=[0.55, 0.45],
        row_heights=[0.45, 0.275, 0.275],
        specs=[
            [{"type": "xy"}, {"rowspan": 3, "type": "xy"}],
            [{"type": "xy"}, None],
            [{"type": "xy"}, None],
        ],
        subplot_titles=("Temperatures", "P-h Diagram", "Compressor RPS", "", "EV Openings", ""),
        vertical_spacing=0.07,
        horizontal_spacing=0.10,
    )

    # ── 静的トレース: 温度（row=1, col=1 → xaxis='x', yaxis='y'）──
    temp_cols = ["Tdi", "Ti", "Tcg", "Tb", "Tf", "Tsh", "Tm", "Ta", "Tsc", "Ts", "Teg"]
    for c in temp_cols:
        if c in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df[c],
                           name=rename_dict.get(c, c), mode="lines"),
                row=1, col=1,
            )

    # ── 静的トレース: RPS（row=2, col=1 → xaxis='x3', yaxis='y3'）──
    rps_col = "rps1"
    if rps_col in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df[rps_col],
                       name=rename_dict.get("RPS", "RPS"),
                       mode="lines", line=dict(color="black")),
            row=2, col=1,
        )

    # ── 静的トレース: EV開度（row=3, col=1 → xaxis='x4', yaxis='y4'）──
    ev_configs = [
        {"col": "EVM%", "color": "red",   "dash": "dot"},
        {"col": "EVT%", "color": "blue",  "dash": "dash"},
        {"col": "EVJ%", "color": "green", "dash": "dashdot"},
    ]
    for ev in ev_configs:
        c = ev["col"]
        if c in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df[c],
                           name=rename_dict.get(c, c),
                           mode="lines",
                           line=dict(color=ev["color"], dash=ev["dash"])),
                row=3, col=1,
            )

    # ── 静的トレース: PH線図背景（row=1, col=2）──
    add_ph_background(fig, FLUID, row=1, col=2, xref=PH_XREF, yref=PH_YREF)

    # 左列3行のx軸を連動させる（ズーム・パンが同期）
    fig.update_xaxes(matches="x", row=2, col=1)
    fig.update_xaxes(matches="x", row=3, col=1)

    # 軸ラベル
    fig.update_yaxes(title_text="Temp [℃]",       row=1, col=1)
    fig.update_yaxes(title_text="Speed [rps]",     row=2, col=1)
    fig.update_yaxes(title_text="EV ratio [%/step]", row=3, col=1)

    bg_trace_count = len(fig.data)
    print(f"  Background traces: {bg_trace_count}")

    # ── フレーム生成 ──
    print("Building animation frames ...")
    frames = []
    moving_data_len = 0

    # カーソル線のY範囲
    temp_vals = df[temp_cols].dropna()
    y1_min = temp_vals.min().min() - 5
    y1_max = temp_vals.max().max() + 5
    y2_max = df[rps_col].max() + 10 if rps_col in df.columns else 100
    ev_cols_present = [ev["col"] for ev in ev_configs if ev["col"] in df.columns]
    y3_max = df[ev_cols_present].max().max() + 0.2 if ev_cols_present else 1.5

    for _, row in df_plot.iterrows():
        row_dict = row.to_dict()
        ts = row_dict.get("Time", row_dict.get("index", ""))

        # PH線図サイクルトレース
        ph_traces = build_cycle_traces(row_dict, circuit_dict, FLUID,
                                       xaxis=PH_XREF, yaxis=PH_YREF)

        # 時系列カーソル線（左列3行それぞれ）
        cursor_temp = go.Scatter(
            x=[ts, ts], y=[y1_min, y1_max],
            mode="lines",
            line=dict(color="rgba(0,0,0,0.4)", width=1, dash="dash"),
            xaxis="x", yaxis="y",
            showlegend=False, hoverinfo="skip",
        )
        cursor_rps = go.Scatter(
            x=[ts, ts], y=[0, y2_max],
            mode="lines",
            line=dict(color="rgba(0,0,0,0.4)", width=1, dash="dash"),
            xaxis="x3", yaxis="y3",
            showlegend=False, hoverinfo="skip",
        )
        cursor_ev = go.Scatter(
            x=[ts, ts], y=[0, y3_max],
            mode="lines",
            line=dict(color="rgba(0,0,0,0.4)", width=1, dash="dash"),
            xaxis="x4", yaxis="y4",
            showlegend=False, hoverinfo="skip",
        )

        moving_data = [cursor_temp, cursor_rps, cursor_ev] + ph_traces
        moving_data_len = len(moving_data)

        frames.append(
            go.Frame(
                data=moving_data,
                name=str(ts),
                traces=list(range(bg_trace_count, bg_trace_count + moving_data_len)),
            )
        )

    # ── ダミートレース（フレームプレースホルダー） ──
    for _ in range(moving_data_len):
        fig.add_trace(go.Scatter(x=[None], y=[None], showlegend=False))

    fig.frames = frames

    # ── スライダー ──
    sliders = [dict(
        steps=[
            dict(
                method="animate",
                args=[[f.name], dict(mode="immediate", frame=dict(duration=0, redraw=True))],
                label=str(f.name),
            )
            for f in frames
        ],
        active=0,
        currentvalue={"prefix": "Time: "},
        pad={"t": 50},
        x=0, y=0,
    )]

    # ── レイアウト ──
    fig.update_layout(
        sliders=sliders,
        height=900,
        title_text="RXYP335FC (GAUSS連携) P-h Diagram Analysis（年月日はダミー）",
        template="plotly_white",
        showlegend=True,
        hovermode="closest",
        xaxis2=dict(range=[100, 600], title="Enthalpy [kJ/kg]"),
        yaxis2=dict(type="log", range=[np.log10(0.3), np.log10(5.0)], title="Pressure [MPa]"),
    )

    # ── HTML 出力 ──
    print(f"Saving HTML to: {OUTPUT_HTML}")
    fig.write_html(OUTPUT_HTML)
    print("Done!")


if __name__ == "__main__":
    main()
