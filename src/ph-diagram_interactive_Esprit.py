"""
左側散布図でカーソル（ホバー/クリック）を当てた点の
Ph線図を右側にリアルタイム描画する。
Plotly のみ使用（Streamlit 不使用）。
"""

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import yaml
import CoolProp.CoolProp as CP
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- パス設定 ---
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "Esprit" / "RXYP615FC_mollier"
PARQUET_FILE = DATA_DIR / "full_result.parquet"
CIRCUIT_YAML = DATA_DIR / "RXYP615FC_circuit.yml"
OUTPUT_HTML = BASE_DIR / "ph_interactive_Esprit.html"

COEFF_CV = 1.561266
FLUID = "R410A"
DIV_ID = "ph-diagram-plot"


def load_data(parquet_path: Path) -> pl.DataFrame:
    df = pl.read_parquet(parquet_path)
    print(f"Shape: {df.shape}")
    return df


def get_p_sat(temp_c: float, fluid: str) -> float | None:
    if temp_c is None:
        return None
    try:
        p_pa = CP.PropsSI("P", "T", temp_c + 273.15, "Q", 1.0, fluid)
        return p_pa / 1e6
    except Exception:
        return None


def preprocess(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        (pl.col("室内EV CV値比率") / COEFF_CV).alias("室内EV CV値比率_補正")
    )
    df = df.with_columns(
        pl.col("圧縮機入口のガス飽和温度").map_elements(
            lambda t: get_p_sat(t, FLUID),
            return_dtype=pl.Float64,
        ).alias("圧縮機入口圧力_MPa(算出)")
    )
    df = df.with_columns(
        pl.when(pl.col("室内EV CV値比率_補正") > 100)
        .then(pl.lit("除外データ"))
        .otherwise(pl.lit("学習データ"))
        .alias("color_label")
    )
    return df


def add_ph_background(fig: go.Figure, fluid: str, row: int = 1, col: int = 2) -> None:
    Tc = CP.PropsSI("Tcrit", fluid)
    T_range = np.linspace(223.15, Tc - 0.1, 100)

    h_l, p_l, h_v, p_v = [], [], [], []
    for T in T_range:
        try:
            h_l.append(CP.PropsSI("H", "T", T, "Q", 0, fluid) / 1000.0)
            p_l.append(CP.PropsSI("P", "T", T, "Q", 0, fluid) / 1e6)
            h_v.append(CP.PropsSI("H", "T", T, "Q", 1, fluid) / 1000.0)
            p_v.append(CP.PropsSI("P", "T", T, "Q", 1, fluid) / 1e6)
        except Exception:
            continue

    fig.add_trace(
        go.Scatter(x=h_l, y=p_l, mode="lines", line=dict(color="black", width=2),
                   showlegend=False, hoverinfo="skip"),
        row=row, col=col,
    )
    fig.add_trace(
        go.Scatter(x=h_v, y=p_v, mode="lines", line=dict(color="black", width=2),
                   showlegend=False, hoverinfo="skip"),
        row=row, col=col,
    )

    T_celsius_list = [0, 20, 40, 60, 80, 100]
    p_range = np.geomspace(0.5e6, 5.0e6, 50)

    for Tc_val in T_celsius_list:
        T_kelvin = Tc_val + 273.15
        h_iso, p_iso = [], []
        for p in p_range:
            try:
                h_iso.append(CP.PropsSI("H", "T", T_kelvin, "P", p, fluid) / 1000.0)
                p_iso.append(p / 1e6)
            except Exception:
                continue

        if h_iso:
            fig.add_trace(
                go.Scatter(
                    x=h_iso, y=p_iso, mode="lines",
                    line=dict(color="green", width=1, dash="dot"),
                    opacity=0.4, showlegend=False, hoverinfo="skip",
                ),
                row=row, col=col,
            )
            x_ref = f"x{col}" if col > 1 else "x"
            y_ref = f"y{col}" if col > 1 else "y"
            fig.add_annotation(
                x=h_iso[-1], y=p_iso[-1], text=f"{Tc_val}°C",
                showarrow=False, font=dict(color="green", size=10),
                xanchor="left", xref=x_ref, yref=y_ref,
            )


def create_ph_traces(current_row: dict, config: dict) -> list:
    """1行分のデータ(辞書)からPh線図トレースのリストを返す"""
    color_map = {"r": "red", "b": "blue", "g": "green", "y": "yellow", "k": "black"}
    style_map = {"-": "solid", "--": "dash", ":": "dot"}

    traces = []
    points_coords = {}

    for node_id, info in config["nodes"].items():
        p_val = current_row.get(info.get("press_ref"))
        h_val = current_row.get(info.get("enthalpy_ref"))
        t_val = current_row.get(info.get("temp"))

        if p_val is None or h_val is None:
            continue
        if isinstance(p_val, pd.Series):
            p_val = p_val.iloc[0]
        if isinstance(h_val, pd.Series):
            h_val = h_val.iloc[0]
        if isinstance(t_val, pd.Series):
            t_val = t_val.iloc[0]

        t_display = f"{t_val:.1f} °C" if t_val is not None else "N/A"
        points_coords[node_id] = (h_val, p_val)
        traces.append(
            go.Scatter(
                x=[h_val], y=[p_val], mode="markers",  # mode="markers+text",
                name=info["label"], # text=[info["label"]], textposition="top center",
                marker=dict(size=10, symbol=info.get("symbol", "circle")),
                customdata=[[t_display, p_val]],
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    "H: %{x:.1f} kJ/kg<br>"
                    "P: %{y:.2f} MPa<br>"
                    "T: %{customdata[0]}<extra></extra>"
                ),
            )
        )

    for conn in config["connections"]:
        if conn["start"] in points_coords and conn["end"] in points_coords:
            h_s, p_s = points_coords[conn["start"]]
            h_e, p_e = points_coords[conn["end"]]
            c = color_map.get(conn.get("color"), conn.get("color", "black"))
            s = style_map.get(conn.get("style"), "solid")
            traces.append(
                go.Scatter(
                    x=[h_s, h_e], y=[p_s, p_e], mode="lines",
                    name=conn.get("label"),
                    line=dict(color=c, width=conn.get("width", 2), dash=s),
                    opacity=0.8,
                )
            )

    return traces


def build_figure(df_pl: pl.DataFrame, circuit_dict: dict) -> go.Figure:
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.5, 0.5],
        subplot_titles=("Scatter: RPS vs SC ", "P-h Diagram"),
        horizontal_spacing=0.1,
    )

    # 左側: 全データ散布図
    color_map_scatter = {"学習データ": "blue", "除外データ": "red"}
    colors = [color_map_scatter.get(label, "gray") for label in df_pl["color_label"]]

    fig.add_trace(
        go.Scatter(
            x=df_pl["圧縮機回転数"],
            y=df_pl["SC"],
            mode="markers",
            marker=dict(color=colors, opacity=0.5, size=6),
            name="All Data",
            customdata=list(range(len(df_pl))),  # JS がインデックスを取得するため
            hovertemplate="RPS: %{x}<br>SC: %{y:.2f}<br>Index: %{customdata}<extra></extra>",
        ),
        row=1, col=1,
    )

    # 右側: Ph背景
    add_ph_background(fig, fluid=FLUID, row=1, col=2)

    bg_trace_count = len(fig.data)

    # フレーム生成
    frames = []
    moving_data_len = 0

    def fmt(v):
        return f"{v:.2f}" if v is not None else "N/A"

    print(f"Building {len(df_pl)} frames...")
    for i, row in enumerate(df_pl.iter_rows(named=True)):
        ph_traces = create_ph_traces(row, config=circuit_dict)
        for trace in ph_traces:
            trace.update(xaxis="x2", yaxis="y2", showlegend=False)

        # 左側: 選択中マーカー
        selected_marker = go.Scatter(
            x=[row["圧縮機回転数"]],
            y=[row["SC"]],
            mode="markers",
            marker=dict(color="black", size=14, symbol="x",
                        line=dict(width=2, color="white")),
            name="Selected",
            xaxis="x1", yaxis="y1", showlegend=False,
        )

        # 右側: SC / SH 値テキスト（Ph線図の左上に表示）
        info_text = (
            f"圧縮機回転数: {fmt(row.get('圧縮機回転数'))} rps<br>"
            f"SC: {fmt(row.get('SC'))} ℃<br>"
            f"SH 室内熱交: {fmt(row.get('SH_室内熱交'))} ℃<br>"
            f"SH 過冷却熱交: {fmt(row.get('SH_過冷却熱交'))} ℃"
        )
        info_trace = go.Scatter(
            x=[110], y=[4.2],
            mode="text",
            text=[info_text],
            textposition="bottom right",
            textfont=dict(size=15, color="darkblue"),
            xaxis="x2", yaxis="y2",
            showlegend=False, hoverinfo="skip",
        )

        moving_data = [selected_marker, info_trace] + ph_traces
        moving_data_len = len(moving_data)

        frames.append(go.Frame(
            data=moving_data,
            name=str(i),
            traces=list(range(bg_trace_count, bg_trace_count + moving_data_len)),
        ))

    # ダミートレース（フレーム数分のスロット確保）
    for _ in range(moving_data_len):
        fig.add_trace(go.Scatter(x=[None], y=[None], showlegend=False))

    fig.frames = frames

    fig.update_layout(
        height=700,
        template="plotly_white",
        xaxis1=dict(title="Compressor RPS"),
        yaxis1=dict(title="SC [℃]"),
        xaxis2=dict(range=[100, 600], title="Enthalpy [kJ/kg]"),
        yaxis2=dict(type="log", range=[np.log10(0.3), np.log10(5.0)],
                    title="Pressure [MPa]"),
    )

    return fig


# --- 埋め込み JavaScript ---
# post_script は Plotly.newPlot().then(fn) の中で実行されるため
# `gd` は存在しない。{plot_id} プレースホルダーで div を取得する。
# plotly_hover: カーソルを当てた点でリアルタイム更新
# plotly_click:  クリックで確定選択（ホバーが外れても維持）
POST_SCRIPT = """
(function() {
    var gd = document.getElementById('{plot_id}');
    var lockedIdx = 0;
    var isLocked = false;

    function animateTo(idx) {
        Plotly.animate(gd, [String(idx)], {
            transition: {duration: 0},
            frame: {duration: 0, redraw: true}
        });
    }

    // 初期表示: 最初のデータ点
    animateTo(0);

    // ホバー: カーソル位置の点をリアルタイム表示
    gd.on('plotly_hover', function(eventData) {
        if (isLocked) return;
        var point = eventData.points[0];
        if (point.curveNumber === 0) {
            animateTo(point.pointIndex);
        }
    });

    // ホバーアウト: ロック中でなければ最後に確定した点に戻す
    gd.on('plotly_unhover', function() {
        if (!isLocked) {
            animateTo(lockedIdx);
        }
    });

    // クリック: 選択を確定（同じ点を再クリックでロック解除）
    gd.on('plotly_click', function(eventData) {
        var point = eventData.points[0];
        if (point.curveNumber !== 0) return;
        var idx = point.pointIndex;

        if (isLocked && lockedIdx === idx) {
            isLocked = false;
        } else {
            isLocked = true;
            lockedIdx = idx;
            animateTo(idx);
        }
    });
})();
"""


def main():
    df = load_data(PARQUET_FILE)
    df = preprocess(df)

    with open(CIRCUIT_YAML, "r", encoding="utf-8") as f:
        circuit_dict = yaml.safe_load(f)

    fig = build_figure(df, circuit_dict)

    fig.write_html(
        str(OUTPUT_HTML),
        div_id=DIV_ID,
        auto_play=False,
        post_script=POST_SCRIPT,
    )
    print(f"Saved: {OUTPUT_HTML}")


if __name__ == "__main__":
    main()
