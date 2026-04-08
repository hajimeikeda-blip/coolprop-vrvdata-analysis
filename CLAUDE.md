# coolprop-vrvdata-analysis プロジェクト

## 概要

ダイキン VRV システム（業務用エアコン）の実機計測データを可視化・分析するプロジェクト。
CoolProp で冷媒物性を計算し、Plotly でインタラクティブな P-h 線図を生成する。

---

## データ構成

```
data/
├── Esprit/RXYP615FC/          # Esprit 連携データ（615FC）
├── Esprit/RXYP615FC_mollier/  # Esprit P-h 線図用データ
├── Gauss/20260407_RXYP335FC/  # GAUSS 連携データ（335FC）★現在の作業対象
│   ├── Combined_RAM260407184816.parquet  # 統合済みデータ（10800行 × 111列）
│   ├── In1_*.parquet / In2_*.parquet     # 室内機データ（2台）
│   ├── Out1_*.parquet                    # 室外機データ
│   ├── RXYP335FC_circuit.yml            # 冷凍サイクル回路定義
│   └── RXYP335FC_Japanese_names.json    # カラム名日本語対応表
└── RQYP335FC/                 # 別機種データ
```

### Combined parquet の主要カラム

| カラム | 説明 |
|---|---|
| `Tdi` | 吐出温度 |
| `Ts` | 吸入温度 |
| `Tcg` | 凝縮温度 |
| `Tb` | 凝縮器出口温度（過冷却液） |
| `Teg` | 蒸発温度 |
| `Ta` | 外気温度 |
| `LMPa` / `HMPa` | 低圧 / 高圧 [MPa] |
| `rps1` | 圧縮機回転数 [rps]（`ﾏ0_INV1回転数(rps)` の整理済み列）|
| `EVM%` / `EVT%` / `EVJ%` | 電子膨張弁開度 |
| `In1_TH1` / `In1_TH3` | 室内機1 熱交換器出口 / 入口温度 |

---

## スクリプト・ノートブック

### `src/ph_diagram_gauss_335FC.py` ★メインスクリプト

RXYP335FC の P-h 線図アニメーションを生成するスタンドアロンスクリプト。
実行すると `ph_interactive_Gauss_335FC.html` を出力する。

```bash
poetry run python src/ph_diagram_gauss_335FC.py
```

**レイアウト構成（3行×2列）:**
- 左列 row1: 温度時系列
- 左列 row2: 圧縮機回転数（rps1）
- 左列 row3: EV開度（EVM%, EVT%, EVJ%）
- 右列 row1–3（rowspan）: P-h 線図（飽和線・等温線背景 + サイクル点）

**設計上の注意:**
- `secondary_y=True` は使用禁止。アニメーション `redraw=True` 時に二軸が再初期化され、
  EV開度トレースがフリッカーするため。EV と RPS は別行に分ける。
- フレームの `traces` インデックスは `bg_trace_count` 以降のダミートレースを指す。
- 軸参照: PH線図 → `xaxis='x2'`, `yaxis='y2'`（rowspan のため2番目の非Noneセル）
- 左列3行のx軸は `matches='x'` で連動。

### `src/ph-diagram_cheker_Esprit.py`

615FC（Esprit 連携）用の P-h 線図スクリプト。Polars 使用。
`polars_ph_analysis.html` を出力する。参考実装として維持。

### `notebooks/Plot_Gauss_335FC.ipynb`

335FC の探索的分析ノートブック。PH線図セルにバグが残っているが、
本スクリプト（`ph_diagram_gauss_335FC.py`）で代替済み。

---

## 回路定義 YAML（`RXYP335FC_circuit.yml`）

現状のノード定義:

| node_id | 温度カラム | 圧力カラム | ラベル |
|---|---|---|---|
| Discharge | Tdi | HMPa | 吐出 |
| Suction | Ts | LMPa | 吸入 |
| CondenserOut | Tb | HMPa | 凝縮器出口 |
| CondenserGas | null | null | （未設定） |
| In1_HexIn | In1_TH3 | LMPa | 室内機熱交換器入口 |
| In1_HexOut | In1_TH1 | LMPa | 室内機熱交換器出口 |

接続（connections）に `CondensorOut`（旧スペルミス）が残っているため、
`CondenserOut` に修正が必要。→ **TODO**

---

## 未完了タスク（TODO）

- [ ] `RXYP335FC_circuit.yml` の connections の `CondensorOut` → `CondenserOut` に修正
- [ ] `CondenserGas` ノードに適切な温度・圧力カラムを設定
- [ ] `notebooks/Plot_Gauss_335FC.ipynb` の PH線図セルをスクリプトに合わせて整理
- [ ] 室内機2台目（In2_TH1, In2_TH3）のノードを回路定義に追加するか検討
- [ ] RQYP335FC データの分析（`data/RQYP335FC/`）

---

## 実行環境

- Python: `poetry run python` で実行（pyenv 3.13.12）
- 主要ライブラリ: pandas, polars, plotly, CoolProp, pyyaml, pyarrow
- 冷媒: R410A（`FLUID = "R410A"` で固定）
