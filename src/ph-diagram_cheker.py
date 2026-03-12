import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import CoolProp.CoolProp as CP
import numpy as np

# ページ設定: ワイドモードで左右に並べやすくする
st.set_page_config(layout="wide", page_title="Refrigeration Cycle Analyzer")


# --- データ読み込み（キャッシュ機能で高速化） ---
@st.cache_data
def load_data():
    path_to_file = '../data/vrv-transfer-learning/transfer_learning_RQYP224FC_20260225.parquet'
    df = pd.read_parquet(path_to_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df

# pH線図の背景（飽和線）を描画する関数
def plot_ph_background(fluid_name="R410A",color='black'):
    Tc = CP.PropsSI('Tcrit', fluid_name)
    T_min = 223.15  # -50 degC
    T_range = np.linspace(T_min, Tc - 0.1, 100)
    
    h_l = [CP.PropsSI('H', 'T', T, 'Q', 0, fluid_name) / 1000 for T in T_range]
    p_l = [CP.PropsSI('P', 'T', T, 'Q', 0, fluid_name) / 1e6 for T in T_range]
    h_v = [CP.PropsSI('H', 'T', T, 'Q', 1, fluid_name) / 1000 for T in T_range]
    p_v = [CP.PropsSI('P', 'T', T, 'Q', 1, fluid_name) / 1e6 for T in T_range]
    
    #plt.plot(h_l, p_sat, 'k-', lw=1.5, alpha=0.7)
    #plt.plot(h_v, p_sat, 'k-', lw=1.5, alpha=0.7)

    plt.plot(h_l, p_l, color=color, lw=2, label=f'{fluid_name} Sat. Curve')
    plt.plot(h_v, p_v, color=color, lw=2)

    # --- 2. 等温線の追加 ---
    # 表示したい温度（Celsius）を設定
    T_celsius = [0, 20, 40, 60, 80, 100]
    # 圧力の描画範囲（y軸の範囲に合わせる）
    p_range = np.geomspace(0.5e6, 10e6, 100) # 0.5MPaから10MPaまで

    for Tc in T_celsius:
        T_kelvin = Tc + 273.15
        h_isotherm = []
        p_isotherm = []
        
        for p in p_range:
            try:
                # 指定した温度・圧力でのエンタルピーを計算
                h = CP.PropsSI('H', 'T', T_kelvin, 'P', p, fluid_name) / 1000
                h_isotherm.append(h)
                p_isotherm.append(p / 1e6)
            except:
                continue # 物性値が計算できない範囲（超臨界など）はスキップ
        
        plt.plot(h_isotherm, p_isotherm, 'g--', lw=1, alpha=0.6)
        # 線の上に温度をテキスト表示（適当な位置：P=1.0MPa付近）
        if h_isotherm:
            plt.text(h_isotherm[10], p_isotherm[10], f'{Tc}°C', fontsize=9, color='green')


# --- メイン処理 ---
st.title("VRV System Analysis Dashboard")

try:
    df = load_data()
except Exception as e:
    st.error(f"データの読み込みに失敗しました。パスを確認してください: {e}")
    st.stop()

# --- サイドバー操作 ---
st.sidebar.header("Data Selection")
# 表示するデータ期間の絞り込み（データ量が多い場合のため）
# get_label ではなく format_func を使用します
start_idx, end_idx = st.sidebar.select_slider(
    "表示期間の選択範囲",
    options=list(df.index),
    value=(0, min(1000, len(df)-1)),
    format_func=lambda x: df.iloc[x]['timestamp'].strftime('%m/%d %H:%M')
)
df_slice = df.iloc[start_idx:end_idx].reset_index(drop=True)

# --- メインスライダー：解析する「時刻」の選択 ---
#selected_idx = st.slider(
#    "解析時刻を選択してください (Time Cursor)",
#    0, len(df_slice)-1, 0,
#    format="時刻インデックス: %d"
#)

# --- 操作系はサイドバーにまとめる ---
st.sidebar.subheader("Time Navigation")
selected_idx = st.sidebar.slider(
    "Time Cursor",
    0, len(df_slice)-1, 0, # 元の値を保持
    key="main_slider"
)
current_row = df_slice.iloc[selected_idx]
current_time = current_row['timestamp']

st.info(f"現在選択中の時刻: {current_time}")

# --- 2カラムレイアウト ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Time Series Plot")
    
    # Plotlyで時系列グラフ作成
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, 
        vertical_spacing=0.1,
        subplot_titles=("Temperatures (Result)", "Control Input"),
        specs=[[{"secondary_y": False}], [{"secondary_y": True}]]
    )

    # 上段: 温度
    temp_cols = ['teg','tcg', 't_liquid', 'compressor_1_dischargetemp']
    for col in temp_cols:
        fig.add_trace(go.Scatter(x=df_slice['timestamp'], y=df_slice[col], name=col), row=1, col=1)

    # 下段: 制御
    fig.add_trace(go.Scatter(x=df_slice['timestamp'], y=df_slice['rpm'], name='RPM', line=dict(color='black')), row=2, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=df_slice['timestamp'], y=df_slice['indoor_ev_pulse'], name='EV Pulse', line=dict(color='red', dash='dot')), row=2, col=1, secondary_y=True)

    # スライダーで選択した時刻に赤い縦線を引く
    fig.add_vline(x=current_time, line_width=3, line_dash="dash", line_color="red")
    
    fig.update_layout(height=600, hovermode='x unified', template='plotly_white', legend=dict(orientation="h", y=-0.1))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("P-h Diagram")
    
    # MatplotlibでpH線図作成
    fig_ph, ax = plt.subplots(figsize=(8, 8))
    fluid = "R410A"
    
    plot_ph_background(fluid)

    try:
        # スライダーで選ばれた時刻のデータを使用して計算
        T_evap_C = current_row['teg']
        T_cond_C = current_row['tcg']
        T_dis_C = current_row['compressor_1_dischargetemp']
        T_accum_in_C = current_row['ts'] #アキュムレータ入口温度
        T_cond_out_C = current_row['t_liquid']      # 凝縮器出口
        T_airout_C = 35.0        # 外気温（仮設定）
        T_liquidpipe_C = current_row['t_liquid']  # 液管温度(データがないので't_liquid'を入れる)
        T_doubleHX_C = current_row['tsh']      # 二重管
        T_indoor_gas_C = current_row['indoor_gas']      # 室内機熱交換器のガス（気相）
        fluid_name = "R410A"

        #st.write(T_liquidpipe_C,T_doubleHX_C)
        
        # --- 圧力・エンタルピ計算 ---
        P_evap_Pa = CP.PropsSI('P', 'T', T_evap_C + 273.15, 'Q', 1, fluid)
        P_cond_Pa = CP.PropsSI('P', 'T', T_cond_C + 273.15, 'Q', 0, fluid)

        C_K_convert = 273.15
        
        P_evap_MPa = P_evap_Pa / 1e6
        P_cond_MPa = P_cond_Pa / 1e6

        # R410Aなどの非共沸冷媒は、Q=0.5などでCoolPropに入力するとエラーになる
        # P_doubleHX_Pa = CP.PropsSI('P','T', T_doubleHX_C + 273.15,'Q',0.5,fluid)
        # P_doubleHX_Pa = CP.PropsSI('P','T', T_doubleHX_C + 273.15,'Q',1.0,fluid)
        # P_doubleHX_MPa = P_doubleHX_Pa / 1e6


        def get_h(T_C, P_Pa, fluid):
            T_K = T_C + 273.15
    
            try:
                # 指定したT, Pで直接計算を試みる（単相域：過冷却液 or 過熱蒸気）
                return CP.PropsSI('H', 'T', T_K, 'P', P_Pa, fluid) / 1000.0
            
            except ValueError:
                # T, Pが飽和域にある場合、あるいは境界上の場合はエラーになるため
                # 飽和蒸気の値を返すなどの例外処理が必要
                print("Warning: State is in the saturation region or input is invalid.")
                return None

        h_inlet = get_h(T_accum_in_C, P_evap_Pa,fluid_name)       # Inlet (Accumu In)
        h_dis = get_h(T_dis_C, P_cond_Pa,fluid_name)             # Discharge
        h_cond_out = get_h(T_cond_out_C, P_cond_Pa,fluid_name)   # Condenser Out
        h_airout = get_h(T_airout_C, P_cond_Pa,fluid_name)  # Condenser (Air outdoorunit) 外気温度
        h_cond_liquid = get_h(T_liquidpipe_C, P_cond_Pa,fluid_name) # Condenser Liquid pipe
        
        h_doubleHX_in = h_cond_out  # 二重管入口　凝縮器出口のエンタルピーは同じ
        h_doubleHX_out = get_h(T_doubleHX_C,P_evap_Pa,fluid_name) # DoubleHX out 　二重管出口のエンタルピーは過冷却管温度から算出する
        
        h_evap_indoor_gas = get_h(T_indoor_gas_C, P_evap_Pa,fluid_name) # 室内ガスガス管
        # 飽和線の交点（補助用）
        h_cond_gas = CP.PropsSI('H', 'P', P_cond_Pa, 'Q', 1, fluid_name) / 1000.0
        h_cond_liquid_sat = CP.PropsSI('H', 'P', P_cond_Pa, 'Q', 0, fluid_name) / 1000.0

       
        # --- 3. プロット (Points) ---
        pts = [
            (h_inlet, P_evap_MPa, 'bs', "Inlet(Accumu In)"),
            (h_evap_indoor_gas, P_evap_MPa, 'bo', "Evaporator Gas"),
            (h_dis, P_cond_MPa, 'rs', "Discharge"),
            (h_cond_out, P_cond_MPa, 'gs', "Condenser Out"),
            (h_cond_gas, P_cond_MPa, 'gx', "Condenser Gas"),
            (h_cond_liquid, P_cond_MPa, 'bs', "Condenser Liquid pipe"),
            (h_doubleHX_in, P_evap_MPa, 'go', "DoubleHX in (EVT out)"),
            (h_doubleHX_out, P_evap_MPa, 'go', "DoubleHX out "),
            (h_airout, P_cond_MPa, 'm.', "Outdoor Unit (Air out)"), # 外気温度 
            (h_cond_out, P_evap_MPa, '.', "Evaporator In")
        ]

        st.write('DoubleHX out Enthalpy=',h_doubleHX_out,'Enthalpy (Evap gas)=',h_evap_indoor_gas)
        st.write('SC: T_cg - T_liquidpipe_cg',T_cond_C - T_liquidpipe_C)
        st.write('SH (evap): T_indoor_gas - T_eg',T_indoor_gas_C - T_evap_C)
        st.write('SH (doubleHX):T_doubleHX - T_eg',T_doubleHX_C - T_evap_C)

        for h, p, style, lbl in pts:
            plt.plot(h, p, style, label=lbl)

        # --- 4. プロット (Lines) ---
        # 圧縮工程
        plt.plot([h_inlet, h_dis], [P_evap_MPa, P_cond_MPa], 'r-', lw=2, label="Compression")
    
        # 凝縮工程 (吐出 -> 凝縮器出口)
        plt.plot([h_dis, h_cond_out], [P_cond_MPa, P_cond_MPa], 'g-', lw=2)
    
        # サブクール・液管・二重管ライン
        #plt.plot([h_cond_out, h_cond_airout], [P_cond_MPa, P_cond_MPa], 'b-', lw=1.5)
        #plt.plot([h_cond_airout, h_cond_liquid], [P_cond_MPa, P_cond_MPa], 'b-', lw=1.5)
        #plt.plot([h_cond_airout, h_doubleHX_in], [P_cond_MPa, P_cond_MPa], 'b-', lw=1.5)
    
        # 蒸発器ライン (膨張弁通過後の等エンタルピ想定)
        # ここでは h_cond_out を基準点として蒸発器入口へつなぐ
        plt.plot([h_cond_out, h_inlet], [P_evap_MPa, P_evap_MPa], 'b:', lw=2, label="Evaporator line")

        # 膨張弁 (等エンタルピ)
        plt.plot([h_cond_out, h_cond_out], [P_cond_MPa, P_evap_MPa], 'k--', lw=1)

        # 二重管熱交換器のライン
        plt.plot([h_cond_out, h_doubleHX_out], [P_evap_MPa, P_evap_MPa], 'b--', lw=1)
        
        ax.set_yscale('log')
        ax.set_xlim(100, 600)
        ax.set_ylim(0.5, 5.0) # 圧力範囲はデータに合わせて調整
        ax.set_xlabel('Enthalpy [kJ/kg]')
        ax.set_ylabel('Pressure [MPa]')
        ax.grid(True, which="both", alpha=0.3)
        #ax.legend()

        # ax.legend(loc='best', borderaxespad=0, fontsize='small')
        ax.legend(loc='upper left', fontsize='small') 

    except Exception as e:
        st.warning(f"この時刻の物性計算ができませんでした: {e}")

    st.pyplot(fig_ph)