
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy import signal


st.set_page_config(page_title="Processamento simples de sinais IMU", layout="wide")

st.title("Processamento simples de sinais IMU")

st.markdown("""
Este app faz uma primeira etapa simples de processamento:

1. carrega dois arquivos;
2. lê as colunas de tempo, X, Y e Z;
3. aplica detrend;
4. interpola os sinais para 100 Hz;
5. aplica filtro passa-baixa;
6. permite escolher quais eixos plotar.
""")


# ============================================================
# PAINEL LATERAL
# ============================================================

st.sidebar.header("Carregar arquivos")

arquivo_1 = st.sidebar.file_uploader(
    "Carregar arquivo 1 — Tornozelo",
    type=["txt", "csv"],
    key="arquivo_1"
)

arquivo_2 = st.sidebar.file_uploader(
    "Carregar arquivo 2 — Cintura/Coluna",
    type=["txt", "csv"],
    key="arquivo_2"
)

st.sidebar.header("Parâmetros")

fs_novo = st.sidebar.number_input(
    "Frequência de interpolação (Hz)",
    min_value=10,
    max_value=500,
    value=100,
    step=10
)

freq_corte = st.sidebar.number_input(
    "Filtro passa-baixa (Hz)",
    min_value=0.1,
    max_value=40.0,
    value=10.0,
    step=0.5
)

ordem_filtro = st.sidebar.number_input(
    "Ordem do filtro Butterworth",
    min_value=1,
    max_value=8,
    value=2,
    step=1
)

st.sidebar.header("Eixos para plotar")

eixos_arquivo_1 = st.sidebar.multiselect(
    "Arquivo 1 — Tornozelo",
    ["X", "Y", "Z", "R"],
    default=["X", "Y", "Z"]
)

eixos_arquivo_2 = st.sidebar.multiselect(
    "Arquivo 2 — Cintura/Coluna",
    ["X", "Y", "Z", "R"],
    default=["X", "Y", "Z"]
)


# ============================================================
# PROCESSAMENTO
# ============================================================

if arquivo_1 is not None and arquivo_2 is not None:

    # ========================================================
    # LEITURA DO ARQUIVO 1
    # ========================================================

    texto_1 = arquivo_1.read().decode("utf-8", errors="ignore")

    dados_1 = pd.read_csv(
        io.StringIO(texto_1),
        sep=r"[;\t,]+",
        engine="python"
    )

    dados_1.columns = [col.strip().upper() for col in dados_1.columns]

    tempo_1 = pd.to_numeric(dados_1.iloc[:, 0], errors="coerce").to_numpy()
    x1_raw = pd.to_numeric(dados_1.iloc[:, 1], errors="coerce").to_numpy()
    y1_raw = pd.to_numeric(dados_1.iloc[:, 2], errors="coerce").to_numpy()
    z1_raw = pd.to_numeric(dados_1.iloc[:, 3], errors="coerce").to_numpy()

    mascara_1 = (
        np.isfinite(tempo_1) &
        np.isfinite(x1_raw) &
        np.isfinite(y1_raw) &
        np.isfinite(z1_raw)
    )

    tempo_1 = tempo_1[mascara_1]
    x1_raw = x1_raw[mascara_1]
    y1_raw = y1_raw[mascara_1]
    z1_raw = z1_raw[mascara_1]

    if np.median(np.diff(tempo_1)) > 1:
        tempo_1 = tempo_1 / 1000.0

    tempo_1 = tempo_1 - tempo_1[0]


    # ========================================================
    # LEITURA DO ARQUIVO 2
    # ========================================================

    texto_2 = arquivo_2.read().decode("utf-8", errors="ignore")

    dados_2 = pd.read_csv(
        io.StringIO(texto_2),
        sep=r"[;\t,]+",
        engine="python"
    )

    dados_2.columns = [col.strip().upper() for col in dados_2.columns]

    tempo_2 = pd.to_numeric(dados_2.iloc[:, 0], errors="coerce").to_numpy()
    x2_raw = pd.to_numeric(dados_2.iloc[:, 1], errors="coerce").to_numpy()
    y2_raw = pd.to_numeric(dados_2.iloc[:, 2], errors="coerce").to_numpy()
    z2_raw = pd.to_numeric(dados_2.iloc[:, 3], errors="coerce").to_numpy()

    mascara_2 = (
        np.isfinite(tempo_2) &
        np.isfinite(x2_raw) &
        np.isfinite(y2_raw) &
        np.isfinite(z2_raw)
    )

    tempo_2 = tempo_2[mascara_2]
    x2_raw = x2_raw[mascara_2]
    y2_raw = y2_raw[mascara_2]
    z2_raw = z2_raw[mascara_2]

    if np.median(np.diff(tempo_2)) > 1:
        tempo_2 = tempo_2 / 1000.0

    tempo_2 = tempo_2 - tempo_2[0]


    # ========================================================
    # DETREND
    # ========================================================

    x1_detrend = signal.detrend(x1_raw)
    y1_detrend = signal.detrend(y1_raw)
    z1_detrend = signal.detrend(z1_raw)

    x2_detrend = signal.detrend(x2_raw)
    y2_detrend = signal.detrend(y2_raw)
    z2_detrend = signal.detrend(z2_raw)


    # ========================================================
    # INTERPOLAÇÃO PARA 100 Hz OU FREQUÊNCIA ESCOLHIDA
    # ========================================================

    dt_novo = 1.0 / fs_novo

    tempo_1_interp = np.arange(tempo_1[0], tempo_1[-1], dt_novo)
    tempo_2_interp = np.arange(tempo_2[0], tempo_2[-1], dt_novo)

    x1_interp = np.interp(tempo_1_interp, tempo_1, x1_detrend)
    y1_interp = np.interp(tempo_1_interp, tempo_1, y1_detrend)
    z1_interp = np.interp(tempo_1_interp, tempo_1, z1_detrend)

    x2_interp = np.interp(tempo_2_interp, tempo_2, x2_detrend)
    y2_interp = np.interp(tempo_2_interp, tempo_2, y2_detrend)
    z2_interp = np.interp(tempo_2_interp, tempo_2, z2_detrend)


    # ========================================================
    # FILTRAGEM PASSA-BAIXA
    # ========================================================

    nyquist = fs_novo / 2.0

    if freq_corte >= nyquist:
        st.error("A frequência de corte precisa ser menor que a frequência de Nyquist.")
        st.stop()

    b, a = signal.butter(
        int(ordem_filtro),
        freq_corte / nyquist,
        btype="low"
    )

    x1_filt = signal.filtfilt(b, a, x1_interp)
    y1_filt = signal.filtfilt(b, a, y1_interp)
    z1_filt = signal.filtfilt(b, a, z1_interp)

    x2_filt = signal.filtfilt(b, a, x2_interp)
    y2_filt = signal.filtfilt(b, a, y2_interp)
    z2_filt = signal.filtfilt(b, a, z2_interp)


    # ========================================================
    # RESULTANTE
    # ========================================================

    r1_filt = np.sqrt(x1_filt**2 + y1_filt**2 + z1_filt**2)
    r2_filt = np.sqrt(x2_filt**2 + y2_filt**2 + z2_filt**2)


    # ========================================================
    # DATAFRAMES FINAIS
    # ========================================================

    df_1 = pd.DataFrame({
        "tempo": tempo_1_interp,
        "X": x1_filt,
        "Y": y1_filt,
        "Z": z1_filt,
        "R": r1_filt
    })

    df_2 = pd.DataFrame({
        "tempo": tempo_2_interp,
        "X": x2_filt,
        "Y": y2_filt,
        "Z": z2_filt,
        "R": r2_filt
    })

    peakAnkle = np.max(df_1["Y"][0:1500])
    peakLumbar = np.max(df_2["X"][0:1500])
    for index,valor in enumerate(df_1["Y"]):
        if valor == peakAnkle:
            df_1["tempo"] = df_1["tempo"] - df_1["tempo"][index] 
            break
    for index,valor in enumerate(df_2["X"]):
        if valor == peakLumbar:
            df_2["tempo"] = df_2["tempo"] - df_2["tempo"][index] 
            break

    # ========================================================
    # INFORMAÇÕES GERAIS
    # ========================================================

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Amostras arquivo 1", len(df_1))
    col2.metric("Duração arquivo 1", f"{df_1['tempo'].iloc[-1]:.2f} s")
    col3.metric("Amostras arquivo 2", len(df_2))
    col4.metric("Duração arquivo 2", f"{df_2['tempo'].iloc[-1]:.2f} s")


    # ========================================================
    # GRÁFICO
    # ========================================================

    st.subheader("Sinais processados")
    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()

        fig.add_trace(go.Scatter(
                    x=df_1["tempo"],
                    y=abs(df_1["R"]),
                    mode="lines",
                    name=f"Arquivo 1 — {"Y"}"
                )
            )
        fig.update_layout(
            height=650,
            xaxis_title="Tempo (s)",
            yaxis_title="Aceleração processada",
            margin=dict(l=40, r=20, t=40, b=40)
        )
    
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
                    x=df_2["tempo"],
                    y=df_2["X"],
                    mode="lines",
                    name=f"Arquivo 2 — {"X"}"
                )
            )

        fig.update_layout(
            height=650,
            xaxis_title="Tempo (s)",
            yaxis_title="Aceleração processada",
            margin=dict(l=40, r=20, t=40, b=40)
        )
    
        st.plotly_chart(fig, use_container_width=True)


    # ========================================================
    # TABELAS E DOWNLOAD
    # ========================================================

    st.subheader("Dados processados")

    aba1, aba2 = st.tabs(["Arquivo 1", "Arquivo 2"])

    with aba1:
        st.dataframe(df_1, use_container_width=True)

        csv_1 = df_1.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Baixar arquivo 1 processado",
            data=csv_1,
            file_name="arquivo_1_processado.csv",
            mime="text/csv"
        )

    with aba2:
        st.dataframe(df_2, use_container_width=True)

        csv_2 = df_2.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Baixar arquivo 2 processado",
            data=csv_2,
            file_name="arquivo_2_processado.csv",
            mime="text/csv"
        )

else:
    st.warning("Carregue os dois arquivos para iniciar o processamento.")
