import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy import signal


st.set_page_config(page_title="Sincronização de sinais IMU", layout="wide")


# -----------------------------
# Funções auxiliares
# -----------------------------
def ler_arquivo(uploaded_file):
    """
    Lê arquivo TXT/CSV com separador ;, vírgula, tabulação ou espaço.
    Espera colunas semelhantes a:
    DURACAO;ACC EIXO X;ACC EIXO Y;ACC EIXO Z
    """
    if uploaded_file is None:
        return None

    raw = uploaded_file.read()
    text = raw.decode("utf-8", errors="ignore")

    df = pd.read_csv(
        io.StringIO(text),
        sep=r"[;\t,]+",
        engine="python"
    )

    df.columns = [c.strip().upper() for c in df.columns]

    # Tenta identificar as colunas
    tempo_col = None
    x_col = None
    y_col = None
    z_col = None

    for c in df.columns:
        if "DUR" in c or "TEMPO" in c or c == "TIME":
            tempo_col = c
        elif "X" in c:
            x_col = c
        elif "Y" in c:
            y_col = c
        elif "Z" in c:
            z_col = c

    if tempo_col is None or x_col is None or y_col is None or z_col is None:
        # fallback: usa as quatro primeiras colunas numéricas
        df_num = df.apply(pd.to_numeric, errors="coerce")
        numeric_cols = df_num.columns[df_num.notna().sum() > 0].tolist()
        if len(numeric_cols) < 4:
            raise ValueError("O arquivo precisa ter pelo menos 4 colunas numéricas: tempo, X, Y e Z.")
        tempo_col, x_col, y_col, z_col = numeric_cols[:4]

    out = pd.DataFrame({
        "tempo_original": pd.to_numeric(df[tempo_col], errors="coerce"),
        "X_raw": pd.to_numeric(df[x_col], errors="coerce"),
        "Y_raw": pd.to_numeric(df[y_col], errors="coerce"),
        "Z_raw": pd.to_numeric(df[z_col], errors="coerce"),
    }).dropna()

    # Converte tempo para segundos se parecer estar em milissegundos
    tempo = out["tempo_original"].to_numpy(dtype=float)
    if np.nanmedian(np.diff(tempo)) > 1:
        tempo = tempo / 1000.0

    out["tempo_s"] = tempo
    return out.reset_index(drop=True)


def estimar_fs(tempo):
    dt = np.diff(tempo)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) == 0:
        return 100.0
    return 1.0 / np.median(dt)


def preprocessar(df, cutoff_hz=10.0, ordem=2):
    """
    Aplica detrend e filtro passa-baixa em X, Y e Z.
    Também calcula a resultante R filtrada.
    """
    tempo = df["tempo_s"].to_numpy(dtype=float)
    fs = estimar_fs(tempo)

    nyq = fs / 2.0
    cutoff = min(cutoff_hz, nyq * 0.95)

    b, a = signal.butter(ordem, cutoff / nyq, btype="low")

    proc = pd.DataFrame()
    proc["tempo_s"] = tempo

    for eixo in ["X", "Y", "Z"]:
        raw = df[f"{eixo}_raw"].to_numpy(dtype=float)
        detr = signal.detrend(raw)
        filt = signal.filtfilt(b, a, detr)
        proc[f"{eixo}"] = filt

    proc["R"] = np.sqrt(proc["X"]**2 + proc["Y"]**2 + proc["Z"]**2)
    return proc, fs


def localizar_pico_ate_10s(df_proc, eixo, janela_s=10.0):
    """
    Procura o pico máximo do eixo escolhido apenas nos primeiros 10 segundos
    contados a partir do início de cada arquivo.
    """
    tempo = df_proc["tempo_s"].to_numpy(dtype=float)
    sinal = df_proc[eixo].to_numpy(dtype=float)

    t0 = tempo[0]
    mascara = (tempo >= t0) & (tempo <= t0 + janela_s)

    if not np.any(mascara):
        raise ValueError("Não há dados dentro dos primeiros 10 segundos.")

    indices = np.where(mascara)[0]
    idx_local = np.argmax(sinal[mascara])
    idx_global = indices[idx_local]

    tempo_pico = tempo[idx_global]
    valor_pico = sinal[idx_global]

    return idx_global, tempo_pico, valor_pico


def aplicar_sincronizacao(df_proc, tempo_pico):
    out = df_proc.copy()
    out["tempo_sync"] = out["tempo_s"] - tempo_pico
    return out


def plotar_sinais(df_tornozelo, df_coluna, eixos_tornozelo, eixos_coluna, usar_tempo_sync=True):
    fig = go.Figure()

    tempo_col = "tempo_sync" if usar_tempo_sync else "tempo_s"

    for eixo in eixos_tornozelo:
        fig.add_trace(go.Scatter(
            x=df_tornozelo[tempo_col],
            y=df_tornozelo[eixo],
            mode="lines",
            name=f"Tornozelo {eixo}"
        ))

    for eixo in eixos_coluna:
        fig.add_trace(go.Scatter(
            x=df_coluna[tempo_col],
            y=df_coluna[eixo],
            mode="lines",
            name=f"Coluna/Cintura {eixo}"
        ))

    fig.add_vline(x=0, line_dash="dash")

    fig.update_layout(
        height=650,
        xaxis_title="Tempo sincronizado (s)" if usar_tempo_sync else "Tempo original (s)",
        yaxis_title="Aceleração após detrend + filtro",
        legend_title="Sinais",
        margin=dict(l=40, r=20, t=40, b=40)
    )

    return fig


# -----------------------------
# Interface
# -----------------------------
st.title("Sincronização de sinais IMU")
st.markdown(
    """
    Este app abre dois arquivos, aplica **detrend**, filtro passa-baixa de **10 Hz** 
    e sincroniza os sinais usando o pico nos **primeiros 10 segundos**:

    - **Tornozelo:** pico máximo no eixo **Y**
    - **Coluna/Cintura:** pico máximo no eixo **X**
    """
)

with st.sidebar:
    st.header("Arquivos")
    arq_tornozelo = st.file_uploader("Arquivo do tornozelo", type=["txt", "csv"], key="tornozelo")
    arq_coluna = st.file_uploader("Arquivo da coluna/cintura", type=["txt", "csv"], key="coluna")

    st.header("Processamento")
    cutoff = st.number_input("Filtro passa-baixa (Hz)", min_value=0.1, max_value=50.0, value=10.0, step=0.5)
    ordem = st.number_input("Ordem do filtro Butterworth", min_value=1, max_value=8, value=2, step=1)
    janela_pico = st.number_input("Procurar pico até (s)", min_value=1.0, max_value=60.0, value=10.0, step=1.0)

    st.header("Eixos para plotar")
    eixos_tornozelo = st.multiselect(
        "Tornozelo",
        ["X", "Y", "Z", "R"],
        default=["Y"]
    )
    eixos_coluna = st.multiselect(
        "Coluna/Cintura",
        ["X", "Y", "Z", "R"],
        default=["X"]
    )

    usar_tempo_sync = st.checkbox("Usar tempo sincronizado", value=True)


if arq_tornozelo is not None and arq_coluna is not None:
    try:
        df_t_raw = ler_arquivo(arq_tornozelo)
        df_c_raw = ler_arquivo(arq_coluna)

        df_t_proc, fs_t = preprocessar(df_t_raw, cutoff_hz=cutoff, ordem=int(ordem))
        df_c_proc, fs_c = preprocessar(df_c_raw, cutoff_hz=cutoff, ordem=int(ordem))

        idx_t, tempo_pico_t, valor_pico_t = localizar_pico_ate_10s(
            df_t_proc, eixo="Y", janela_s=janela_pico
        )
        idx_c, tempo_pico_c, valor_pico_c = localizar_pico_ate_10s(
            df_c_proc, eixo="X", janela_s=janela_pico
        )

        df_t_sync = aplicar_sincronizacao(df_t_proc, tempo_pico_t)
        df_c_sync = aplicar_sincronizacao(df_c_proc, tempo_pico_c)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Fs tornozelo", f"{fs_t:.2f} Hz")
        col2.metric("Pico tornozelo Y", f"{tempo_pico_t:.3f} s")
        col3.metric("Fs coluna/cintura", f"{fs_c:.2f} Hz")
        col4.metric("Pico coluna X", f"{tempo_pico_c:.3f} s")

        st.info(
            f"Tempo zero definido pelo pico nos primeiros {janela_pico:.1f} s: "
            f"tornozelo Y = {tempo_pico_t:.3f} s; "
            f"coluna/cintura X = {tempo_pico_c:.3f} s."
        )

        fig = plotar_sinais(
            df_t_sync,
            df_c_sync,
            eixos_tornozelo,
            eixos_coluna,
            usar_tempo_sync=usar_tempo_sync
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Visualização ao redor do tempo zero")
        janela_zoom = st.slider("Janela de zoom ao redor do tempo zero (s)", 1.0, 10.0, 3.0, 0.5)

        df_t_zoom = df_t_sync[
            (df_t_sync["tempo_sync"] >= -janela_zoom) &
            (df_t_sync["tempo_sync"] <= janela_zoom)
        ]
        df_c_zoom = df_c_sync[
            (df_c_sync["tempo_sync"] >= -janela_zoom) &
            (df_c_sync["tempo_sync"] <= janela_zoom)
        ]

        fig_zoom = plotar_sinais(
            df_t_zoom,
            df_c_zoom,
            eixos_tornozelo,
            eixos_coluna,
            usar_tempo_sync=True
        )
        fig_zoom.update_layout(height=450)
        st.plotly_chart(fig_zoom, use_container_width=True)

        st.subheader("Exportar dados sincronizados")

        df_t_export = df_t_sync.copy()
        df_t_export.insert(0, "sensor", "tornozelo")

        df_c_export = df_c_sync.copy()
        df_c_export.insert(0, "sensor", "coluna_cintura")

        df_export = pd.concat([df_t_export, df_c_export], ignore_index=True)

        csv = df_export.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Baixar CSV sincronizado",
            data=csv,
            file_name="sinais_sincronizados.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Erro ao processar os arquivos: {e}")

else:
    st.warning("Envie os dois arquivos para iniciar a análise.")
