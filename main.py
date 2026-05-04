import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.signal import detrend, butter, filtfilt
from scipy.io import savemat

st.set_page_config(page_title="APA - Análise de Aceleração", layout="wide")

# =========================================================
# Funções auxiliares
# =========================================================

def read_txt_uploaded(uploaded_file):
    """Lê arquivo TXT/CSV com pelo menos 4 colunas numéricas: tempo, accX, accY, accZ."""
    if uploaded_file is None:
        return None

    raw = uploaded_file.read()
    uploaded_file.seek(0)

    # Tenta ler com separadores comuns: vírgula, ponto e vírgula, tabulação ou espaços.
    try:
        df = pd.read_csv(
            io.BytesIO(raw),
            sep=r"[,;\t\s]+",
            engine="python",
            header=None,
            comment="#",
        )
    except Exception as e:
        raise ValueError(f"Não foi possível ler o arquivo: {e}")

    # Remove colunas completamente vazias e tenta converter para numérico.
    df = df.dropna(axis=1, how="all")
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(how="all")

    # Se a primeira linha era cabeçalho, ela vira NaN e será removida.
    df = df.dropna()

    if df.shape[1] < 4:
        raise ValueError("O arquivo precisa ter pelo menos 4 colunas numéricas: tempo, accX, accY e accZ.")

    return df.iloc[:, :4].copy()


def butter_lowpass_filter(x, cutoff_hz, fs_hz=100.0, order=2):
    nyq = fs_hz / 2.0
    wn = cutoff_hz / nyq
    b, a = butter(order, wn, btype="low")
    return filtfilt(b, a, x)


def moving_average(x, window=50):
    return pd.Series(x).rolling(window=window, center=False, min_periods=1).mean().to_numpy()


def process_sensor(df, sensor_type, fs_hz=100.0):
    """
    Reproduz a lógica do MATLAB:
    - tempo em ms convertido para s
    - aceleração dividida por 9.81
    - detrend
    - cintura accY invertida
    - filtros diferentes por eixo
    """
    tempo = df.iloc[:, 0].to_numpy(dtype=float) / 1000.0
    accX = detrend(df.iloc[:, 1].to_numpy(dtype=float) / 9.81)
    accY = detrend(df.iloc[:, 2].to_numpy(dtype=float) / 9.81)
    accZ = detrend(df.iloc[:, 3].to_numpy(dtype=float) / 9.81)

    if sensor_type == "cintura":
        accY = -1.0 * accY

    accX_fit = moving_average(accX, 50)
    accY_fit = moving_average(accY, 50)
    accR = np.sqrt(accX**2 + accY**2 + accZ**2)

    if sensor_type == "tornozelo":
        # MATLAB: tornozelo_accY filtrado em 20 Hz; tornozelo_accX em 3 Hz
        accY = butter_lowpass_filter(accY, cutoff_hz=20, fs_hz=fs_hz, order=2)
        accX = butter_lowpass_filter(accX, cutoff_hz=3, fs_hz=fs_hz, order=2)
        idx_zero = int(np.argmax(accY))
        tempo_sync = tempo - tempo[idx_zero]
    else:
        # MATLAB: cintura_accX filtrado em 20 Hz; cintura_accY em 3 Hz
        accX = butter_lowpass_filter(accX, cutoff_hz=20, fs_hz=fs_hz, order=2)
        accY = butter_lowpass_filter(accY, cutoff_hz=3, fs_hz=fs_hz, order=2)
        idx_zero = int(np.argmax(accX))
        tempo_sync = tempo - tempo[idx_zero]

    return {
        "tempo_original": tempo,
        "tempo": tempo_sync,
        "accX": accX,
        "accY": accY,
        "accZ": accZ,
        "accR": accR,
        "accX_fit": accX_fit,
        "accY_fit": accY_fit,
        "idx_zero": idx_zero,
    }


def safe_slice(center_idx, before, after, n):
    start = max(0, center_idx - before)
    end = min(n, center_idx + after + 1)
    return slice(start, end)


def nearest_index(time_array, selected_time, start_idx=0):
    valid = np.arange(start_idx, len(time_array))
    if len(valid) == 0:
        return start_idx
    return int(valid[np.argmin(np.abs(time_array[valid] - selected_time))])


def line_plot(x, y, title, y_label="Aceleração (g)", vline=None, y_range=None, y2=None, y2_name="Média móvel"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Sinal", line=dict(color="black")))
    if y2 is not None:
        fig.add_trace(go.Scatter(x=x, y=y2, mode="lines", name=y2_name, line=dict(color="red")))
    if vline is not None:
        fig.add_vline(x=vline, line_dash="dash", line_color="red")
    fig.update_layout(
        title=title,
        xaxis_title="Tempo (s)",
        yaxis_title=y_label,
        height=330,
        margin=dict(l=30, r=20, t=50, b=30),
        font=dict(family="Times New Roman", size=16),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    if y_range is not None:
        fig.update_yaxes(range=y_range)
    return fig


# =========================================================
# Interface
# =========================================================

st.title("APA - Conversão MATLAB para Python/Streamlit")
st.caption("Importe os sinais do tornozelo e da cintura, sincronize, visualize e salve janelas de análise.")

with st.sidebar:
    st.header("1. Importar dados")
    fs_hz = st.number_input("Frequência de amostragem assumida (Hz)", min_value=1.0, max_value=1000.0, value=100.0, step=1.0)
    tornozelo_file = st.file_uploader("Arquivo do tornozelo (.txt/.csv)", type=["txt", "csv"], key="tornozelo")
    cintura_file = st.file_uploader("Arquivo da cintura (.txt/.csv)", type=["txt", "csv"], key="cintura")

    processar = st.button("Processar arquivos", type="primary")

if processar:
    try:
        tornozelo_df = read_txt_uploaded(tornozelo_file)
        cintura_df = read_txt_uploaded(cintura_file)
        if tornozelo_df is None or cintura_df is None:
            st.warning("Importe os dois arquivos antes de processar.")
        else:
            st.session_state["tornozelo"] = process_sensor(tornozelo_df, "tornozelo", fs_hz)
            st.session_state["cintura"] = process_sensor(cintura_df, "cintura", fs_hz)
            st.session_state["janelas"] = []
            st.success("Arquivos processados e sincronizados automaticamente pelos picos iniciais.")
    except Exception as e:
        st.error(str(e))

if "tornozelo" not in st.session_state or "cintura" not in st.session_state:
    st.info("Carregue os dois arquivos na barra lateral para iniciar.")
    st.stop()

t = st.session_state["tornozelo"]
c = st.session_state["cintura"]

st.header("2. Sincronização")
col1, col2 = st.columns(2)

# Índices automáticos
idx_to_auto = t["idx_zero"]
idx_ci_auto = c["idx_zero"]

with col1:
    sl = safe_slice(idx_to_auto, 100, 100, len(t["tempo"]))
    st.plotly_chart(
        line_plot(t["tempo"][sl], t["accY"][sl], "Tornozelo Y - sincronização", vline=0, y_range=[-3, 3]),
        use_container_width=True,
    )
with col2:
    sl = safe_slice(idx_ci_auto, 100, 100, len(c["tempo"]))
    st.plotly_chart(
        line_plot(c["tempo"][sl], c["accX"][sl], "Cintura X - sincronização", vline=0, y_range=[-3, 3]),
        use_container_width=True,
    )

with st.expander("Sincronização manual"):
    st.write("No MATLAB a seleção era feita por clique com `ginput`. Aqui, selecione o novo tempo zero usando campos numéricos.")
    colm1, colm2, colm3 = st.columns([1, 1, 1])
    with colm1:
        novo_zero_tornozelo = st.number_input("Novo zero tornozelo (s)", value=0.0, step=0.01, format="%.4f")
    with colm2:
        novo_zero_cintura = st.number_input("Novo zero cintura (s)", value=0.0, step=0.01, format="%.4f")
    with colm3:
        if st.button("Aplicar sincronização manual"):
            idx_to = nearest_index(t["tempo"], novo_zero_tornozelo, max(0, idx_to_auto - 100))
            idx_ci = nearest_index(c["tempo"], novo_zero_cintura, max(0, idx_ci_auto - 100))
            t["tempo"] = t["tempo"] - t["tempo"][idx_to]
            c["tempo"] = c["tempo"] - c["tempo"][idx_ci]
            t["idx_zero"] = idx_to
            c["idx_zero"] = idx_ci
            st.success("Sincronização manual aplicada.")
            st.rerun()

st.header("3. Ver registros")
start_to = min(len(t["tempo"]) - 1, t["idx_zero"] + 100)
start_ci = min(len(c["tempo"]) - 1, c["idx_zero"] + 100)

col3, col4 = st.columns(2)
with col3:
    st.plotly_chart(
        line_plot(
            t["tempo"][start_to:],
            np.abs(t["accX"])[start_to:],
            "Tornozelo |accX|",
            y_range=[-0.8, 0.8],
            y2=t["accX_fit"][start_to:],
        ),
        use_container_width=True,
    )
with col4:
    st.plotly_chart(
        line_plot(
            c["tempo"][start_ci:],
            c["accY"][start_ci:],
            "Cintura accY",
            y_range=[-0.25, 0.25],
            y2=c["accY_fit"][start_ci:],
        ),
        use_container_width=True,
    )

st.header("4. Selecionar e salvar janela")
min_time = float(max(t["tempo"][start_to], c["tempo"][start_ci]))
max_time = float(min(t["tempo"][-1], c["tempo"][-1]))
selected_time = st.slider("Tempo central da janela (s)", min_value=min_time, max_value=max_time, value=min_time, step=0.01)

j_to = nearest_index(t["tempo"], selected_time, start_to)
j_ci = nearest_index(c["tempo"], selected_time, start_ci)
win_to = safe_slice(j_to, 200, 100, len(t["tempo"]))
win_ci = safe_slice(j_ci, 200, 100, len(c["tempo"]))

col5, col6 = st.columns(2)
with col5:
    st.plotly_chart(
        line_plot(
            t["tempo"][win_to],
            np.abs(t["accX"])[win_to],
            "Janela - Tornozelo |accX|",
            vline=t["tempo"][j_to],
            y_range=[-0.8, 0.8],
        ),
        use_container_width=True,
    )
with col6:
    st.plotly_chart(
        line_plot(
            c["tempo"][win_ci],
            c["accY"][win_ci],
            "Janela - Cintura accY",
            vline=c["tempo"][j_ci],
            y_range=[-0.25, 0.25],
        ),
        use_container_width=True,
    )

if st.button("Salvar janela na sessão"):
    janela = pd.DataFrame({
        "tornozelo_tempo": t["tempo"][win_to],
        "tornozelo_abs_accX": np.abs(t["accX"])[win_to],
        "cintura_tempo": c["tempo"][win_ci],
        "cintura_accY": c["accY"][win_ci],
    })
    st.session_state["janelas"].append(janela)
    st.success(f"Janela salva. Total de janelas na sessão: {len(st.session_state['janelas'])}")

if st.session_state.get("janelas"):
    st.subheader("Janelas salvas")
    janela_idx = st.selectbox("Selecionar janela", range(1, len(st.session_state["janelas"]) + 1))
    janela_df = st.session_state["janelas"][janela_idx - 1]
    st.dataframe(janela_df, use_container_width=True)

    csv_bytes = janela_df.to_csv(index=False).encode("utf-8")
    st.download_button("Baixar janela selecionada em CSV", csv_bytes, file_name=f"janela_{janela_idx}.csv", mime="text/csv")

    # Exporta todas as janelas para um arquivo .mat em memória.
    mat_dict = {f"janela_{i+1}": df.to_numpy() for i, df in enumerate(st.session_state["janelas"])}
    mat_buffer = io.BytesIO()
    savemat(mat_buffer, mat_dict)
    st.download_button(
        "Baixar todas as janelas em MAT",
        mat_buffer.getvalue(),
        file_name="resultados.mat",
        mime="application/octet-stream",
    )
