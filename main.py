import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.signal import detrend, butter, filtfilt

st.set_page_config(page_title="Sincronização de sinais IMU", layout="wide")

st.title("Sincronização de sinais: tornozelo e coluna/cintura")
st.markdown(
    """
Este app importa dois arquivos de aceleração, aplica **detrend**, filtro passa-baixa de **10 Hz** 
e sincroniza as séries temporais usando:

- **Tornozelo:** pico máximo do eixo **Y**
- **Coluna/Cintura:** pico máximo do eixo **X**

Após a sincronização, esses dois eventos passam a ser o **tempo zero** de cada registro.
"""
)


def read_sensor_file(uploaded_file):
    """Read semicolon/comma/space separated TXT/CSV with time, X, Y, Z columns."""
    if uploaded_file is None:
        return None

    raw = uploaded_file.read()
    uploaded_file.seek(0)

    # Try common separators. The uploaded examples use semicolon.
    text = raw.decode("utf-8", errors="ignore")
    try:
        df = pd.read_csv(io.StringIO(text), sep=r"[;,	]+", engine="python")
    except Exception:
        df = pd.read_csv(io.StringIO(text), sep=r"\s+", engine="python")

    # If parsing failed into one column, try whitespace.
    if df.shape[1] < 4:
        df = pd.read_csv(io.StringIO(text), sep=r"\s+", engine="python")

    if df.shape[1] < 4:
        raise ValueError("O arquivo precisa ter pelo menos 4 colunas: tempo, accX, accY e accZ.")

    df = df.iloc[:, :4].copy()
    df.columns = ["tempo_original", "X_original", "Y_original", "Z_original"]

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna().reset_index(drop=True)

    return df


def estimate_fs(time_s):
    dt = np.diff(time_s)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) == 0:
        return 100.0
    return 1.0 / np.median(dt)


def lowpass_filter(y, fs, cutoff=10.0, order=2):
    nyq = fs / 2.0
    if cutoff >= nyq:
        cutoff = 0.45 * fs
    wn = cutoff / nyq
    b, a = butter(order, wn, btype="low")
    return filtfilt(b, a, y)


def preprocess(df, cutoff=10.0, order=2, time_unit="ms"):
    out = df.copy()

    if time_unit == "ms":
        out["tempo_s"] = out["tempo_original"] / 1000.0
    else:
        out["tempo_s"] = out["tempo_original"]

    fs = estimate_fs(out["tempo_s"].to_numpy())

    for axis in ["X", "Y", "Z"]:
        raw = out[f"{axis}_original"].to_numpy(dtype=float)
        det = detrend(raw, type="linear")
        filt = lowpass_filter(det, fs=fs, cutoff=cutoff, order=order)
        out[f"{axis}_detrend"] = det
        out[f"{axis}_filtrado"] = filt

    out["R_filtrado"] = np.sqrt(
        out["X_filtrado"] ** 2 + out["Y_filtrado"] ** 2 + out["Z_filtrado"] ** 2
    )
    out.attrs["fs"] = fs
    return out


def sync_by_peak(df, axis_col):
    """Return dataframe with synchronized time based on max amplitude of selected filtered axis."""
    out = df.copy()
    idx_peak = int(np.nanargmax(out[axis_col].to_numpy()))
    t0 = float(out.loc[idx_peak, "tempo_s"])
    amp0 = float(out.loc[idx_peak, axis_col])
    out["tempo_sinc"] = out["tempo_s"] - t0
    return out, idx_peak, t0, amp0


def plot_signals(df, selected_axes, title, time_col="tempo_sinc", peak_info=None):
    fig = go.Figure()
    for axis in selected_axes:
        ycol = f"{axis}_filtrado" if axis != "R" else "R_filtrado"
        fig.add_trace(
            go.Scatter(
                x=df[time_col],
                y=df[ycol],
                mode="lines",
                name=axis,
            )
        )

    fig.add_vline(x=0, line_dash="dash", annotation_text="t = 0")

    if peak_info is not None:
        idx_peak, axis_col = peak_info
        fig.add_trace(
            go.Scatter(
                x=[df.loc[idx_peak, time_col]],
                y=[df.loc[idx_peak, axis_col]],
                mode="markers",
                marker=dict(size=10),
                name="Pico de sincronização",
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Tempo sincronizado (s)",
        yaxis_title="Aceleração filtrada",
        height=430,
        legend_title="Eixos",
    )
    return fig


with st.sidebar:
    st.header("Arquivos")
    tornozelo_file = st.file_uploader("Arquivo do tornozelo", type=["txt", "csv"], key="tornozelo")
    coluna_file = st.file_uploader("Arquivo da coluna/cintura", type=["txt", "csv"], key="coluna")

    st.header("Pré-processamento")
    time_unit = st.radio("Unidade da coluna de tempo", ["ms", "s"], index=0, horizontal=True)
    cutoff = st.number_input("Filtro passa-baixa (Hz)", min_value=0.1, max_value=50.0, value=10.0, step=0.5)
    order = st.number_input("Ordem do Butterworth", min_value=1, max_value=8, value=2, step=1)

    st.header("Eixos para plotar")
    axes_options = ["X", "Y", "Z", "R"]
    eixos_tornozelo = st.multiselect("Tornozelo", axes_options, default=["Y"])
    eixos_coluna = st.multiselect("Coluna/Cintura", axes_options, default=["X"])

if tornozelo_file is None or coluna_file is None:
    st.info("Importe os dois arquivos para iniciar a análise.")
    st.stop()

try:
    tornozelo_raw = read_sensor_file(tornozelo_file)
    coluna_raw = read_sensor_file(coluna_file)

    tornozelo = preprocess(tornozelo_raw, cutoff=cutoff, order=int(order), time_unit=time_unit)
    coluna = preprocess(coluna_raw, cutoff=cutoff, order=int(order), time_unit=time_unit)

    # Synchronization rule requested by the user.
    tornozelo, idx_tornozelo, t0_tornozelo, amp_tornozelo = sync_by_peak(tornozelo, "Y_filtrado")
    coluna, idx_coluna, t0_coluna, amp_coluna = sync_by_peak(coluna, "X_filtrado")

except Exception as e:
    st.error(f"Erro ao processar os arquivos: {e}")
    st.stop()

st.subheader("Resumo da sincronização")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Fs tornozelo", f"{tornozelo.attrs['fs']:.2f} Hz")
col2.metric("Pico tornozelo Y", f"{amp_tornozelo:.4f}", f"t0 = {t0_tornozelo:.3f} s")
col3.metric("Fs coluna/cintura", f"{coluna.attrs['fs']:.2f} Hz")
col4.metric("Pico coluna X", f"{amp_coluna:.4f}", f"t0 = {t0_coluna:.3f} s")

st.markdown(
    f"""
**Critério aplicado:**  
- Tornozelo: `tempo_sinc = tempo_s - {t0_tornozelo:.6f}`  
- Coluna/Cintura: `tempo_sinc = tempo_s - {t0_coluna:.6f}`
"""
)

left, right = st.columns(2)
with left:
    st.plotly_chart(
        plot_signals(
            tornozelo,
            eixos_tornozelo,
            "Tornozelo sincronizado pelo pico máximo do eixo Y",
            peak_info=(idx_tornozelo, "Y_filtrado"),
        ),
        use_container_width=True,
    )
with right:
    st.plotly_chart(
        plot_signals(
            coluna,
            eixos_coluna,
            "Coluna/Cintura sincronizada pelo pico máximo do eixo X",
            peak_info=(idx_coluna, "X_filtrado"),
        ),
        use_container_width=True,
    )

st.subheader("Sobreposição dos sinais selecionados")
fig_overlay = go.Figure()
for axis in eixos_tornozelo:
    ycol = f"{axis}_filtrado" if axis != "R" else "R_filtrado"
    fig_overlay.add_trace(go.Scatter(x=tornozelo["tempo_sinc"], y=tornozelo[ycol], mode="lines", name=f"Tornozelo {axis}"))
for axis in eixos_coluna:
    ycol = f"{axis}_filtrado" if axis != "R" else "R_filtrado"
    fig_overlay.add_trace(go.Scatter(x=coluna["tempo_sinc"], y=coluna[ycol], mode="lines", name=f"Coluna/Cintura {axis}"))
fig_overlay.add_vline(x=0, line_dash="dash", annotation_text="tempo zero")
fig_overlay.update_layout(
    xaxis_title="Tempo sincronizado (s)",
    yaxis_title="Aceleração filtrada",
    height=520,
    legend_title="Sinais",
)
st.plotly_chart(fig_overlay, use_container_width=True)

st.subheader("Exportar dados sincronizados")
export_tornozelo = tornozelo[["tempo_original", "tempo_s", "tempo_sinc", "X_filtrado", "Y_filtrado", "Z_filtrado", "R_filtrado"]].copy()
export_coluna = coluna[["tempo_original", "tempo_s", "tempo_sinc", "X_filtrado", "Y_filtrado", "Z_filtrado", "R_filtrado"]].copy()

c1, c2 = st.columns(2)
with c1:
    st.download_button(
        "Baixar tornozelo sincronizado CSV",
        data=export_tornozelo.to_csv(index=False).encode("utf-8"),
        file_name="tornozelo_sincronizado.csv",
        mime="text/csv",
    )
with c2:
    st.download_button(
        "Baixar coluna_cintura sincronizada CSV",
        data=export_coluna.to_csv(index=False).encode("utf-8"),
        file_name="coluna_cintura_sincronizada.csv",
        mime="text/csv",
    )

with st.expander("Ver primeiras linhas dos dados sincronizados"):
    st.write("Tornozelo")
    st.dataframe(export_tornozelo.head(20), use_container_width=True)
    st.write("Coluna/Cintura")
    st.dataframe(export_coluna.head(20), use_container_width=True)
