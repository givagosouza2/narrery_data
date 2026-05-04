import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.signal import detrend, butter, filtfilt


st.set_page_config(page_title="Análise de acelerometria", layout="wide")

st.title("Análise de dois sensores: detrend + filtro passa-baixa")
st.write(
    "Abra dois arquivos TXT/CSV, aplique detrend, filtro passa-baixa de 10 Hz "
    "e escolha quais eixos deseja plotar para cada sensor."
)


def read_sensor_file(uploaded_file):
    """Lê arquivos separados por ;, vírgula, tab ou espaço."""
    if uploaded_file is None:
        return None

    raw = uploaded_file.getvalue().decode("utf-8", errors="ignore")
    df = pd.read_csv(io.StringIO(raw), sep=r"[;,	 ]+", engine="python")

    # Normaliza nomes de colunas
    df.columns = [str(c).strip() for c in df.columns]

    # Mantém apenas colunas numéricas
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(how="all")

    if df.shape[1] < 4:
        raise ValueError("O arquivo precisa ter pelo menos 4 colunas: tempo, Acc X, Acc Y e Acc Z.")

    df = df.iloc[:, :4].copy()
    df.columns = ["tempo_original", "acc_x", "acc_y", "acc_z"]

    # Tempo: se estiver em ms, converte para segundos.
    t = df["tempo_original"].to_numpy(dtype=float)
    dt_median = np.nanmedian(np.diff(t))
    if dt_median > 1:  # típico: 10 ms
        df["tempo_s"] = df["tempo_original"] / 1000.0
    else:
        df["tempo_s"] = df["tempo_original"]

    return df


def estimate_fs(time_s):
    diffs = np.diff(time_s)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if len(diffs) == 0:
        return 100.0
    return 1.0 / np.median(diffs)


def preprocess(df, cutoff_hz=10.0, order=2, apply_abs=False):
    out = df.copy()
    fs = estimate_fs(out["tempo_s"].to_numpy(dtype=float))
    nyq = fs / 2.0

    if cutoff_hz >= nyq:
        raise ValueError(
            f"A frequência de corte ({cutoff_hz} Hz) precisa ser menor que Nyquist ({nyq:.2f} Hz)."
        )

    b, a = butter(order, cutoff_hz / nyq, btype="low")

    for axis in ["acc_x", "acc_y", "acc_z"]:
        y = out[axis].to_numpy(dtype=float)
        y = pd.Series(y).interpolate(limit_direction="both").to_numpy()
        y_det = detrend(y, type="linear")
        y_filt = filtfilt(b, a, y_det)
        if apply_abs:
            y_filt = np.abs(y_filt)
        out[f"{axis}_detrend"] = y_det
        out[f"{axis}_filt"] = y_filt

    out["acc_r_filt"] = np.sqrt(
        out["acc_x_filt"] ** 2 + out["acc_y_filt"] ** 2 + out["acc_z_filt"] ** 2
    )
    return out, fs


def plot_sensor(df, selected_axes, title, y_source):
    fig, ax = plt.subplots(figsize=(12, 4))
    map_cols = {
        "X": f"acc_x_{y_source}",
        "Y": f"acc_y_{y_source}",
        "Z": f"acc_z_{y_source}",
        "R = sqrt(X²+Y²+Z²)": "acc_r_filt" if y_source == "filt" else None,
    }

    for axis in selected_axes:
        col = map_cols.get(axis)
        if col is not None and col in df.columns:
            ax.plot(df["tempo_s"], df[col], label=axis)

    ax.set_title(title)
    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("Aceleração")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    st.pyplot(fig)


with st.sidebar:
    st.header("Arquivos")
    file1 = st.file_uploader("Arquivo 1", type=["txt", "csv"], key="file1")
    nome1 = st.text_input("Nome do arquivo 1", value="Cintura")

    file2 = st.file_uploader("Arquivo 2", type=["txt", "csv"], key="file2")
    nome2 = st.text_input("Nome do arquivo 2", value="Tornozelo")

    st.header("Pré-processamento")
    cutoff = st.number_input("Filtro passa-baixa (Hz)", min_value=0.1, max_value=49.0, value=10.0, step=0.5)
    order = st.selectbox("Ordem do filtro Butterworth", [2, 4, 6], index=0)
    apply_abs = st.checkbox("Plotar valores absolutos após filtro", value=False)
    y_source_label = st.radio("Sinal para plotagem", ["Filtrado", "Apenas detrend"], index=0)
    y_source = "filt" if y_source_label == "Filtrado" else "detrend"

    st.header("Eixos")
    axes1 = st.multiselect(f"Eixos para {nome1}", ["X", "Y", "Z", "R = sqrt(X²+Y²+Z²)"], default=["X"])
    axes2 = st.multiselect(f"Eixos para {nome2}", ["X", "Y", "Z", "R = sqrt(X²+Y²+Z²)"], default=["Y"])

if file1 is None or file2 is None:
    st.info("Envie os dois arquivos para iniciar a análise.")
    st.stop()

try:
    df1 = read_sensor_file(file1)
    df2 = read_sensor_file(file2)

    proc1, fs1 = preprocess(df1, cutoff_hz=cutoff, order=order, apply_abs=apply_abs)
    proc2, fs2 = preprocess(df2, cutoff_hz=cutoff, order=order, apply_abs=apply_abs)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"{nome1}: amostras", len(proc1))
    c2.metric(f"{nome1}: Fs estimada", f"{fs1:.1f} Hz")
    c3.metric(f"{nome2}: amostras", len(proc2))
    c4.metric(f"{nome2}: Fs estimada", f"{fs2:.1f} Hz")

    tab1, tab2, tab3 = st.tabs(["Gráficos", "Dados processados", "Exportar"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            plot_sensor(proc1, axes1, nome1, y_source)
        with col2:
            plot_sensor(proc2, axes2, nome2, y_source)

        st.subheader("Sobreposição opcional")
        overlay = st.checkbox("Mostrar os dois sensores no mesmo gráfico")
        if overlay:
            fig, ax = plt.subplots(figsize=(14, 5))
            for axis in axes1:
                col = {"X": f"acc_x_{y_source}", "Y": f"acc_y_{y_source}", "Z": f"acc_z_{y_source}", "R = sqrt(X²+Y²+Z²)": "acc_r_filt"}.get(axis)
                if col and col in proc1.columns:
                    ax.plot(proc1["tempo_s"], proc1[col], label=f"{nome1} {axis}")
            for axis in axes2:
                col = {"X": f"acc_x_{y_source}", "Y": f"acc_y_{y_source}", "Z": f"acc_z_{y_source}", "R = sqrt(X²+Y²+Z²)": "acc_r_filt"}.get(axis)
                if col and col in proc2.columns:
                    ax.plot(proc2["tempo_s"], proc2[col], label=f"{nome2} {axis}")
            ax.set_xlabel("Tempo (s)")
            ax.set_ylabel("Aceleração")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")
            st.pyplot(fig)

    with tab2:
        st.write(f"### {nome1}")
        st.dataframe(proc1.head(1000), use_container_width=True)
        st.write(f"### {nome2}")
        st.dataframe(proc2.head(1000), use_container_width=True)

    with tab3:
        csv1 = proc1.to_csv(index=False).encode("utf-8")
        csv2 = proc2.to_csv(index=False).encode("utf-8")
        st.download_button(f"Baixar {nome1} processado CSV", data=csv1, file_name=f"{nome1}_processado.csv", mime="text/csv")
        st.download_button(f"Baixar {nome2} processado CSV", data=csv2, file_name=f"{nome2}_processado.csv", mime="text/csv")

except Exception as e:
    st.error(f"Erro na análise: {e}")
