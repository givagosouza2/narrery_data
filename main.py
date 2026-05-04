
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy import signal


st.set_page_config(page_title="Sincronização por desvio da baseline", layout="wide")


# -----------------------------
# Leitura e pré-processamento
# -----------------------------
def ler_arquivo(uploaded_file):
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

    tempo = out["tempo_original"].to_numpy(dtype=float)

    # Se o intervalo mediano for > 1, provavelmente está em milissegundos
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


# -----------------------------
# Sincronização por baseline
# -----------------------------
def detectar_saida_baseline(
    df_proc,
    eixo,
    duracao_baseline_s=0.2,
    n_dp=5.0,
    procurar_apos_baseline=True,
    persistencia_amostras=1
):
    """
    Calcula baseline nos primeiros duracao_baseline_s segundos.
    Detecta o primeiro momento em que o sinal ultrapassa:
    média + n_dp*DP ou média - n_dp*DP.

    Se persistencia_amostras > 1, exige que a condição ocorra por N amostras consecutivas.
    """
    tempo = df_proc["tempo_s"].to_numpy(dtype=float)
    sinal = df_proc[eixo].to_numpy(dtype=float)

    t_ini = tempo[0]
    t_fim_base = t_ini + duracao_baseline_s

    mask_base = (tempo >= t_ini) & (tempo <= t_fim_base)

    if np.sum(mask_base) < 3:
        raise ValueError(
            f"Baseline com poucas amostras no eixo {eixo}. "
            "Aumente a duração da baseline ou verifique a frequência de amostragem."
        )

    baseline = sinal[mask_base]
    media = np.mean(baseline)
    dp = np.std(baseline, ddof=1)

    limite_sup = media + n_dp * dp
    limite_inf = media - n_dp * dp

    if procurar_apos_baseline:
        mask_busca = tempo > t_fim_base
    else:
        mask_busca = tempo >= t_ini

    indices_busca = np.where(mask_busca)[0]

    if len(indices_busca) == 0:
        raise ValueError("Não há dados após o período de baseline.")

    acima_ou_abaixo = (sinal > limite_sup) | (sinal < limite_inf)

    idx_evento = None

    if persistencia_amostras <= 1:
        candidatos = indices_busca[acima_ou_abaixo[indices_busca]]
        if len(candidatos) > 0:
            idx_evento = candidatos[0]
    else:
        for idx in indices_busca:
            fim = idx + persistencia_amostras
            if fim <= len(sinal):
                if np.all(acima_ou_abaixo[idx:fim]):
                    idx_evento = idx
                    break

    if idx_evento is None:
        raise ValueError(
            f"Nenhum deslocamento além de ±{n_dp:.1f} DP foi detectado no eixo {eixo}."
        )

    tempo_evento = tempo[idx_evento]
    valor_evento = sinal[idx_evento]

    direcao = "acima" if valor_evento > limite_sup else "abaixo"

    info = {
        "idx_evento": idx_evento,
        "tempo_evento": tempo_evento,
        "valor_evento": valor_evento,
        "media_baseline": media,
        "dp_baseline": dp,
        "limite_superior": limite_sup,
        "limite_inferior": limite_inf,
        "t_ini_baseline": t_ini,
        "t_fim_baseline": t_fim_base,
        "direcao": direcao,
    }

    return info


def aplicar_sincronizacao(df_proc, tempo_evento):
    out = df_proc.copy()
    out["tempo_sync"] = out["tempo_s"] - tempo_evento
    return out


# -----------------------------
# Plotagens
# -----------------------------
def adicionar_linhas_baseline(fig, info, prefixo, tempo_col="tempo_s"):
    # Linhas horizontais de média e limites
    fig.add_hline(
        y=info["media_baseline"],
        line_dash="dot",
        annotation_text=f"{prefixo} média base",
        annotation_position="top left"
    )
    fig.add_hline(
        y=info["limite_superior"],
        line_dash="dash",
        annotation_text=f"{prefixo} +2DP",
        annotation_position="top left"
    )
    fig.add_hline(
        y=info["limite_inferior"],
        line_dash="dash",
        annotation_text=f"{prefixo} -2DP",
        annotation_position="bottom left"
    )


def plotar_sinais(
    df_tornozelo,
    df_coluna,
    eixos_tornozelo,
    eixos_coluna,
    usar_tempo_sync=True
):
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

    if usar_tempo_sync:
        fig.add_vline(x=0, line_dash="dash", annotation_text="tempo zero")

    fig.update_layout(
        height=650,
        xaxis_title="Tempo sincronizado (s)" if usar_tempo_sync else "Tempo original (s)",
        yaxis_title="Aceleração após detrend + filtro",
        legend_title="Sinais",
        margin=dict(l=40, r=20, t=40, b=40)
    )

    return fig


def plotar_deteccao(df_proc, eixo, info, titulo):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_proc["tempo_s"],
        y=df_proc[eixo],
        mode="lines",
        name=eixo
    ))

    fig.add_vrect(
        x0=info["t_ini_baseline"],
        x1=info["t_fim_baseline"],
        opacity=0.15,
        line_width=0,
        annotation_text="baseline",
        annotation_position="top left"
    )

    fig.add_hline(y=info["media_baseline"], line_dash="dot", annotation_text="média baseline")
    fig.add_hline(y=info["limite_superior"], line_dash="dash", annotation_text="+2 DP")
    fig.add_hline(y=info["limite_inferior"], line_dash="dash", annotation_text="-2 DP")

    fig.add_vline(
        x=info["tempo_evento"],
        line_dash="dash",
        annotation_text=f"evento: {info['tempo_evento']:.3f}s"
    )

    fig.add_trace(go.Scatter(
        x=[info["tempo_evento"]],
        y=[info["valor_evento"]],
        mode="markers",
        marker=dict(size=10),
        name="evento"
    ))

    fig.update_layout(
        title=titulo,
        height=420,
        xaxis_title="Tempo original (s)",
        yaxis_title="Aceleração após detrend + filtro",
        margin=dict(l=40, r=20, t=50, b=40)
    )

    return fig


# -----------------------------
# Interface
# -----------------------------
st.title("Sincronização por deslocamento em relação à baseline")

st.markdown(
    """
    Este app abre dois arquivos, aplica **detrend**, filtro passa-baixa e sincroniza os sinais
    pelo primeiro deslocamento além de **média ± 2 desvios-padrões** da baseline inicial.

    Critérios usados:

    - **Tornozelo:** eixo **Y**
    - **Coluna/Cintura:** eixo **X**
    - **Baseline:** primeiros **200 ms** por padrão
    """
)

with st.sidebar:
    st.header("Arquivos")
    arq_tornozelo = st.file_uploader("Arquivo do tornozelo", type=["txt", "csv"], key="tornozelo")
    arq_coluna = st.file_uploader("Arquivo da coluna/cintura", type=["txt", "csv"], key="coluna")

    st.header("Processamento")
    cutoff = st.number_input("Filtro passa-baixa (Hz)", min_value=0.1, max_value=50.0, value=10.0, step=0.5)
    ordem = st.number_input("Ordem do filtro Butterworth", min_value=1, max_value=8, value=2, step=1)

    st.header("Sincronização")
    duracao_baseline_ms = st.number_input(
        "Duração da baseline (ms)",
        min_value=50,
        max_value=5000,
        value=200,
        step=50
    )
    n_dp = st.number_input(
        "Limiar em desvios-padrões",
        min_value=0.5,
        max_value=10.0,
        value=2.0,
        step=0.5
    )
    persistencia = st.number_input(
        "Persistência mínima (amostras)",
        min_value=1,
        max_value=50,
        value=1,
        step=1,
        help="Use valores maiores, como 3 ou 5, para evitar detecção por ruído."
    )

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

        duracao_baseline_s = duracao_baseline_ms / 1000.0

        info_t = detectar_saida_baseline(
            df_t_proc,
            eixo="Y",
            duracao_baseline_s=duracao_baseline_s,
            n_dp=n_dp,
            procurar_apos_baseline=True,
            persistencia_amostras=int(persistencia)
        )

        info_c = detectar_saida_baseline(
            df_c_proc,
            eixo="X",
            duracao_baseline_s=duracao_baseline_s,
            n_dp=n_dp,
            procurar_apos_baseline=True,
            persistencia_amostras=int(persistencia)
        )

        df_t_sync = aplicar_sincronizacao(df_t_proc, info_t["tempo_evento"])
        df_c_sync = aplicar_sincronizacao(df_c_proc, info_c["tempo_evento"])

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Fs tornozelo", f"{fs_t:.2f} Hz")
        col2.metric("Evento tornozelo Y", f"{info_t['tempo_evento']:.3f} s")
        col3.metric("Fs coluna/cintura", f"{fs_c:.2f} Hz")
        col4.metric("Evento coluna X", f"{info_c['tempo_evento']:.3f} s")

        st.info(
            f"Tempo zero definido pelo primeiro deslocamento além de média ± {n_dp:.1f} DP "
            f"após baseline de {duracao_baseline_ms} ms. "
            f"Tornozelo Y: {info_t['tempo_evento']:.3f} s ({info_t['direcao']}); "
            f"coluna/cintura X: {info_c['tempo_evento']:.3f} s ({info_c['direcao']})."
        )

        st.subheader("Detecção do evento de sincronização")

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                plotar_deteccao(df_t_proc, "Y", info_t, "Tornozelo - eixo Y"),
                use_container_width=True
            )
        with c2:
            st.plotly_chart(
                plotar_deteccao(df_c_proc, "X", info_c, "Coluna/Cintura - eixo X"),
                use_container_width=True
            )

        st.subheader("Sinais sincronizados")

        fig = plotar_sinais(
            df_t_sync,
            df_c_sync,
            eixos_tornozelo,
            eixos_coluna,
            usar_tempo_sync=usar_tempo_sync
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Zoom ao redor do tempo zero")
        janela_zoom = st.slider("Janela de zoom ao redor do tempo zero (s)", 0.5, 10.0, 3.0, 0.5)

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

        st.subheader("Parâmetros da baseline")

        resumo = pd.DataFrame([
            {
                "Sensor": "Tornozelo",
                "Eixo": "Y",
                "Média baseline": info_t["media_baseline"],
                "DP baseline": info_t["dp_baseline"],
                "Limite inferior": info_t["limite_inferior"],
                "Limite superior": info_t["limite_superior"],
                "Tempo do evento (s)": info_t["tempo_evento"],
                "Valor no evento": info_t["valor_evento"],
                "Direção": info_t["direcao"],
            },
            {
                "Sensor": "Coluna/Cintura",
                "Eixo": "X",
                "Média baseline": info_c["media_baseline"],
                "DP baseline": info_c["dp_baseline"],
                "Limite inferior": info_c["limite_inferior"],
                "Limite superior": info_c["limite_superior"],
                "Tempo do evento (s)": info_c["tempo_evento"],
                "Valor no evento": info_c["valor_evento"],
                "Direção": info_c["direcao"],
            },
        ])

        st.dataframe(resumo, use_container_width=True)

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
            file_name="sinais_sincronizados_baseline.csv",
            mime="text/csv"
        )

        csv_resumo = resumo.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Baixar resumo da sincronização",
            data=csv_resumo,
            file_name="resumo_sincronizacao_baseline.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Erro ao processar os arquivos: {e}")

else:
    st.warning("Envie os dois arquivos para iniciar a análise.")
