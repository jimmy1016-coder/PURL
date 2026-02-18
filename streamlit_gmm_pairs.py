"""
GMM / vMF Mixture 기반 Noisy Pair 탐지 Streamlit 앱.

각 화자 내 발화 쌍에 대해:
  - GMM: cosine similarity 분포에 2-component Gaussian Mixture를 fitting
  - vMF: L2-normalized embedding에 2-component von Mises-Fisher Mixture를 fitting
하여 clean pair / noisy pair을 분류하고 시각화합니다.
쌍을 클릭하면 두 음성을 비교 청취할 수 있습니다.

실행 방법:
    streamlit run streamlit_gmm_pairs.py
"""

import sys
from pathlib import Path

# Add src directory to Python path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import itertools

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import torch
import torch.nn.functional as F
import torchaudio
from sklearn.mixture import GaussianMixture
from scipy.special import ive as _ive
from scipy.stats import norm

from data.augmentations import SpecNormalization
from data.data_utils import crop_or_pad, load_spk2utt
from data.feature_extractors import SBFbank
from models import ECAPAEncoder


# ========================
# 설정
# ========================
CKPT_PATH = "/home/sooyoung/interspeech/chns/outputs/supcon/checkpoints/epoch=199_step=333400.ckpt"
DATA_DIR = Path("/home/sooyoung/interspeech/dev/aac/")
SPK2UTT_PATH = "/home/sooyoung/interspeech/dev/aac/spk2utt"
SAMPLE_RATE = 16000
SEGMENT_LENGTH = 48000


# ========================
# 모델 / 데이터 로딩
# ========================
@st.cache_resource
def load_model(ckpt_path, device):
    """체크포인트에서 모델을 로드합니다."""
    checkpoint = torch.load(ckpt_path, map_location=device)

    encoder = ECAPAEncoder(
        input_size=80,
        channels=[256, 256, 256, 256, 768],
        kernel_sizes=[5, 3, 3, 3, 1],
        dilations=[1, 2, 3, 4, 1],
        attention_channels=128,
        lin_neurons=192,
        res2net_scale=8,
        se_channels=128,
    )

    state_dict = checkpoint["state_dict"]
    encoder_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("encoder."):
            new_key = k.replace("encoder.", "")
            encoder_state_dict[new_key] = v

    encoder.load_state_dict(encoder_state_dict)
    encoder = encoder.to(device)
    encoder.eval()

    return encoder


@st.cache_resource
def create_feature_extractor(_device):
    """Feature extractor 생성"""
    feature_extractor = SBFbank(
        sample_rate=16000,
        f_min=0,
        f_max=8000,
        n_fft=400,
        n_mels=80,
        win_length=400,
        hop_length=160,
        postprocessor=SpecNormalization(),
    )
    return feature_extractor.to(_device)


@st.cache_data
def load_spk2utt_cached(path):
    """spk2utt 로드 (캐시)"""
    return load_spk2utt(path)


@st.cache_data
def extract_embeddings_for_speaker(
    speaker_id,
    files,
    _encoder,
    _feature_extractor,
    data_dir,
    device_str,
    max_files=None,
):
    """한 화자의 발화들에 대해 embedding을 추출합니다."""
    device = torch.device(device_str)
    embeddings = []
    valid_files = []

    if max_files is not None:
        files = files[:max_files]

    for filename in files:
        filepath = Path(data_dir) / filename
        if not filepath.exists():
            continue

        try:
            audio, sr = torchaudio.load(filepath, channels_first=True)
        except Exception:
            continue

        if sr != SAMPLE_RATE:
            audio = F.resample(audio, orig_freq=sr, new_freq=SAMPLE_RATE)

        audio = crop_or_pad(audio, SEGMENT_LENGTH)
        audio = audio.to(device)

        with torch.no_grad():
            spec = _feature_extractor(audio)
            emb = _encoder(spec)
            emb = emb.squeeze().cpu().numpy()

        embeddings.append(emb)
        valid_files.append(filename)

    return np.array(embeddings), valid_files


# ========================
# GMM 분석
# ========================
@st.cache_data
def compute_pairwise_similarities(embeddings):
    """모든 발화 쌍의 cosine similarity를 계산합니다."""
    # L2 normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
    embeddings_norm = embeddings / norms

    # 전체 similarity matrix
    sim_matrix = embeddings_norm @ embeddings_norm.T

    n = len(embeddings)
    pair_indices = []
    similarities = []

    for i in range(n):
        for j in range(i + 1, n):
            pair_indices.append((i, j))
            similarities.append(sim_matrix[i, j])

    return np.array(similarities), pair_indices


@st.cache_data
def fit_gmm(similarities, n_components=2, random_state=42):
    """Cosine similarity 분포에 GMM을 fitting합니다."""
    X = similarities.reshape(-1, 1)
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        random_state=random_state,
        n_init=5,
        max_iter=300,
    )
    gmm.fit(X)

    # 각 샘플의 posterior probability
    probs = gmm.predict_proba(X)
    labels = gmm.predict(X)

    # Clean component = 평균이 더 높은 component
    means = gmm.means_.flatten()
    clean_comp = np.argmax(means)
    noisy_comp = np.argmin(means)

    return gmm, probs, labels, clean_comp, noisy_comp


# ========================
# vMF Mixture Model
# ========================
def _log_vmf_normalizer(kappa, d):
    """vMF 분포의 log normalizing constant C_d(kappa).

    log C_d(κ) = (d/2-1)*log(κ) - (d/2)*log(2π) - log I_{d/2-1}(κ)
    여기서 I_v는 제1종 수정 베셀 함수이며, overflow를 피하기 위해
    exponentially-scaled 버전 ive를 사용합니다.
    """
    halfD = d / 2.0
    order = halfD - 1.0
    # ive(v, κ) = I_v(κ) * exp(-κ), so log I_v(κ) = log(ive(v,κ)) + κ
    log_ive_val = np.log(np.maximum(_ive(order, kappa), 1e-300))
    log_bessel = log_ive_val + kappa
    log_C = (halfD - 1) * np.log(np.maximum(kappa, 1e-10)) \
            - halfD * np.log(2 * np.pi) \
            - log_bessel
    return log_C


def _estimate_kappa(R_bar, d):
    """Mean resultant length R_bar로부터 κ를 추정합니다.

    Banerjee et al. (2005) 근사:
        κ ≈ R̄(d - R̄²) / (1 - R̄²)
    """
    kappa = R_bar * (d - R_bar ** 2) / (1.0 - R_bar ** 2 + 1e-10)
    return float(np.clip(kappa, 1e-6, 1e5))


@st.cache_data
def fit_vmf_mixture(embeddings, n_components=2, max_iter=200, tol=1e-6, random_state=42):
    """L2-normalized embedding에 vMF mixture model을 EM으로 fitting합니다.

    Returns
    -------
    dict with keys: weights, mus, kappas, resp (N,K), labels, clean_comp,
    noisy_comp, n_iter
    """
    rng = np.random.RandomState(random_state)

    # L2 normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
    X = embeddings / norms  # (N, d)
    N, d = X.shape

    # ── Initialise with K-means-style seeding ──
    # Pick K random points as initial means
    init_idx = rng.choice(N, size=n_components, replace=False)
    mus = X[init_idx].copy()  # (K, d)
    mus /= np.linalg.norm(mus, axis=1, keepdims=True) + 1e-10

    weights = np.ones(n_components) / n_components
    kappas = np.full(n_components, 10.0)

    resp = np.zeros((N, n_components))

    for iteration in range(max_iter):
        # ── E-step ──
        log_resp = np.zeros((N, n_components))
        for k in range(n_components):
            log_C = _log_vmf_normalizer(kappas[k], d)
            log_resp[:, k] = (
                np.log(weights[k] + 1e-300) + log_C + kappas[k] * (X @ mus[k])
            )

        # log-sum-exp normalisation
        log_max = log_resp.max(axis=1, keepdims=True)
        log_sum = log_max + np.log(np.exp(log_resp - log_max).sum(axis=1, keepdims=True))
        resp_new = np.exp(log_resp - log_sum)

        # ── Convergence check ──
        diff = np.abs(resp_new - resp).max()
        resp = resp_new
        if diff < tol:
            break

        # ── M-step ──
        Nk = resp.sum(axis=0)  # (K,)
        weights = Nk / N

        for k in range(n_components):
            weighted_sum = (resp[:, k : k + 1] * X).sum(axis=0)  # (d,)
            R_bar_vec_norm = np.linalg.norm(weighted_sum)
            mus[k] = weighted_sum / (R_bar_vec_norm + 1e-10)
            R_bar_k = R_bar_vec_norm / (Nk[k] + 1e-10)
            kappas[k] = _estimate_kappa(R_bar_k, d)

    labels = resp.argmax(axis=1)

    # Clean component = 더 많은 발화가 할당된 (majority) component
    counts = np.bincount(labels, minlength=n_components)
    clean_comp = int(np.argmax(counts))
    noisy_comp = int(np.argmin(counts))

    # 만약 두 component의 수가 같으면 κ가 큰 쪽을 clean으로
    if counts[clean_comp] == counts[noisy_comp]:
        clean_comp = int(np.argmax(kappas))
        noisy_comp = int(np.argmin(kappas))

    return {
        "weights": weights,
        "mus": mus,
        "kappas": kappas,
        "resp": resp,
        "labels": labels,
        "clean_comp": clean_comp,
        "noisy_comp": noisy_comp,
        "n_iter": iteration + 1,
    }


# ========================
# DataFrame 생성
# ========================
def build_pairs_dataframe(similarities, pair_indices, valid_files, gmm_probs, clean_comp, noisy_comp):
    """GMM 결과로 쌍 DataFrame을 구성합니다."""
    rows = []
    for k, (i, j) in enumerate(pair_indices):
        rows.append(
            {
                "pair_idx": k,
                "file_a_idx": i,
                "file_b_idx": j,
                "file_a": valid_files[i],
                "file_b": valid_files[j],
                "similarity": similarities[k],
                "p_clean": gmm_probs[k, clean_comp],
                "p_noisy": gmm_probs[k, noisy_comp],
                "label": "clean" if gmm_probs[k, clean_comp] > 0.5 else "noisy",
            }
        )

    df = pd.DataFrame(rows)
    return df


def build_pairs_dataframe_vmf(similarities, pair_indices, valid_files, vmf_result):
    """vMF 결과로 쌍 DataFrame을 구성합니다.

    발화 레벨 posterior에서 pair 레벨 확률을 도출합니다:
        P(pair clean) = P(utt_a ∈ clean) * P(utt_b ∈ clean)
    """
    resp = vmf_result["resp"]
    cc = vmf_result["clean_comp"]
    nc = vmf_result["noisy_comp"]

    rows = []
    for k, (i, j) in enumerate(pair_indices):
        p_clean_pair = resp[i, cc] * resp[j, cc]
        p_noisy_pair = 1.0 - p_clean_pair
        rows.append(
            {
                "pair_idx": k,
                "file_a_idx": i,
                "file_b_idx": j,
                "file_a": valid_files[i],
                "file_b": valid_files[j],
                "similarity": similarities[k],
                "p_clean": p_clean_pair,
                "p_noisy": p_noisy_pair,
                "label": "clean" if p_clean_pair > 0.5 else "noisy",
            }
        )

    return pd.DataFrame(rows)


# ========================
# 시각화
# ========================
def plot_gmm_distribution(similarities, gmm, clean_comp, noisy_comp):
    """GMM fitting 결과와 similarity 분포 히스토그램을 그립니다."""
    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_.flatten())
    weights = gmm.weights_

    x_range = np.linspace(
        similarities.min() - 0.1,
        similarities.max() + 0.1,
        500,
    )

    fig = go.Figure()

    # Histogram
    fig.add_trace(
        go.Histogram(
            x=similarities,
            nbinsx=60,
            histnorm="probability density",
            name="Similarity Distribution",
            marker_color="rgba(100, 149, 237, 0.5)",
            marker_line_color="rgba(100, 149, 237, 0.8)",
            marker_line_width=1,
        )
    )

    # Clean component
    y_clean = weights[clean_comp] * norm.pdf(x_range, means[clean_comp], stds[clean_comp])
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=y_clean,
            mode="lines",
            name=f"Clean (mu={means[clean_comp]:.3f}, std={stds[clean_comp]:.3f})",
            line=dict(color="green", width=3),
        )
    )

    # Noisy component
    y_noisy = weights[noisy_comp] * norm.pdf(x_range, means[noisy_comp], stds[noisy_comp])
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=y_noisy,
            mode="lines",
            name=f"Noisy (mu={means[noisy_comp]:.3f}, std={stds[noisy_comp]:.3f})",
            line=dict(color="red", width=3),
        )
    )

    # Mixture (total)
    y_total = y_clean + y_noisy
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=y_total,
            mode="lines",
            name="GMM Total",
            line=dict(color="black", width=2, dash="dash"),
        )
    )

    fig.update_layout(
        title="Pairwise Cosine Similarity Distribution + GMM Fit",
        xaxis_title="Cosine Similarity",
        yaxis_title="Density",
        height=450,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        barmode="overlay",
    )

    return fig


def plot_pair_scatter(df_pairs):
    """Clean/Noisy pair을 scatter plot으로 시각화합니다."""
    fig = go.Figure()

    # Clean pairs
    df_clean = df_pairs[df_pairs["label"] == "clean"]
    fig.add_trace(
        go.Scatter(
            x=df_clean["pair_idx"],
            y=df_clean["similarity"],
            mode="markers",
            marker=dict(size=6, color="green", opacity=0.6),
            name=f"Clean pairs (n={len(df_clean)})",
            text=[
                f"Pair #{row.pair_idx}<br>"
                f"Sim: {row.similarity:.4f}<br>"
                f"P(clean): {row.p_clean:.3f}<br>"
                f"A: {Path(row.file_a).name}<br>"
                f"B: {Path(row.file_b).name}"
                for row in df_clean.itertuples()
            ],
            hovertemplate="%{text}<extra></extra>",
            customdata=df_clean["pair_idx"].tolist(),
        )
    )

    # Noisy pairs
    df_noisy = df_pairs[df_pairs["label"] == "noisy"]
    fig.add_trace(
        go.Scatter(
            x=df_noisy["pair_idx"],
            y=df_noisy["similarity"],
            mode="markers",
            marker=dict(size=6, color="red", opacity=0.6),
            name=f"Noisy pairs (n={len(df_noisy)})",
            text=[
                f"Pair #{row.pair_idx}<br>"
                f"Sim: {row.similarity:.4f}<br>"
                f"P(noisy): {row.p_noisy:.3f}<br>"
                f"A: {Path(row.file_a).name}<br>"
                f"B: {Path(row.file_b).name}"
                for row in df_noisy.itertuples()
            ],
            hovertemplate="%{text}<extra></extra>",
            customdata=df_noisy["pair_idx"].tolist(),
        )
    )

    fig.update_layout(
        title="All Pairs — Similarity (sorted by pair index)",
        xaxis_title="Pair Index",
        yaxis_title="Cosine Similarity",
        height=350,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
    )

    return fig


def plot_vmf_distribution(similarities, pair_indices, vmf_result):
    """vMF mixture 결과를 similarity 히스토그램 위에 시각화합니다.

    vMF는 고차원 구면 위의 분포이므로 1D density curve 대신
    clean/noisy pair을 색으로 구분하여 보여줍니다.
    """
    resp = vmf_result["resp"]
    cc = vmf_result["clean_comp"]
    kappas = vmf_result["kappas"]
    weights = vmf_result["weights"]

    sims_clean, sims_noisy = [], []
    for s, (i, j) in zip(similarities, pair_indices):
        if resp[i, cc] > 0.5 and resp[j, cc] > 0.5:
            sims_clean.append(s)
        else:
            sims_noisy.append(s)

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=sims_clean,
            nbinsx=60,
            histnorm="probability density",
            name=f"Clean pairs (n={len(sims_clean)})",
            marker_color="rgba(50, 205, 50, 0.5)",
            marker_line_color="rgba(50, 205, 50, 0.8)",
            marker_line_width=1,
        )
    )

    fig.add_trace(
        go.Histogram(
            x=sims_noisy,
            nbinsx=60,
            histnorm="probability density",
            name=f"Noisy pairs (n={len(sims_noisy)})",
            marker_color="rgba(255, 69, 0, 0.5)",
            marker_line_color="rgba(255, 69, 0, 0.8)",
            marker_line_width=1,
        )
    )

    nc = vmf_result["noisy_comp"]
    fig.update_layout(
        title=(
            f"vMF Mixture — Similarity Distribution "
            f"(κ_clean={kappas[cc]:.1f}, κ_noisy={kappas[nc]:.1f}, "
            f"w=[{weights[cc]:.2f}, {weights[nc]:.2f}])"
        ),
        xaxis_title="Cosine Similarity",
        yaxis_title="Density",
        height=450,
        barmode="overlay",
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
    )

    return fig


def plot_vmf_utterance_posterior(valid_files, vmf_result):
    """각 발화의 vMF posterior (P(clean)) 를 바 차트로 표시합니다."""
    resp = vmf_result["resp"]
    cc = vmf_result["clean_comp"]
    p_clean = resp[:, cc]

    colors = ["green" if p > 0.5 else "red" for p in p_clean]
    names = [Path(f).name for f in valid_files]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=list(range(len(valid_files))),
            y=p_clean,
            marker_color=colors,
            text=[f"{names[i]}<br>P(clean)={p_clean[i]:.3f}" for i in range(len(valid_files))],
            hovertemplate="%{text}<extra></extra>",
        )
    )

    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="threshold=0.5")

    fig.update_layout(
        title="Per-Utterance P(clean) from vMF Mixture",
        xaxis_title="Utterance Index",
        yaxis_title="P(clean)",
        height=350,
        showlegend=False,
    )

    return fig


# ========================
# 오디오 재생 헬퍼
# ========================
def render_audio_player(filepath, label=""):
    """오디오 파일을 로드하여 Streamlit 오디오 플레이어를 표시합니다."""
    filepath = Path(filepath)
    if not filepath.exists():
        st.error(f"파일 없음: {filepath}")
        return

    try:
        audio, sr = torchaudio.load(filepath)
        audio_np = audio.numpy().flatten()
        if label:
            st.markdown(f"**{label}**")
        st.code(str(filepath.name), language=None)
        st.audio(audio_np, sample_rate=sr)
        duration = len(audio_np) / sr
        st.caption(f"Duration: {duration:.2f}s | SR: {sr}Hz")
    except Exception as e:
        st.error(f"오디오 로드 실패: {e}")


# ========================
# Main
# ========================
def main():
    st.set_page_config(
        page_title="Noisy Pair Detector (GMM / vMF)",
        page_icon="🔍",
        layout="wide",
    )

    st.title("🔍 Noisy Pair Detector — GMM / vMF")
    st.markdown(
        "화자 내 발화 쌍을 **GMM** 또는 **vMF Mixture** 로 fitting하여 "
        "**clean pair**과 **noisy pair**을 분리합니다. 쌍을 선택하면 두 음성을 비교할 수 있습니다."
    )

    # Device 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_str = str(device)

    # 모델 로드
    with st.spinner("모델 로딩 중..."):
        encoder = load_model(CKPT_PATH, device)
        feature_extractor = create_feature_extractor(device)

    # spk2utt 로드
    spk2utt = load_spk2utt_cached(SPK2UTT_PATH)
    speaker_ids = sorted(list(spk2utt.keys()))

    # ─── Sidebar ───
    st.sidebar.header("설정")

    selected_speaker = st.sidebar.selectbox("화자 선택", speaker_ids, index=0)

    fitting_method = st.sidebar.radio(
        "Fitting 방법",
        ["GMM", "vMF Mixture"],
        index=0,
        help="GMM: similarity 분포에 Gaussian Mixture 적용\n\n"
             "vMF Mixture: L2-normalized embedding에 von Mises-Fisher Mixture 적용",
    )

    files = spk2utt[selected_speaker]
    st.sidebar.info(f"총 발화 수: {len(files)}개")

    show_mode = st.sidebar.radio(
        "표시할 쌍 필터",
        ["All", "Noisy only", "Clean only"],
        index=1,
        help="테이블에 표시할 쌍을 필터링합니다.",
    )

    sort_by = st.sidebar.selectbox(
        "정렬 기준",
        ["p_noisy (높은순)", "p_clean (높은순)", "similarity (낮은순)", "similarity (높은순)"],
        index=0,
    )

    # ─── Embedding 추출 ───
    with st.spinner(f"'{selected_speaker}' 발화 embedding 추출 중..."):
        embeddings, valid_files = extract_embeddings_for_speaker(
            speaker_id=selected_speaker,
            files=files,
            _encoder=encoder,
            _feature_extractor=feature_extractor,
            data_dir=str(DATA_DIR),
            device_str=device_str,
        )

    if len(embeddings) < 3:
        st.error(f"유효한 발화가 3개 미만입니다 ({len(embeddings)}개). Fitting이 불가합니다.")
        return

    # ─── Pairwise similarity ───
    with st.spinner("Pairwise cosine similarity 계산 중..."):
        similarities, pair_indices = compute_pairwise_similarities(embeddings)

    n_pairs = len(similarities)
    st.sidebar.metric("발화 수 (유효)", len(valid_files))
    st.sidebar.metric("총 쌍 수", f"{n_pairs:,}")

    # ═══════════════════════════════════════════
    # Fitting: GMM vs vMF
    # ═══════════════════════════════════════════
    if fitting_method == "GMM":
        # ─── GMM Fitting ───
        with st.spinner("GMM fitting 중 (2 components)..."):
            gmm, gmm_probs, gmm_labels, clean_comp, noisy_comp = fit_gmm(similarities)

        df_pairs = build_pairs_dataframe(
            similarities, pair_indices, valid_files, gmm_probs, clean_comp, noisy_comp
        )

        n_clean = (df_pairs["label"] == "clean").sum()
        n_noisy = (df_pairs["label"] == "noisy").sum()

        means = gmm.means_.flatten()
        stds = np.sqrt(gmm.covariances_.flatten())
        weights = gmm.weights_

        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("Clean pairs", f"{n_clean:,} ({n_clean / n_pairs:.1%})")
        with col_m2:
            st.metric("Noisy pairs", f"{n_noisy:,} ({n_noisy / n_pairs:.1%})")
        with col_m3:
            st.metric("GMM BIC", f"{gmm.bic(similarities.reshape(-1, 1)):.1f}")

        fig_dist = plot_gmm_distribution(similarities, gmm, clean_comp, noisy_comp)
        st.plotly_chart(fig_dist, use_container_width=True)

        with st.expander("GMM Component 상세 정보"):
            comp_df = pd.DataFrame(
                {
                    "Component": ["Clean", "Noisy"],
                    "Mean": [means[clean_comp], means[noisy_comp]],
                    "Std": [stds[clean_comp], stds[noisy_comp]],
                    "Weight": [weights[clean_comp], weights[noisy_comp]],
                }
            )
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

    else:
        # ─── vMF Mixture Fitting ───
        with st.spinner("vMF Mixture fitting 중 (2 components, EM)..."):
            vmf_result = fit_vmf_mixture(embeddings)

        df_pairs = build_pairs_dataframe_vmf(
            similarities, pair_indices, valid_files, vmf_result
        )

        n_clean = (df_pairs["label"] == "clean").sum()
        n_noisy = (df_pairs["label"] == "noisy").sum()

        cc = vmf_result["clean_comp"]
        nc = vmf_result["noisy_comp"]
        kappas = vmf_result["kappas"]
        weights_vmf = vmf_result["weights"]
        utt_labels = vmf_result["labels"]
        n_utt_clean = int((utt_labels == cc).sum())
        n_utt_noisy = int((utt_labels == nc).sum())

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("Clean pairs", f"{n_clean:,} ({n_clean / n_pairs:.1%})")
        with col_m2:
            st.metric("Noisy pairs", f"{n_noisy:,} ({n_noisy / n_pairs:.1%})")
        with col_m3:
            st.metric("Clean utterances", f"{n_utt_clean} / {len(valid_files)}")
        with col_m4:
            st.metric("EM iterations", vmf_result["n_iter"])

        fig_dist = plot_vmf_distribution(similarities, pair_indices, vmf_result)
        st.plotly_chart(fig_dist, use_container_width=True)

        # Per-utterance posterior
        fig_utt = plot_vmf_utterance_posterior(valid_files, vmf_result)
        st.plotly_chart(fig_utt, use_container_width=True)

        with st.expander("vMF Component 상세 정보"):
            comp_df = pd.DataFrame(
                {
                    "Component": ["Clean", "Noisy"],
                    "κ (concentration)": [kappas[cc], kappas[nc]],
                    "Weight": [weights_vmf[cc], weights_vmf[nc]],
                    "# Utterances": [n_utt_clean, n_utt_noisy],
                }
            )
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

    # ─── Pair Scatter (공통) ───
    fig_scatter = plot_pair_scatter(df_pairs)
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.divider()

    # ─── Pair 목록 ───
    st.subheader("📋 Pair 목록")

    # 필터
    if show_mode == "Noisy only":
        df_show = df_pairs[df_pairs["label"] == "noisy"].copy()
    elif show_mode == "Clean only":
        df_show = df_pairs[df_pairs["label"] == "clean"].copy()
    else:
        df_show = df_pairs.copy()

    # 정렬
    if sort_by == "p_noisy (높은순)":
        df_show = df_show.sort_values("p_noisy", ascending=False)
    elif sort_by == "p_clean (높은순)":
        df_show = df_show.sort_values("p_clean", ascending=False)
    elif sort_by == "similarity (낮은순)":
        df_show = df_show.sort_values("similarity", ascending=True)
    else:
        df_show = df_show.sort_values("similarity", ascending=False)

    df_show = df_show.reset_index(drop=True)

    st.caption(f"표시 중: {len(df_show)}개 쌍 ({show_mode})")

    # 페이지네이션
    PAGE_SIZE = 20
    total_pages = max(1, (len(df_show) + PAGE_SIZE - 1) // PAGE_SIZE)
    page = st.number_input("페이지", min_value=1, max_value=total_pages, value=1, step=1)
    start_idx = (page - 1) * PAGE_SIZE
    end_idx = min(start_idx + PAGE_SIZE, len(df_show))

    df_page = df_show.iloc[start_idx:end_idx]

    # 테이블 표시
    display_df = df_page[["pair_idx", "file_a", "file_b", "similarity", "p_clean", "p_noisy", "label"]].copy()
    display_df["file_a"] = display_df["file_a"].apply(lambda x: Path(x).name)
    display_df["file_b"] = display_df["file_b"].apply(lambda x: Path(x).name)
    display_df["similarity"] = display_df["similarity"].apply(lambda x: f"{x:.4f}")
    display_df["p_clean"] = display_df["p_clean"].apply(lambda x: f"{x:.3f}")
    display_df["p_noisy"] = display_df["p_noisy"].apply(lambda x: f"{x:.3f}")

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "pair_idx": st.column_config.NumberColumn("Pair #", width="small"),
            "file_a": st.column_config.TextColumn("File A"),
            "file_b": st.column_config.TextColumn("File B"),
            "similarity": st.column_config.TextColumn("Cos Sim"),
            "p_clean": st.column_config.TextColumn("P(clean)"),
            "p_noisy": st.column_config.TextColumn("P(noisy)"),
            "label": st.column_config.TextColumn("Label"),
        },
    )

    st.divider()

    # ─── 쌍 비교 청취 ───
    st.subheader("🔊 쌍 비교 청취")

    # 드롭다운 또는 버튼으로 쌍 선택
    pair_options = []
    for _, row in df_page.iterrows():
        tag = "🟢" if row["label"] == "clean" else "🔴"
        pair_options.append(
            f"{tag} Pair #{row['pair_idx']}  |  sim={row['similarity']:.4f}  |  "
            f"{Path(row['file_a']).name}  ↔  {Path(row['file_b']).name}"
        )

    if len(pair_options) == 0:
        st.info("표시할 쌍이 없습니다.")
        return

    selected_pair_str = st.selectbox("비교할 쌍 선택", pair_options, index=0)
    selected_row = df_page.iloc[pair_options.index(selected_pair_str)]

    # 쌍 상세 정보
    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        label_color = "green" if selected_row["label"] == "clean" else "red"
        st.markdown(
            f"**Label:** :{label_color}[{selected_row['label'].upper()}]"
        )
    with col_info2:
        st.markdown(f"**Cosine Similarity:** `{selected_row['similarity']:.4f}`")
    with col_info3:
        st.markdown(
            f"**P(clean):** `{selected_row['p_clean']:.3f}` &nbsp; "
            f"**P(noisy):** `{selected_row['p_noisy']:.3f}`"
        )

    # 두 오디오 나란히 재생
    col_a, col_b = st.columns(2)
    with col_a:
        render_audio_player(DATA_DIR / selected_row["file_a"], label="File A")
    with col_b:
        render_audio_player(DATA_DIR / selected_row["file_b"], label="File B")

    # ─── 빠른 탐색: 버튼으로 이전/다음 쌍 ───
    st.divider()
    st.subheader("⚡ 빠른 탐색")
    st.caption("현재 페이지 내의 쌍을 버튼으로 빠르게 탐색합니다.")

    for idx_in_page, (_, row) in enumerate(df_page.iterrows()):
        tag = "🟢" if row["label"] == "clean" else "🔴"
        btn_label = (
            f"{tag} #{row['pair_idx']} | sim={row['similarity']:.4f} | "
            f"{Path(row['file_a']).name} ↔ {Path(row['file_b']).name}"
        )
        if st.button(btn_label, key=f"quick_{row['pair_idx']}"):
            st.markdown("---")
            qcol_a, qcol_b = st.columns(2)
            with qcol_a:
                render_audio_player(DATA_DIR / row["file_a"], label="File A")
            with qcol_b:
                render_audio_player(DATA_DIR / row["file_b"], label="File B")


if __name__ == "__main__":
    main()
