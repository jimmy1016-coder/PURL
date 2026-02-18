"""
화자별 embedding t-SNE 시각화를 Streamlit으로 인터랙티브하게 보여주는 앱.
점을 클릭하면 해당 음성이 재생됩니다.

실행 방법:
    streamlit run streamlit_tsne.py
"""

import sys
from pathlib import Path

# Add src directory to Python path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn.functional as F
import torchaudio
from sklearn.manifold import TSNE

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
    
    state_dict = checkpoint['state_dict']
    encoder_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('encoder.'):
            new_key = k.replace('encoder.', '')
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


@st.cache_data
def compute_tsne(embeddings, perplexity):
    """t-SNE 계산"""
    if len(embeddings) < 2:
        return None
    
    actual_perplexity = min(perplexity, len(embeddings) - 1)
    if actual_perplexity < 1:
        actual_perplexity = 1
    
    tsne = TSNE(n_components=2, perplexity=actual_perplexity, random_state=42, n_iter=1000)
    return tsne.fit_transform(embeddings)


def main():
    st.set_page_config(
        page_title="Speaker Embedding t-SNE Viewer",
        page_icon="🎵",
        layout="wide"
    )
    
    st.title("🎵 Speaker Embedding t-SNE Visualization")
    st.markdown("점을 클릭하면 해당 음성이 재생됩니다.")
    
    # Device 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_str = str(device)
    
    # 모델 로드
    with st.spinner("모델 로딩 중..."):
        encoder = load_model(CKPT_PATH, device)
        feature_extractor = create_feature_extractor(device)
    
    # spk2utt 로드
    spk2utt = load_spk2utt_cached(SPK2UTT_PATH)
    speaker_ids = sorted(list(spk2utt.keys()))
    
    # Sidebar 설정
    st.sidebar.header("설정")
    
    # 화자 선택
    selected_speaker = st.sidebar.selectbox(
        "화자 선택",
        speaker_ids,
        index=0
    )
    
    # Perplexity 설정
    perplexity = st.sidebar.slider(
        "t-SNE Perplexity",
        min_value=2,
        max_value=50,
        value=10,
        help="낮을수록 국소 구조, 높을수록 전역 구조 강조"
    )
    
    # 최대 파일 수 설정
    max_files = st.sidebar.number_input(
        "최대 발화 수 (0=전체)",
        min_value=0,
        max_value=500,
        value=0,
        help="0이면 모든 발화 사용"
    )
    if max_files == 0:
        max_files = None
    
    # 화자 정보 표시
    files = spk2utt[selected_speaker]
    st.sidebar.info(f"총 발화 수: {len(files)}개")
    
    # Embedding 추출
    with st.spinner(f"{selected_speaker} 발화 embedding 추출 중..."):
        embeddings, valid_files = extract_embeddings_for_speaker(
            speaker_id=selected_speaker,
            files=files,
            _encoder=encoder,
            _feature_extractor=feature_extractor,
            data_dir=str(DATA_DIR),
            device_str=device_str,
            max_files=max_files,
        )
    
    if len(embeddings) < 2:
        st.error(f"유효한 발화가 2개 미만입니다. ({len(embeddings)}개)")
        return
    
    # t-SNE 계산
    with st.spinner("t-SNE 계산 중..."):
        embeddings_2d = compute_tsne(embeddings, perplexity)
    
    if embeddings_2d is None:
        st.error("t-SNE 계산 실패")
        return
    
    # Centroid 계산
    centroid = embeddings_2d.mean(axis=0)
    
    # DataFrame 생성
    df = pd.DataFrame({
        'x': embeddings_2d[:, 0],
        'y': embeddings_2d[:, 1],
        'file': valid_files,
        'index': range(len(valid_files)),
    })
    
    # Plotly 차트 생성
    fig = go.Figure()
    
    # Utterance points
    fig.add_trace(go.Scatter(
        x=df['x'],
        y=df['y'],
        mode='markers',
        marker=dict(
            size=12,
            color='steelblue',
            opacity=0.7,
            line=dict(width=1, color='white')
        ),
        text=[f"Index: {i}<br>File: {f}" for i, f in enumerate(valid_files)],
        hovertemplate='%{text}<extra></extra>',
        name=f'Utterances (n={len(valid_files)})',
        customdata=list(range(len(valid_files))),
    ))
    
    # Centroid (별 모양)
    fig.add_trace(go.Scatter(
        x=[centroid[0]],
        y=[centroid[1]],
        mode='markers',
        marker=dict(
            size=25,
            color='red',
            symbol='star',
            line=dict(width=2, color='darkred')
        ),
        name='Centroid',
        hovertemplate='Centroid<extra></extra>',
    ))
    
    fig.update_layout(
        title=f"Speaker: {selected_speaker} (perplexity={min(perplexity, len(embeddings)-1)})",
        xaxis_title="t-SNE Dimension 1",
        yaxis_title="t-SNE Dimension 2",
        height=600,
        hovermode='closest',
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
    )
    
    # 차트 표시 (selection 이벤트 사용)
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_points = st.plotly_chart(
            fig, 
            use_container_width=True, 
            key="tsne_plot",
            on_select="rerun",
            selection_mode="points",
        )
    
    with col2:
        st.subheader("🔊 음성 재생")
        
        # 선택된 점 처리
        if selected_points and selected_points.selection and selected_points.selection.points:
            point_data = selected_points.selection.points[0]
            
            # point_index 추출
            if 'point_index' in point_data:
                idx = point_data['point_index']
                
                # centroid가 아닌 경우만 처리 (첫 번째 trace가 utterances)
                if 'curve_number' not in point_data or point_data.get('curve_number', 0) == 0:
                    if idx < len(valid_files):
                        selected_file = valid_files[idx]
                        filepath = DATA_DIR / selected_file
                        
                        st.success(f"선택된 파일:")
                        st.code(selected_file, language=None)
                        
                        if filepath.exists():
                            # 오디오 로드 및 재생
                            try:
                                audio, sr = torchaudio.load(filepath)
                                audio_np = audio.numpy().flatten()
                                st.audio(audio_np, sample_rate=sr)
                                
                                # 추가 정보
                                duration = len(audio_np) / sr
                                st.caption(f"Duration: {duration:.2f}s | Sample Rate: {sr}Hz")
                            except Exception as e:
                                st.error(f"오디오 로드 실패: {e}")
                        else:
                            st.error(f"파일이 존재하지 않습니다: {filepath}")
                else:
                    st.info("Centroid를 선택했습니다. 발화 점을 선택해주세요.")
        else:
            st.info("👆 왼쪽 차트에서 점을 클릭하세요!")
        
        # 파일 목록
        st.subheader("📁 발화 목록")
        with st.expander(f"전체 발화 보기 ({len(valid_files)}개)"):
            for i, f in enumerate(valid_files):
                if st.button(f"▶ {i}: {Path(f).name}", key=f"btn_{i}"):
                    filepath = DATA_DIR / f
                    if filepath.exists():
                        audio, sr = torchaudio.load(filepath)
                        st.audio(audio.numpy().flatten(), sample_rate=sr)


if __name__ == "__main__":
    main()
