"""
화자별 embedding을 추출하고 t-SNE로 시각화하는 스크립트.
각 화자 ID별로 embedding들과 centroid(별 모양)를 plot하여 저장합니다.
"""

import sys
from pathlib import Path

# Add src directory to Python path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import argparse
import pickle
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from sklearn.manifold import TSNE
from tqdm import tqdm

from data.augmentations import SpecNormalization
from data.data_utils import crop_or_pad, load_spk2utt
from data.feature_extractors import SBFbank
from models import ECAPAEncoder
from trainers import SupConTrainer


def load_model(ckpt_path, device):
    """체크포인트에서 모델을 로드합니다."""
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # 모델 생성 (config와 동일한 구조)
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
    
    # state_dict에서 encoder 부분만 추출
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


def create_feature_extractor(device):
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
    return feature_extractor.to(device)


def extract_embeddings_for_speaker(
    encoder, 
    feature_extractor, 
    data_dir, 
    files, 
    sample_rate=16000, 
    segment_length=48000,
    device='cuda',
    max_files=None,
):
    """한 화자의 발화들에 대해 embedding을 추출합니다."""
    embeddings = []
    
    if max_files is not None:
        files = files[:max_files]
    
    for filename in files:
        filepath = data_dir / filename
        if not filepath.exists():
            continue
        
        try:
            audio, sr = torchaudio.load(filepath, channels_first=True)
        except Exception:
            continue
        
        if sr != sample_rate:
            audio = F.resample(audio, orig_freq=sr, new_freq=sample_rate)
        
        audio = crop_or_pad(audio, segment_length)
        audio = audio.to(device)
        
        with torch.no_grad():
            spec = feature_extractor(audio)  # [1, T, 80]
            emb = encoder(spec)  # [1, 1, 192]
            emb = emb.squeeze().cpu().numpy()
        
        embeddings.append(emb)
    
    return np.array(embeddings)


def visualize_speaker_tsne(embeddings, speaker_id, output_path, perplexity=5):
    """한 화자의 embedding들을 t-SNE로 시각화합니다."""
    if len(embeddings) < 2:
        print(f"  [Skip] {speaker_id}: 샘플 수 부족 ({len(embeddings)}개)")
        return False
    
    # t-SNE 적용
    # perplexity는 샘플 수 - 1 이하여야 함
    actual_perplexity = min(perplexity, len(embeddings) - 1)
    if actual_perplexity < 1:
        actual_perplexity = 1
    
    tsne = TSNE(n_components=2, perplexity=actual_perplexity, random_state=42, n_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Centroid 계산 (t-SNE 공간에서)
    centroid = embeddings_2d.mean(axis=0)
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Embedding points
    ax.scatter(
        embeddings_2d[:, 0], 
        embeddings_2d[:, 1], 
        c='steelblue', 
        alpha=0.7, 
        s=60, 
        edgecolors='white',
        linewidths=0.5,
        label=f'Utterances (n={len(embeddings)})'
    )
    
    # Centroid (별 모양)
    ax.scatter(
        centroid[0], 
        centroid[1], 
        c='red', 
        marker='*', 
        s=400, 
        edgecolors='darkred',
        linewidths=1.5,
        label='Centroid',
        zorder=5
    )
    
    ax.set_title(f'Speaker: {speaker_id}\nt-SNE Visualization (perplexity={actual_perplexity})', fontsize=14)
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return True


def main():
    parser = argparse.ArgumentParser(description='화자별 embedding t-SNE 시각화')
    parser.add_argument('--ckpt_path', type=str, 
                        default='/home/sooyoung/interspeech/chns/outputs/chns-supcon/checkpoints/epoch=184_step=308395.ckpt',
                        help='체크포인트 경로')
    parser.add_argument('--data_dir', type=str, 
                        default='/home/sooyoung/interspeech/dev/aac/',
                        help='데이터 디렉터리')
    parser.add_argument('--spk2utt_path', type=str, 
                        default='/home/sooyoung/interspeech/dev/aac/spk2utt',
                        help='spk2utt 파일 경로')
    parser.add_argument('--output_dir', type=str, 
                        default='/home/sooyoung/interspeech/chns/outputs/tsne_visualizations',
                        help='출력 디렉터리')
    parser.add_argument('--max_speakers', type=int, default=None,
                        help='시각화할 최대 화자 수 (None이면 전체)')
    parser.add_argument('--max_files_per_speaker', type=int, default=None,
                        help='화자당 최대 발화 수 (None이면 전체)')
    parser.add_argument('--perplexity', type=float, default=30,
                        help='t-SNE perplexity')
    parser.add_argument('--device', type=str, default='cuda',
                        help='device (cuda or cpu)')
    args = parser.parse_args()
    
    # 디바이스 설정
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 출력 디렉터리 생성
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # 모델 로드
    print(f"Loading model from: {args.ckpt_path}")
    encoder = load_model(args.ckpt_path, device)
    feature_extractor = create_feature_extractor(device)
    
    # spk2utt 로드
    print(f"Loading spk2utt from: {args.spk2utt_path}")
    spk2utt = load_spk2utt(args.spk2utt_path)
    speaker_ids = list(spk2utt.keys())
    
    if args.max_speakers is not None:
        speaker_ids = speaker_ids[:args.max_speakers]
    
    print(f"Total speakers to visualize: {len(speaker_ids)}")
    
    data_dir = Path(args.data_dir)
    
    # 각 화자별로 처리
    success_count = 0
    for speaker_id in tqdm(speaker_ids, desc="Processing speakers"):
        files = spk2utt[speaker_id]
        
        # Embedding 추출
        embeddings = extract_embeddings_for_speaker(
            encoder=encoder,
            feature_extractor=feature_extractor,
            data_dir=data_dir,
            files=files,
            device=device,
            max_files=args.max_files_per_speaker,
        )
        
        if len(embeddings) == 0:
            print(f"  [Skip] {speaker_id}: No valid files")
            continue
        
        # 시각화
        output_path = output_dir / f"{speaker_id}.png"
        success = visualize_speaker_tsne(
            embeddings=embeddings,
            speaker_id=speaker_id,
            output_path=output_path,
            perplexity=args.perplexity,
        )
        
        if success:
            success_count += 1
    
    print(f"\nDone! Successfully visualized {success_count}/{len(speaker_ids)} speakers")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
