"""
Noisy sample 필터링 스크립트 (Batch 처리 버전).

각 화자별로:
1. 모든 발화의 embedding 추출 (batch 단위)
2. Centroid 계산
3. 각 발화와 centroid의 cosine similarity 계산
4. 상위 N% (기본 50%)만 clean으로 선택
5. 새로운 spk2utt_clean 파일 생성

사용법:
    python filter_noisy_samples.py --keep_ratio 0.5 --batch_size 64
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

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from data.augmentations import SpecNormalization
from data.data_utils import crop_or_pad, load_spk2utt
from data.feature_extractors import SBFbank
from models import ECAPAEncoder


class SimpleUtteranceDataset(Dataset):
    """단순히 파일 경로 목록을 받아서 spectrogram을 반환하는 Dataset"""
    
    def __init__(self, file_paths, data_dir, feature_extractor, sample_rate=16000, segment_length=48000):
        self.file_paths = file_paths
        self.data_dir = Path(data_dir)
        self.feature_extractor = feature_extractor
        self.sample_rate = sample_rate
        self.segment_length = segment_length
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        filename = self.file_paths[idx]
        filepath = self.data_dir / filename
        
        try:
            audio, sr = torchaudio.load(filepath, channels_first=True)
            
            if sr != self.sample_rate:
                audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=self.sample_rate)
            
            audio = crop_or_pad(audio, self.segment_length)
            spec = self.feature_extractor(audio)  # [1, T, 80]
            
            return spec.squeeze(0), idx, True  # spec, index, valid flag
            
        except Exception as e:
            # 실패 시 더미 데이터 반환
            dummy_spec = torch.zeros(300, 80)  # 대략적인 크기
            return dummy_spec, idx, False


def collate_fn(batch):
    """가변 길이 spectrogram을 패딩하여 배치로 만듦"""
    specs, indices, valid_flags = zip(*batch)
    
    # 최대 길이 찾기
    max_len = max(s.shape[0] for s in specs)
    
    # 패딩
    padded_specs = []
    for spec in specs:
        if spec.shape[0] < max_len:
            pad_size = max_len - spec.shape[0]
            spec = F.pad(spec, (0, 0, 0, pad_size))
        padded_specs.append(spec)
    
    batch_specs = torch.stack(padded_specs, dim=0)  # [B, T, 80]
    indices = torch.tensor(indices)
    valid_flags = torch.tensor(valid_flags)
    
    return batch_specs, indices, valid_flags


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


def create_feature_extractor():
    """Feature extractor 생성 (CPU에서 동작)"""
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
    return feature_extractor


def extract_all_embeddings_batch(encoder, all_files, data_dir, device, batch_size=64, num_workers=4):
    """모든 파일의 embedding을 batch 단위로 추출"""
    
    feature_extractor = create_feature_extractor()
    
    dataset = SimpleUtteranceDataset(
        file_paths=all_files,
        data_dir=data_dir,
        feature_extractor=feature_extractor,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    all_embeddings = [None] * len(all_files)
    valid_mask = [False] * len(all_files)
    
    with torch.no_grad():
        for batch_specs, batch_indices, batch_valid in tqdm(dataloader, desc="Extracting embeddings"):
            batch_specs = batch_specs.to(device)
            
            # Embedding 추출
            embeddings = encoder(batch_specs)  # [B, 1, 192]
            embeddings = embeddings.squeeze(1).cpu().numpy()  # [B, 192]
            
            # 결과 저장
            for i, (idx, valid) in enumerate(zip(batch_indices, batch_valid)):
                idx = idx.item()
                if valid:
                    all_embeddings[idx] = embeddings[i]
                    valid_mask[idx] = True
    
    return all_embeddings, valid_mask


def cosine_similarity(a, b):
    """Cosine similarity 계산"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


def filter_speaker_utterances(embeddings, files, keep_ratio=0.5):
    """
    화자 내에서 centroid와 가까운 상위 keep_ratio만 선택
    """
    if len(embeddings) == 0:
        return [], [], []
    
    # Centroid 계산
    centroid = np.mean(embeddings, axis=0)
    
    # 각 발화와 centroid의 similarity 계산
    similarities = [cosine_similarity(emb, centroid) for emb in embeddings]
    
    # Similarity 기준 정렬
    sorted_indices = np.argsort(similarities)[::-1]  # 내림차순
    
    # 상위 keep_ratio 선택
    n_keep = max(1, int(len(files) * keep_ratio))  # 최소 1개는 유지
    
    clean_indices = sorted_indices[:n_keep]
    noisy_indices = sorted_indices[n_keep:]
    
    clean_files = [files[i] for i in clean_indices]
    noisy_files = [files[i] for i in noisy_indices]
    
    return clean_files, noisy_files, similarities


def main():
    parser = argparse.ArgumentParser(description='Noisy sample 필터링 (Batch 버전)')
    parser.add_argument('--ckpt_path', type=str,
                        default='/home/sooyoung/interspeech/chns/outputs/chns-supcon/checkpoints/epoch=184_step=308395.ckpt',
                        help='체크포인트 경로')
    parser.add_argument('--data_dir', type=str,
                        default='/home/sooyoung/interspeech/dev/aac/',
                        help='데이터 디렉터리')
    parser.add_argument('--spk2utt_path', type=str,
                        default='/home/sooyoung/interspeech/dev/aac/spk2utt',
                        help='원본 spk2utt 파일 경로')
    parser.add_argument('--output_spk2utt', type=str,
                        default='/home/sooyoung/interspeech/dev/aac/spk2utt_clean',
                        help='출력 spk2utt 파일 경로')
    parser.add_argument('--output_stats', type=str,
                        default='/home/sooyoung/interspeech/chns/outputs/filtering_stats.pkl',
                        help='필터링 통계 저장 경로')
    parser.add_argument('--keep_ratio', type=float, default=0.5,
                        help='유지할 비율 (0.5 = 상위 50%%)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='배치 크기')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='DataLoader worker 수')
    parser.add_argument('--device', type=str, default='cuda',
                        help='device (cuda or cpu)')
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Num workers: {args.num_workers}")
    
    # 모델 로드
    print(f"\nLoading model from: {args.ckpt_path}")
    encoder = load_model(args.ckpt_path, device)
    
    # spk2utt 로드
    print(f"Loading spk2utt from: {args.spk2utt_path}")
    spk2utt = load_spk2utt(args.spk2utt_path)
    
    # 모든 파일 목록과 화자 매핑 생성
    all_files = []
    file_to_speaker = {}
    speaker_file_indices = defaultdict(list)
    
    for speaker_id, files in spk2utt.items():
        for f in files:
            idx = len(all_files)
            all_files.append(f)
            file_to_speaker[f] = speaker_id
            speaker_file_indices[speaker_id].append(idx)
    
    print(f"Total files: {len(all_files):,}")
    print(f"Total speakers: {len(spk2utt)}")
    
    # 모든 embedding 추출 (batch)
    print(f"\nExtracting embeddings...")
    all_embeddings, valid_mask = extract_all_embeddings_batch(
        encoder=encoder,
        all_files=all_files,
        data_dir=args.data_dir,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    
    # 화자별 필터링
    print(f"\nFiltering speakers...")
    spk2utt_clean = {}
    spk2utt_noisy = {}
    all_stats = {}
    
    total_original = 0
    total_clean = 0
    total_noisy = 0
    
    for speaker_id in tqdm(spk2utt.keys(), desc="Processing speakers"):
        indices = speaker_file_indices[speaker_id]
        
        # 유효한 embedding만 수집
        speaker_embeddings = []
        speaker_files = []
        
        for idx in indices:
            if valid_mask[idx] and all_embeddings[idx] is not None:
                speaker_embeddings.append(all_embeddings[idx])
                speaker_files.append(all_files[idx])
        
        total_original += len(indices)
        
        if len(speaker_embeddings) == 0:
            continue
        
        speaker_embeddings = np.array(speaker_embeddings)
        
        # 필터링
        clean_files, noisy_files, similarities = filter_speaker_utterances(
            speaker_embeddings, speaker_files, keep_ratio=args.keep_ratio
        )
        
        # 결과 저장
        if len(clean_files) > 0:
            spk2utt_clean[speaker_id] = clean_files
        
        if len(noisy_files) > 0:
            spk2utt_noisy[speaker_id] = noisy_files
        
        # 통계 저장
        all_stats[speaker_id] = {
            'original_count': len(indices),
            'valid_count': len(speaker_files),
            'clean_count': len(clean_files),
            'noisy_count': len(noisy_files),
            'similarities': dict(zip(speaker_files, similarities)),
            'clean_files': clean_files,
            'noisy_files': noisy_files,
        }
        
        total_clean += len(clean_files)
        total_noisy += len(noisy_files)
    
    # spk2utt_clean 파일 저장
    print(f"\nSaving clean spk2utt to: {args.output_spk2utt}")
    with open(args.output_spk2utt, 'w') as f:
        for speaker_id, files in spk2utt_clean.items():
            line = speaker_id + ' ' + ' '.join(files)
            f.write(line + '\n')
    
    # 통계 저장
    Path(args.output_stats).parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving stats to: {args.output_stats}")
    with open(args.output_stats, 'wb') as f:
        pickle.dump({
            'spk2utt_clean': spk2utt_clean,
            'spk2utt_noisy': spk2utt_noisy,
            'all_stats': all_stats,
            'keep_ratio': args.keep_ratio,
        }, f)
    
    # 요약 출력
    print("\n" + "="*50)
    print("FILTERING SUMMARY")
    print("="*50)
    print(f"Keep ratio: {args.keep_ratio:.1%}")
    print(f"Total speakers: {len(spk2utt)}")
    print(f"Speakers with clean data: {len(spk2utt_clean)}")
    print(f"")
    print(f"Original utterances: {total_original:,}")
    print(f"Clean utterances: {total_clean:,} ({total_clean/total_original:.1%})")
    print(f"Noisy utterances: {total_noisy:,} ({total_noisy/total_original:.1%})")
    print("="*50)
    
    print(f"\n다음 단계:")
    print(f"1. config 파일에서 spk2utt_file_path를 변경하세요:")
    print(f"   spk2utt_file_path: {args.output_spk2utt}")
    print(f"2. 모델을 다시 학습하세요:")
    print(f"   python run_train.py --config configs/supcon.yaml")


if __name__ == "__main__":
    main()
