"""
NCU (Noisy Correspondence Unlearning) 유틸리티.

매 에폭 시작 시:
  1. 현재 모델로 전체 utterance embedding 추출
  2. 화자별 pairwise cosine similarity 계산 후 2-component GMM fit
  3. 결과를 pickle로 저장 (매 에폭 덮어쓰기)

Dataset에서 pair 샘플링 시:
  - 저장된 embedding + GMM으로 clean/noisy 판별
"""

import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from sklearn.mixture import GaussianMixture
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from data.augmentations import SpecNormalization
from data.data_utils import crop_or_pad, load_spk2utt
from data.feature_extractors import SBFbank


# ========================
# Embedding 추출용 Dataset
# ========================
class _SimpleUtteranceDataset(Dataset):
    """파일 경로 목록을 받아서 spectrogram을 반환하는 Dataset."""

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
            return spec.squeeze(0), idx, True
        except Exception:
            dummy_spec = torch.zeros(300, 80)
            return dummy_spec, idx, False


def _collate_fn(batch):
    """가변 길이 spectrogram을 패딩하여 배치로 만든다."""
    specs, indices, valid_flags = zip(*batch)
    max_len = max(s.shape[0] for s in specs)

    padded_specs = []
    for spec in specs:
        if spec.shape[0] < max_len:
            pad_size = max_len - spec.shape[0]
            spec = F.pad(spec, (0, 0, 0, pad_size))
        padded_specs.append(spec)

    return torch.stack(padded_specs, dim=0), torch.tensor(indices), torch.tensor(valid_flags)


# ========================
# Embedding 추출
# ========================
def create_feature_extractor(device="cpu"):
    """Feature extractor (SBFbank) 생성."""
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


def extract_all_embeddings(
    encoder,
    spk2utt: dict,
    data_dir: str,
    device: torch.device,
    sample_rate: int = 16000,
    segment_length: int = 48000,
    batch_size: int = 64,
    num_workers: int = 4,
) -> dict:
    """전체 utterance embedding을 batch 단위로 추출한다.

    Returns:
        embeddings_dict: {file_path: np.ndarray(192,)}
    """
    # 전체 파일 목록 구축
    all_files = []
    for speaker_id, files in spk2utt.items():
        all_files.extend(files)
    all_files_unique = list(dict.fromkeys(all_files))  # 중복 제거, 순서 유지

    feature_extractor = create_feature_extractor(device="cpu")

    dataset = _SimpleUtteranceDataset(
        file_paths=all_files_unique,
        data_dir=data_dir,
        feature_extractor=feature_extractor,
        sample_rate=sample_rate,
        segment_length=segment_length,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_fn,
        pin_memory=True,
    )

    embeddings_dict = {}

    encoder.eval()
    with torch.no_grad():
        for batch_specs, batch_indices, batch_valid in tqdm(dataloader, desc="[NCU] Extracting embeddings"):
            batch_specs = batch_specs.to(device)
            embeddings = encoder(batch_specs)  # [B, 1, 192]
            embeddings = embeddings.squeeze(1).cpu().numpy()  # [B, 192]

            for i, (idx, valid) in enumerate(zip(batch_indices, batch_valid)):
                idx = idx.item()
                if valid:
                    embeddings_dict[all_files_unique[idx]] = embeddings[i]

    return embeddings_dict


# ========================
# Per-Speaker GMM Fit
# ========================
def _compute_pairwise_similarities(embeddings: np.ndarray) -> np.ndarray:
    """화자 내 embedding들의 모든 pairwise cosine similarity를 계산한다."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
    embeddings_norm = embeddings / norms
    sim_matrix = embeddings_norm @ embeddings_norm.T

    n = len(embeddings)
    similarities = []
    for i in range(n):
        for j in range(i + 1, n):
            similarities.append(sim_matrix[i, j])

    return np.array(similarities)


def _fit_gmm(similarities: np.ndarray, n_components: int = 2, random_state: int = 42):
    """Cosine similarity 분포에 2-component GMM을 fitting한다.

    Returns:
        gmm: fitted GaussianMixture
        clean_comp: clean component index (similarity 평균이 높은 쪽)
    """
    X = similarities.reshape(-1, 1)
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        random_state=random_state,
        n_init=5,
        max_iter=300,
    )
    gmm.fit(X)

    means = gmm.means_.flatten()
    clean_comp = int(np.argmax(means))

    return gmm, clean_comp


def fit_per_speaker_gmm(
    spk2utt: dict,
    embeddings_dict: dict,
    min_utterances: int = 3,
) -> dict:
    """화자별로 pairwise similarity 기반 GMM을 fit한다.

    Args:
        spk2utt: {speaker_id: [file_paths]}
        embeddings_dict: {file_path: np.ndarray(192,)}
        min_utterances: GMM fit에 필요한 최소 발화 수

    Returns:
        gmm_dict: {speaker_id: {"gmm": GaussianMixture, "clean_comp": int}}
    """
    gmm_dict = {}

    for speaker_id, files in tqdm(spk2utt.items(), desc="[NCU] Fitting per-speaker GMMs"):
        # 유효한 embedding만 수집
        valid_embeddings = []
        for f in files:
            if f in embeddings_dict:
                valid_embeddings.append(embeddings_dict[f])

        if len(valid_embeddings) < min_utterances:
            # 발화가 너무 적으면 GMM fit 불가 → 모든 pair를 clean으로 처리
            gmm_dict[speaker_id] = None
            continue

        embeddings_arr = np.array(valid_embeddings)
        similarities = _compute_pairwise_similarities(embeddings_arr)

        if len(similarities) < 3:
            gmm_dict[speaker_id] = None
            continue

        # similarity 분산이 거의 없으면 GMM fit이 무의미
        if np.std(similarities) < 1e-6:
            gmm_dict[speaker_id] = None
            continue

        try:
            gmm, clean_comp = _fit_gmm(similarities)
            gmm_dict[speaker_id] = {"gmm": gmm, "clean_comp": clean_comp}
        except Exception:
            gmm_dict[speaker_id] = None

    return gmm_dict


# ========================
# Pair Clean/Noisy 판별
# ========================
def get_pair_p_clean(gmm_entry, emb1: np.ndarray, emb2: np.ndarray) -> float:
    """두 embedding의 cosine similarity를 GMM에 넣어 clean 확률을 반환한다.

    Args:
        gmm_entry: {"gmm": GaussianMixture, "clean_comp": int} 또는 None
        emb1, emb2: 192-dim embeddings

    Returns:
        p_clean: clean일 확률 [0, 1]
    """
    if gmm_entry is None:
        # GMM fit이 안 된 화자 → clean으로 처리
        return 1.0

    gmm = gmm_entry["gmm"]
    clean_comp = gmm_entry["clean_comp"]

    # cosine similarity 계산
    norm1 = np.linalg.norm(emb1) + 1e-8
    norm2 = np.linalg.norm(emb2) + 1e-8
    sim = np.dot(emb1, emb2) / (norm1 * norm2)

    p_clean = gmm.predict_proba([[sim]])[0, clean_comp]
    return float(p_clean)


def check_pair_clean(gmm_entry, emb1: np.ndarray, emb2: np.ndarray, threshold: float = 0.5) -> bool:
    """두 embedding의 cosine similarity를 GMM에 넣어 clean 여부를 판별한다.

    Args:
        gmm_entry: {"gmm": GaussianMixture, "clean_comp": int} 또는 None
        emb1, emb2: 192-dim embeddings
        threshold: clean으로 판별할 p_clean 기준값

    Returns:
        True if clean, False if noisy
    """
    return get_pair_p_clean(gmm_entry, emb1, emb2) > threshold


# ========================
# Pickle 저장/로드
# ========================
def save_ncu_labels(path: str, embeddings_dict: dict, gmm_dict: dict):
    """NCU labels를 pickle로 저장한다 (매 에폭 덮어쓰기)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "embeddings": embeddings_dict,
        "gmms": gmm_dict,
    }

    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_ncu_labels(path: str):
    """Pickle에서 NCU labels를 로드한다.

    Returns:
        embeddings_dict: {file_path: np.ndarray(192,)}
        gmm_dict: {speaker_id: {"gmm": GaussianMixture, "clean_comp": int} or None}
    """
    with open(path, "rb") as f:
        data = pickle.load(f)

    return data["embeddings"], data["gmms"]
