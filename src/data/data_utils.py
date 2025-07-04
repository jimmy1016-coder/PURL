import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import paired_cosine_distances


def crop_or_pad(audio: torch.Tensor, segment_length: int):
    """
    Randomly crops or pads audio to desired length.
    """
    diff = audio.shape[-1] - segment_length

    if diff > 0:
        offset = np.random.randint(low=0, high=diff)
        output = audio[:, offset : offset + segment_length]
    elif diff < 0:
        diff *= -1
        left_pad = np.random.randint(low=0, high=diff)
        right_pad = diff - left_pad
        pad = (left_pad, right_pad)
        output = F.pad(audio, pad)
    else:
        output = audio

    return output


def read_voxceleb_pairs_txt(pairs_txt_file):
    labels = []
    files_1 = []
    files_2 = []

    with open(pairs_txt_file) as f:
        lines = f.read().splitlines()

    for line in lines:
        label, file1, file2 = line.split(" ")
        labels.append(int(label))
        files_1.append(file1)
        files_2.append(file2)

    return labels, files_1, files_2


def write_pairs_output_txt(emb_dict, files_1, files_2, output_txt_path):
    embds_1 = np.vstack([emb_dict[p] for p in files_1])
    embds_2 = np.vstack([emb_dict[p] for p in files_2])
    # cosine similarity
    scores = 1 - paired_cosine_distances(embds_1, embds_2)

    with open(output_txt_path, "w") as outfile:
        for score, f1, f2 in zip(scores, files_1, files_2):
            outfile.write(f"{score} {f1} {f2}\n")


def to_one_hot(k, classes_num):
    target = np.zeros(classes_num)
    target[k] = 1
    return target


def load_utt2spk(file_path):
    with open(file_path) as f:
        rows = [i.strip() for i in f.readlines()]
        result = {i.split()[0]: i.split()[1] for i in rows}
    return result


def load_spk2utt(file_path):
    with open(file_path) as f:
        rows = [i.strip() for i in f.readlines()]
        result = {i.split()[0]: i.split()[1:] for i in rows}
    return result
