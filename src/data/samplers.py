import pickle
import random
from typing import Optional

import numpy as np
from torch.utils.data import Sampler

from data.data_utils import read_voxceleb_pairs_txt, load_spk2utt


class ClusterBatchSampler(Sampler):
    def __init__(
        self,
        cluster_dict_path: str,
        batch_size: int,
        steps_per_epoch: int,
        hard_ratio: float,
        allow_repeat_speakers: bool = False,
        allow_multi_cluster: bool = False,
        cluster_sims_path: Optional[str] = None,
        sample_unique_rest: bool = False,
    ):
        assert not (
            allow_multi_cluster and allow_repeat_speakers
        ), "Choose one: allow_repeat_speakers or allow_multi_cluster"
        with open(cluster_dict_path, "rb") as file:
            self.cluster2spk = pickle.load(file)

        self.top_similar_clusters = None
        if cluster_sims_path:
            with open(cluster_sims_path, "rb") as file:
                self.top_similar_clusters = pickle.load(file)
            print("CHOOSING FROM SIMILAR CLUSTERS")

        self.all_spk_ids = set()

        for _, s_ids in self.cluster2spk.items():
            self.all_spk_ids.update(s_ids)

        self.all_spk_ids = np.asarray(list(self.all_spk_ids))
        print(f"NUM_TRAINING_SPEAKERS: {len(self.all_spk_ids)}")

        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.hard_per_batch = int(hard_ratio * batch_size)
        self.cluster_ids = list(self.cluster2spk.keys())
        self.allow_repeat_speakers = allow_repeat_speakers
        self.allow_multi_cluster = allow_multi_cluster
        self.sample_unique_rest = sample_unique_rest

        if allow_repeat_speakers:
            print("Allowing repeating speakers within batch!")
        elif allow_multi_cluster:
            print("Allowing multiple clusters within batch!")
        else:
            print(
                "If clusters are much smaller than hard_ratio * batch_size, "
                "hard_ratio will not work properly, consider choosing allow_repeat_speakers "
                "or allow_multi_cluster."
            )

    def __iter__(self):
        for _ in range(self.steps_per_epoch):
            # Randomly choose a cluster
            remaining_batch_clusters = self.cluster_ids[:]
            cluster_speakers = []

            chosen_cluster = random.choice(remaining_batch_clusters)
            remaining_batch_clusters.remove(chosen_cluster)
            cluster_speakers.extend(self.cluster2spk[chosen_cluster])

            if self.top_similar_clusters:
                remaining_similar_clusters = self.top_similar_clusters[chosen_cluster][:]

            if not (self.allow_repeat_speakers or self.allow_multi_cluster):
                # if there is less than desired hard speakers per batch, take less
                num_hard = min(len(cluster_speakers), self.hard_per_batch)
                num_rest = self.batch_size - num_hard
                hard_speakers = np.random.choice(cluster_speakers, size=num_hard, replace=False)
            else:
                num_hard = self.hard_per_batch
                if self.allow_repeat_speakers:
                    hard_speakers = np.random.choice(cluster_speakers, size=num_hard, replace=True)
                elif self.allow_multi_cluster:
                    while len(cluster_speakers) < num_hard:
                        if self.top_similar_clusters:
                            chosen_cluster = remaining_similar_clusters[0]  # take the most similar available cluster
                            remaining_similar_clusters.remove(chosen_cluster)
                        else:
                            chosen_cluster = random.choice(remaining_batch_clusters)
                            remaining_batch_clusters.remove(chosen_cluster)
                        cluster_speakers.extend(self.cluster2spk[chosen_cluster])
                    hard_speakers = np.random.choice(cluster_speakers, size=num_hard, replace=False)

            num_rest = self.batch_size - num_hard
            if self.sample_unique_rest:
                remaining_speakers = list(set(self.all_spk_ids[:]) - set(hard_speakers))
                rest_speakers = np.random.choice(remaining_speakers, size=num_rest, replace=False)
            else:
                rest_speakers = np.random.choice(self.all_spk_ids, size=num_rest, replace=False)
            batch = np.concatenate((hard_speakers, rest_speakers))
            np.random.shuffle(batch)

            yield list(batch)

    def __len__(self):
        return self.steps_per_epoch


class UniqueBatchSampler(Sampler):
    def __init__(self, spk2utt_file_path, valid_spk_ids, batch_size, steps_per_epoch):
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch

        spk2utt = load_spk2utt(spk2utt_file_path)

        # remove valid indices from spk2utt
        for id in valid_spk_ids:
            spk2utt.pop(id)

        self.train_speaker_ids = list(spk2utt.keys())
        print(f"NUM_TRAINING_SPEAKERS: {len(self.train_speaker_ids)}")

    def __iter__(self):
        for _ in range(self.steps_per_epoch):
            batch = np.random.choice(self.train_speaker_ids, size=self.batch_size, replace=False)
            yield list(batch)

    def __len__(self):
        return self.steps_per_epoch


