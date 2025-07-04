from pathlib import Path
import logging
from collections import defaultdict
import pickle

import numpy as np
from setproctitle import setproctitle
from lightning.pytorch.cli import LightningCLI


logger = logging.getLogger("lightning")


class ClusteringLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--ckpt_name", required=True)


if __name__ == "__main__":
    cli = ClusteringLightningCLI(
        run=False, save_config_kwargs={"overwrite": True}, parser_kwargs={"parser_mode": "omegaconf"}
    )

    output_dir = Path(cli.trainer.logger.save_dir)
    setproctitle(f"{output_dir.name}-clustering")

    ckpt_path = output_dir / "checkpoints" / f"{cli.config.ckpt_name}.ckpt"

    prediction_outputs = cli.trainer.predict(model=cli.model, datamodule=cli.datamodule, ckpt_path=ckpt_path)

    spk_ids = []
    centroids = []

    for spk_id, centroid in prediction_outputs:
        spk_ids.append(spk_id)
        centroids.append(centroid)

    labels = cli.model.cluster(np.asarray(centroids))

    # create a cluster map of structure: "cluster_label": "speaker_id"
    cluster_map = defaultdict(list)

    for sid, label in zip(spk_ids, labels):
        cluster_map[label].append(sid)

    with open(cli.model.cluster_output_file, 'wb') as file:
        pickle.dump(cluster_map, file)
    print(f"Cluster id to speaker id map saved to: {cli.model.cluster_output_file}")