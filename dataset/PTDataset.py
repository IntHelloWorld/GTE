from pathlib import Path

from dgl import load_graphs
from dgl.data import DGLDataset
from dgl.data.utils import load_info


class PTDataset(DGLDataset):
    def __init__(self, raw_dir):
        dataset_name = Path(raw_dir).name
        super(PTDataset, self).__init__(name=dataset_name, url=None, raw_dir=raw_dir, force_reload=False, verbose=False)

    def process(self):
        dataset_dir = Path(self.raw_dir)
        for f in dataset_dir.iterdir():
            if f.suffix == ".bin":
                self.graphs, label_dict = load_graphs(str(f))
                self.labels = label_dict["labels"]

            elif f.suffix == ".pkl":
                info_dict = load_info(str(f))
                self.n_layers = info_dict["n_layers"]
                self.node_types = info_dict["node_types"]
                self.p_samples = info_dict["p_samples"]

            else:
                pass

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx], self.n_layers[idx], self.node_types[idx], self.p_samples[idx]

    def __len__(self):
        return len(self.graphs)
