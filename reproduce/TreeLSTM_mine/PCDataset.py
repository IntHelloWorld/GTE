from pathlib import Path

from dgl import load_graphs
from dgl.data import DGLDataset
from dgl.data.utils import load_info


class PCDataset(DGLDataset):
    def __init__(self, raw_dir):
        dataset_name = Path(raw_dir).name
        super(PCDataset, self).__init__(name=dataset_name, url=None, raw_dir=raw_dir, force_reload=False, verbose=False)

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

            else:
                pass
        
        example = False
        if example:
            self.graphs = self.graphs[:1000]
            self.labels = self.labels[:1000]
            self.n_layers = self.n_layers[:1000]
            self.node_types = self.node_types[:1000]

    def __getitem__(self, idx):
        return (self.graphs[idx], self.labels[idx], self.n_layers[idx], self.node_types[idx])

    def __len__(self):
        return len(self.graphs)
