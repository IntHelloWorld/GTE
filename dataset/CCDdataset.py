from pathlib import Path

from dgl import load_graphs
from dgl.data import DGLDataset
from dgl.data.utils import load_info


class CCDdataset(DGLDataset):
    def __init__(self, raw_dir, graphs_dir):
        dataset_name = Path(raw_dir).name
        self.graphs_dir = Path(graphs_dir)
        super(CCDdataset, self).__init__(name=dataset_name, url=None, raw_dir=raw_dir, force_reload=False, verbose=False)

    def process(self):
        """load graphs"""
        for f in self.graphs_dir.iterdir():
            if f.suffix == ".bin":
                graphs, _ = load_graphs(str(f))
            elif f.suffix == ".pkl":
                info_dict = load_info(str(f))
                n_layers = info_dict["n_layers"]
                node_types = info_dict["node_types"]
                idxs = info_dict["idx"]
            else:
                pass
        all_graphs = {}
        for i in range(len(idxs)):
            all_graphs[idxs[i]] = {"g":graphs[i], "n_layer":n_layers[i], "type":node_types[i]}

        self.samples = []
        """load dataset"""
        with open(self.raw_dir, 'r') as f:
            for line in f:
                idx1, idx2, sign = line.strip().split('\t')
                label = 1 if sign == "1" else -1
                g1 = (all_graphs[idx1]["g"], all_graphs[idx1]["n_layer"], all_graphs[idx1]["type"])
                g2 = (all_graphs[idx2]["g"], all_graphs[idx2]["n_layer"], all_graphs[idx2]["type"])
                sample = (g1, g2, label)
                self.samples.append(sample)


    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)
