import json
import os

from torch.utils.data import Dataset


class ReverseLangMnliDataset(Dataset):
    """
    Reverse sentence prediction using MNLI dataset
    """

    def __init__(self, file_or_dir, input_transformer=None, label_transformer=None):
        self.label_transformer = label_transformer
        self.input_transformer = input_transformer
        input_file = file_or_dir
        if os.path.isdir(file_or_dir):
            files = os.listdir(file_or_dir)
            assert len(files) == 1, f"Expecting just one file in {file_or_dir}"
            input_file = files[0]
        self._data_x = self._read_file(input_file)

    def _read_file(self, input_file):
        result_x = []
        with open(input_file) as f:
            for l in f:
                data_l = json.loads(l)
                result_x.append(data_l["sentence1"])
                result_x.append(data_l["sentence2"])

        return result_x

    def __len__(self):
        return len(self._data_x)

    def __getitem__(self, index):
        x = self._data_x[index]
        y = self._data_x[index]

        if self.input_transformer:
            x = self.input_transformer(x)

        if self.label_transformer:
            y = self.label_transformer(y)

        return x,y
