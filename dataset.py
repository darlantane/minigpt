import torch
from torch.utils.data import Dataset


class GPTDataset(Dataset):

    def __init__(self, data, block_size):
        """
        data : texte déjà encodé en entiers
        block_size : longueur du contexte
        """

        self.data = torch.tensor(
            data,
            dtype=torch.long
        )

        self.block_size = block_size


    def __len__(self):
        """
        Nombre d'exemples disponibles
        """

        return len(self.data) - self.block_size


    def __getitem__(self, idx):
        """
        Retourne un exemple (X,Y)
        """

        x = self.data[idx : idx + self.block_size]

        y = self.data[idx + 1 : idx + self.block_size + 1]

        return x, y