import torch
import pandas as pd
import click
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from loguru import logger


class CriteoParquetDataset(Dataset):
    def __init__(self, file_name: str):
        df = pd.read_parquet(file_name)
        self.total_rows = len(df)
        self.label_tensor = torch.from_numpy(df["labels"].values).to(torch.float32)
        dense_columns = [f for f in df.columns if f.startswith("DENSE")]
        sparse_columns = [f for f in df.columns if f.startswith("SPARSE")]
        self.dense_tensor = torch.from_numpy(df[dense_columns].values)
        self.sparse_tensor = torch.from_numpy(df[sparse_columns].values)

    def __len__(self):
        return self.total_rows

    def __getitem__(self, idx):
        return self.label_tensor[idx], self.dense_tensor[idx], \
        self.sparse_tensor[idx]


@click.command()
@click.option('--file_path', type=click.Path(exists=True),
              help='Path to the parquet file')
def process_file(file_path):
    """
    Process the file specified by --file_path.
    """
    logger.info("Reading the parquet file {}...".format(file_path))

    dataset = CriteoParquetDataset(file_path)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    for labels, dense, sparse in data_loader:
        logger.info("Labels: {}".format(labels))
        logger.info("Dense: {}".format(dense))
        logger.info("Sparse: {}".format(sparse))

        logger.info("Labels size and dtype: {}, {}".format(labels.size(), labels.dtype))
        logger.info("Dense size and dtype: {}, {}".format(dense.size(), dense.dtype))
        logger.info("Sparse size and dtype: {}, {}".format(sparse.size(), sparse.dtype))
        break


if __name__ == "__main__":
    process_file()
