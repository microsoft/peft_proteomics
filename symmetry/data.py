# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import torch
import typing as T
import pandas as pd
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer
from omegaconf import OmegaConf
from logutil import BaseLogger

def SYMM_MAP(task: str = "homooligomer"):
    if task == "homooligomer":
        return {
            "C1":0, "C2":1,"C3":2,"C4":3,"C5":4,"C6":5,"C7":6,"C8":6,"C9":6,"C10":7,"C11":7,"C12":7,"C13":7,"C14":7,
            "C15":7,"C16":7,"C17":7,"D2":8,"D3":9,"D4":10,"D5":11,"D6":12,"D7":12,"D8":12,"D9":12,"D10":12,"D11":12,"D12":12,
            "H":13,"O":14,"T":15, "I":16
            }
    elif task == "QUEEN":
        return {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4, "6": 5, "7": 6, "8": 7, "10": 8, "12": 9, "14": 10, "24": 11}
    else:
        raise ValueError(f"Unrecognized task {task}")

def init_data_dictionary(
    meta_data_file: str,
) -> pd.DataFrame:
    meta_df = pd.read_csv(meta_data_file, header=0, sep='\t', dtype='str')
    
    train_df = meta_df[meta_df["split"] == "train"]
    valid_df = meta_df[meta_df["split"] == "validation"]
    test_df = meta_df[meta_df["split"] == "test"]
    
    return train_df, valid_df, test_df


def get_token_datasets(
    params: OmegaConf,
) -> T.Tuple[Dataset, Dataset]:
    
    train_dict, valid_dict, _ = init_data_dictionary(params.META_DATA_FILE)
    
    train_dataset = ESMTokenizerDataset(train_dict, params.max_crop, params.esm_pretrained, params.n_classes)
    valid_dataset = ESMTokenizerDataset(valid_dict, params.max_crop, params.esm_pretrained, params.n_classes)
    
    return train_dataset, valid_dataset

def get_datasets(
    params: OmegaConf,
    logg: BaseLogger,
    type: str = "token"
) -> T.Tuple[Dataset, Dataset]:
    """
    Convenience wrapper function for get_token_datasets and get_embedding_datasets
    """

    if type == "token":
        return get_token_datasets(params, logg)
    else: raise ValueError(f"Unrecognized type {type}")

def get_dataloaders(
    train_dataset: Dataset, 
    valid_dataset: Dataset,
    params: OmegaConf, 
    logg: BaseLogger
) -> T.Tuple[Sampler, Sampler, DataLoader, DataLoader]:
    
    rank = params.rank
    world_size = params.world_size
    
    if params.epoch_size % world_size == 0:
        n_train = params.epoch_size
    else:
        n_train = world_size * (params.epoch_size // world_size)
        logg.warning(f"Epoch size {params.epoch_size} not divisible by world size {world_size}, setting n_train to {n_train}")
    
    train_sampler = DistributedMySampler(train_dataset, [], logg, num_example_per_epoch=int(n_train/world_size)*world_size,
                                            num_replicas=world_size, rank=rank, shuffle=True)
    if hasattr(params, "valid_size"):
        valid_dataset = torch.utils.data.Subset(valid_dataset, list(range(params.valid_size)))
    valid_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    def loader_collate_fn(batch_list):
        feats = []
        labels = []

        for (f, l) in batch_list:
            feats.append(f)
            labels.append(l)

        PAD_VALUE = 0.
        feats = pad_sequence(feats, batch_first = True, padding_value = PAD_VALUE)
        feats = feats[:, :params.max_crop]
        atten_map = (feats != PAD_VALUE).int()
        
        labels = torch.stack(labels)


        return (feats, labels, atten_map) 
    
    train_loader = DataLoader(
            train_dataset, 
            sampler=train_sampler,
            batch_size=params.batch_size,
            num_workers=params.num_workers,
            pin_memory=params.pin_memory,
            collate_fn = loader_collate_fn,
        )
        
    valid_loader = DataLoader(
        valid_dataset, 
        sampler=valid_sampler,
        batch_size=params.batch_size,
        shuffle=False,
        num_workers=params.num_workers,
        pin_memory=params.pin_memory,
        collate_fn = loader_collate_fn,
    )
    
    return train_sampler, valid_sampler, train_loader, valid_loader
    
def one_hot_encode(label: int, num_classes: int):
    """
    Converts a label to a one-hot encoding vector.

    Args:
        label (int): The label to be encoded.
        num_classes (int): The total number of classes.

    Returns:
        one_hot (list): A one-hot encoding vector.
    """
    one_hot = torch.zeros((num_classes))
    one_hot[label] = 1
    return one_hot

def label_list_to_vec(labels: T.List[str], n_classes: int = 2, symm_map: T.Dict[str, int] = SYMM_MAP()):
    label_vecs = []

    for l in labels:
        if l in symm_map:
            homomer_label = symm_map[l] 
        else:
            homomer_label = n_classes - 1
        label_vecs.append(one_hot_encode(homomer_label, num_classes=n_classes))
    
    # address multilabel cases
    if len(label_vecs) > 1:
        label_vecs = torch.stack(label_vecs,dim=0)
        label_vecs = torch.sum(label_vecs,dim=0)
    else:
        label_vecs = label_vecs[0]

    return label_vecs

    
class ESMTokenizerDataset(Dataset):
    def __init__(
            self, 
            data_df: pd.DataFrame,
            max_crop: int = 1024,
            esm_variant: str = "esm2_t36_3B_UR50D",
            n_classes: int = 18,
        ):
        
        self.data_df = data_df
        self.max_crop = max_crop - 2
        self.tokenizer = AutoTokenizer.from_pretrained(f"facebook/{esm_variant}")
        self.n_classes = n_classes
    
    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index: int) -> T.Tuple[torch.Tensor, torch.Tensor]:
        row = self.data_df.iloc[index]
        sequence = row["sequence"]
        sequence = sequence[:self.max_crop]
        
        labels = row["label"].split()
        label = label_list_to_vec(labels, self.n_classes, symm_map = SYMM_MAP())
        
        tokens = self.tokenizer(sequence, return_tensors="pt")["input_ids"].squeeze()
        
        return (tokens, label)

class DistributedMySampler(Sampler):
    def __init__(
            self,
            dataset: Dataset, 
            weights: T.List[float],
            logg: BaseLogger,
            num_example_per_epoch: int = 25600,
            num_replicas: T.Optional[int] = None,
            rank: T.Optional[int] = None, 
            replacement: bool = False, 
            shuffle: bool = False, 
        ):
        
        self.logg = logg
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        
        self.logg.info(f"Num_examples: {num_example_per_epoch}, world_size: {num_replicas}")
        assert num_example_per_epoch % num_replicas == 0

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.total_size = num_example_per_epoch
        self.num_samples = self.total_size // self.num_replicas
        self.rank = rank
        self.epoch = 0
        self.replacement = replacement
        self.weights = weights
        self.shuffle = shuffle

    def __iter__(self):
        indices = torch.arange(len(self.dataset))
        
        # shuffle indices
        if self.shuffle:
            # deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = indices[torch.randperm(len(indices), generator=g)]

        # per each gpu
        indices = indices[self.rank:self.total_size:self.num_replicas]

        return iter(indices.tolist())

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int):
        self.epoch = epoch