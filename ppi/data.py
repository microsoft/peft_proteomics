# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import torch
import typing as T
import pandas as pd
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
from Bio import SeqIO
from transformers import AutoTokenizer
from omegaconf import OmegaConf
from logutil import BaseLogger

def init_data_dictionary(
    meta_data_file: str,
    _: str,
) -> pd.DataFrame:
    meta_df = pd.read_csv(meta_data_file, header=0, sep='\t', dtype='str')
    
    train_df = meta_df[meta_df["split"] == "train"]
    valid_df = meta_df[meta_df["split"] == "valid"]
    test_df = meta_df[meta_df["split"] == "test"]
    
    return train_df, valid_df, test_df

def get_token_datasets(
    params: OmegaConf
) -> T.Tuple[Dataset, Dataset]:
    
    train_dict, valid_dict, _ = init_data_dictionary(params.META_DATA_FILE, params.SPLITS_FILE)
    
    train_dataset = ESMTokenizerDataset(train_dict, params.TRAIN_SEQUENCES_FILE, params.max_crop, params.esm_pretrained)
    valid_dataset = ESMTokenizerDataset(valid_dict, params.VALID_SEQUENCES_FILE, params.max_crop, params.esm_pretrained)
    
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
        feats0 = []
        feats1 = []
        labels = []

        for (f, l) in batch_list:
            feats0.append(f[0])
            feats1.append(f[1])
            labels.append(l)

        PAD_VALUE = 0.
        feats0 = pad_sequence(feats0, batch_first = True, padding_value = PAD_VALUE)
        feats0 = feats0[:, :params.max_crop]
        attn_map0 = (feats0 != PAD_VALUE).int()
        feats1 = pad_sequence(feats1, batch_first = True, padding_value = PAD_VALUE)
        feats1 = feats1[:, :params.max_crop]
        attn_map1 = (feats1 != PAD_VALUE).int()
        
        labels = torch.stack(labels)


        return ((feats0, feats1), labels, (attn_map0, attn_map1)) 
    
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
    
class ESMTokenizerDataset(Dataset):
    def __init__(
            self, 
            data_df: pd.DataFrame,
            fasta_file: str,
            max_crop: int = 1024,
            esm_variant: str = "esm2_t36_3B_UR50D"
        ):
        
        self.data_df = data_df
        self.fasta_dictionary = SeqIO.to_dict(SeqIO.parse(fasta_file, "fasta"))
        self.max_crop = max_crop - 2
        self.tokenizer = AutoTokenizer.from_pretrained(f"facebook/{esm_variant}")
    
    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index: int) -> T.Tuple[torch.Tensor, torch.Tensor]:
        row = self.data_df.iloc[index]
        s0, s1 = row["s0"], row["s1"]
        label = torch.tensor([int(row["label"])])

        s0 = s0[:self.max_crop]
        s1 = s1[:self.max_crop]
        
        tok0 = self.tokenizer(s0, return_tensors="pt")["input_ids"].squeeze()
        tok1 = self.tokenizer(s1, return_tensors="pt")["input_ids"].squeeze()
        
        return ((tok0, tok1), label)

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