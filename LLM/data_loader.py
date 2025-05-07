import torch
from torch.utils.data import Dataset, DataLoader

def get_data_loader_BERT(config, data, shuffle = False, drop_last = False, batch_size = None):
    if batch_size == None:
        batch = min(config.batch_size, len(data))
    else:
        batch = min(batch_size, len(data))
    dataset = BERTDataset(data, config)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=config.num_workers,
        collate_fn=dataset.collate_fn,
        drop_last=drop_last)

    return data_loader

class BERTDataset(Dataset):    
    def __init__(self, data, config):
        self.data = data
        self.config = config

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (self.data[idx], idx)

    def collate_fn(self, data):

        batch_instance = {'input': [],'ids': [], 'mask': []} 
        batch_label = []
        batch_idx = []

        batch_label = torch.tensor([item[0]['relation'] for item in data])
        batch_instance['ids'] = torch.tensor([item[0]['ids'] for item in data])
        batch_instance['mask'] = torch.tensor([item[0]['mask'] for item in data])
        batch_instance['input'] = [item[0]['input'] for item in data]

        batch_idx = torch.tensor([item[1] for item in data])
        
        return batch_instance, batch_label, batch_idx
    
