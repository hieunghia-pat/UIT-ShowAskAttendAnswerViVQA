import torch
from torch.utils import data
from torchvision import transforms
from data_utils.vivqa import ViVQA
import config
import re

def collate_fn(batch):
    # put question lengths in descending order so that we can use packed sequences later
    batch.sort(key=lambda x: x[-1], reverse=True)
    return data.dataloader.default_collate(batch)


def get_loader(train=False, test=False):
    """ Returns a data loader for the desired split """
    assert train + test == 1, 'need to set exactly one of {train, test} to True'
    json_path = config.json_train_path if train else config.json_test_path
    split = ViVQA(
        json_path,
        config.preprocessed_path
    )
    loader = torch.utils.data.DataLoader(
        split,
        batch_size=config.batch_size,
        shuffle=train,  # only shuffle the data in training
        pin_memory=True,
        num_workers=config.data_workers,
        collate_fn=collate_fn,
    )
    return loader

def preprocess_question(question):
    question = question.lower().strip().split()
    return ["<sos>"] + question + ["<eos>"]

def preprocess_answer(answer):
    answer = re.sub(" ", "_", answer.strip()).lower()
    return answer

def get_transform(target_size, central_fraction=1.0):
    return transforms.Compose([
        transforms.Scale(int(target_size / central_fraction)),
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])