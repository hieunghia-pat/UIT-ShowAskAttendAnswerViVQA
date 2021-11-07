import torch
from torch.utils import data
from torch.utils.data.dataset import random_split
from data_utils.utils import preprocess_question, preprocess_answer
from data_utils.vocab import Vocab
import h5py
import json
import config

class ViVQA(data.Dataset):
    """ VQA dataset, open-ended """
    def __init__(self, json_path, image_features_path):
        super(ViVQA, self).__init__()
        with open(json_path, 'r') as fd:
            json_data = json.load(fd)

        # vocab
        self.vocab = Vocab(json_path)

        # q and a
        self.questions, self.answers, self.image_ids = self.load_json(json_data)

        # v
        self.image_features_path = image_features_path
        self.image_id_to_index = self._create_image_id_to_index()

    @property
    def max_question_length(self):
        if not hasattr(self, '_max_length'):
            self._max_length = max(map(len, self.questions)) + 2
        return self._max_length + 2

    @property
    def output_cats(self):
        return list(set(self.answers))

    @property
    def num_tokens(self):
        return len(self.vocab.stoi)

    def load_json(self, json_data):
        questions = []
        answers = []
        image_ids = []
        for ann in json_data["annotations"]:
            questions.append(preprocess_question(ann["question"]))
            answers.append(preprocess_answer(ann["answer"]))
            image_ids.append(ann["img_id"])

        return questions, answers, image_ids

    def _create_image_id_to_index(self):
        """ Create a mapping from a COCO image id into the corresponding index into the h5 file """
        with h5py.File(self.image_features_path, 'r') as features_file:
            coco_ids = features_file['ids'][()]
        coco_id_to_index = {id: i for i, id in enumerate(coco_ids)}
        return coco_id_to_index

    def _encode_question(self, question):
        """ Turn a question into a vector of indices and a question length """
        vec = torch.ones(self.max_question_length).long() * self.vocab.stoi["<pad>"]
        for i, token in enumerate(question):
            vec[i] = self.vocab.stoi[token]
        return vec, len(question)

    def _encode_answer(self, answer):
        """ Turn an answer into a vector """
        # answer vec will be a vector of answer counts to determine which answers will contribute to the loss.
        # this should be multiplied with 0.1 * negative log-likelihoods that a model produces and then summed up
        # to get the loss that is weighted by how many humans gave that answer
        answer_vec = torch.zeros(len(self.output_cats))
        answer_vec[self.output_cats.index(answer)] = 1

        return answer_vec

    def _load_image(self, image_id):
        """ Load an image """
        if not hasattr(self, 'features_file'):
            # Loading the h5 file has to be done here and not in __init__ because when the DataLoader
            # forks for multiple works, every child would use the same file object and fail
            # Having multiple readers using different file objects is fine though, so we just init in here.
            self.features_file = h5py.File(self.image_features_path, 'r')
        index = self.image_id_to_index[image_id]
        dataset = self.features_file['features']
        img = dataset[index].astype('float32')

        return torch.from_numpy(img)

    def __getitem__(self, idx):
        q, q_length = self._encode_question(self.questions[idx])
        a = self._encode_answer(self.answers[idx])
        image_id = self.image_ids[idx]
        v = self._load_image(image_id)

        return v, q, a, q_length

    def __len__(self):
        return len(self.questions)

def collate_fn(batch):
    # put question lengths in descending order so that we can use packed sequences later
    batch.sort(key=lambda x: x[-1], reverse=True)

    return data.dataloader.default_collate(batch)


def get_loader(dataset):
    """ Returns a data loader for the desired split """

    train_len = int(len(dataset) * 0.8)
    val_len = len(dataset) - train_len

    trainset, valset = random_split(dataset, [train_len, val_len], generator=torch.Generator().manual_seed(13))
    
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=config.batch_size,
        shuffle=True,  # only shuffle the data in training
        pin_memory=True,
        num_workers=config.data_workers,
        collate_fn=collate_fn,
    )

    val_loader = torch.utils.data.DataLoader(
        valset,
        batch_size=config.batch_size,
        shuffle=True,  # only shuffle the data in training
        pin_memory=True,
        num_workers=config.data_workers,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader