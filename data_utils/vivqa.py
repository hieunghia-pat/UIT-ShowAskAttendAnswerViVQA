import torch
from torch.utils import data
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
        self.vocab = Vocab(json_data, specials=["<unk>", "<sos>", "<eos>"])

        # q and a
        self.questions, self.answers, self.image_ids = self.load_json(json_data)

        # v
        self.image_features_path = image_features_path
        self.image_id_to_index = self._create_image_id_to_index()

    @property
    def max_question_length(self):
        if not hasattr(self, '_max_length'):
            self._max_length = max(map(len, self.question))
        return self._max_length

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
        answer_vec = torch.zeros(len(self.answers))
        answer_vec[self.answers.index(answer)] = 1

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
        return len(self.anns)

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