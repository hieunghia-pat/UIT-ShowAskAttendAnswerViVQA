import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import config
from metric_utils.tracker import Tracker
from model.asaa import ASAA
from data_utils.vivqa import ViVQA
from metric_utils.metrics import Metrics

import re
import os

metrics = Metrics()

def main():

    cudnn.benchmark = True

    train_dataset = ViVQA(config.json_train_path, config.preprocessed_path)
    test_dataset = ViVQA(config.json_train_path, config.preprocessed_path)

    net = nn.DataParallel(ASAA(train_dataset.num_tokens, len(train_dataset.output_cats))).cuda()

    if len(os.listdir("saved_model")) == 0:
        raise Exception("No checkpoint found")
    
    for checkpoint in os.listdir("saved_model"):
        if re.match(r'model_best_+', checkpoint):
            checkpoint = os.path.join("saved_model", checkpoint)

            print(f"Evaluating {checkpoint}")

            saved_model = torch.load(checkpoint)
            net.load_state_dict(saved_model["weights"])
            train_dataset.vocab = saved_model["vocab"]
            test_dataset.vocab = saved_model["vocab"]
            
            results = metrics.evaluate(net, test_dataset, train_dataset, Tracker(), prefix="Evaluation")

            print(f"Accuracy: {results['accuracy']} - Precision: {results['precision']} - Recall: {results['recall']} - F1 score {results['F1']}")
            print("+"*13)

if __name__ == '__main__':
    main()
