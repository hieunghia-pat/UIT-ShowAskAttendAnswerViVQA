import json
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import config
# from metric_utils.tracker import Tracker
from model.asaa import ASAA
from data_utils.vivqa import ViVQA, collate_fn
from metric_utils.metrics import Metrics

import re
import os
from tqdm import tqdm

metrics = Metrics()

def main():

    cudnn.benchmark = True

    test_dataset = ViVQA(config.json_test_path, config.preprocessed_path)

    if len(os.listdir("saved_models")) == 0:
        raise Exception("No checkpoint found")

    for checkpoint_name in os.listdir("saved_models"):
        if re.match(r'model_best_+', checkpoint_name):
            checkpoint = os.path.join("saved_models", checkpoint_name)

            print(f"Evaluating {checkpoint}")

            saved_model = torch.load(checkpoint)
            test_dataset.vocab = saved_model["vocab"]
            metrics.vocab = saved_model["vocab"]

            net = nn.DataParallel(ASAA(len(test_dataset.vocab.stoi), len(test_dataset.vocab.output_cats))).cuda()
            net.load_state_dict(saved_model["weights"])

            loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=config.batch_size,
                shuffle=True,
                pin_memory=True,
                num_workers=config.data_workers,
                collate_fn=collate_fn)

            tq = tqdm(loader, ncols=0)

            results = {
                "predicted": [],
                "gt": []
            }
            for v, q, a, q_len in tq:
                v = v.cuda()
                q = q.cuda()
                a = a.cuda()
                q_len = q_len.cuda()

                out = net(v, q, q_len)
                predicted_answers = test_dataset.vocab._decode_answer(out.cpu())
                gt_answers = test_dataset.vocab._decode_answer(a.cpu())
                for predicted_answer, gt_answer in zip(predicted_answers, gt_answers):
                    results["predicted"].append(predicted_answer)
                    results["gt"].append(gt_answer)

            json.dump(results, open(f"{checkpoint_name.split('.')[0]}_results.json", "w+"), ensure_ascii=False)

            # results = metrics.evaluate(net, test_dataset, Tracker(), prefix="Evaluation")

 
            # print(f"Accuracy: {results['accuracy']} - Precision: {results['precision']} - Recall: {results['recall']} - F1 score {results['F1']}")
            # print("+"*13)

if __name__ == '__main__':
    main()