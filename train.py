import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from tqdm import tqdm

import numpy as np

import config
from data_utils.vocab import Vocab
from model.asaa import ASAA
from data_utils.vivqa import ViVQA, get_loader
from metric_utils.metrics import Metrics
from metric_utils.tracker import Tracker

import os


def update_learning_rate(optimizer, iteration):
    lr = config.initial_lr * 0.5**(float(iteration) / config.lr_halflife)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


total_iterations = 0
metrics = Metrics()

def run(net, loaders, optimizer, tracker, train=False, prefix='', epoch=0):
    """ Run an epoch over the given loader """
    if train:
        net.train()
        tracker_class, tracker_params = tracker.MovingMeanMonitor, {'momentum': 0.99}
    else:
        net.eval()
        tracker_class, tracker_params = tracker.MeanMonitor, {}

    for loader in loaders:
        tq = tqdm(loader, desc='Epoch {:03d} - {} - Fold {}'.format(epoch, prefix, loaders.index(loader)+1), ncols=0)
        loss_tracker = tracker.track('{}_loss'.format(prefix), tracker_class(**tracker_params))
        acc_tracker = tracker.track('{}_accuracy'.format(prefix), tracker_class(**tracker_params))
        pre_tracker = tracker.track('{}_precision'.format(prefix), tracker_class(**tracker_params))
        rec_tracker = tracker.track('{}_recall'.format(prefix), tracker_class(**tracker_params))
        f1_tracker = tracker.track('{}_F1'.format(prefix), tracker_class(**tracker_params))

        log_softmax = nn.LogSoftmax(dim=-1).cuda()
        for v, q, a, q_len in tq:
            v = v.cuda()
            q = q.cuda()
            a = a.cuda()
            q_len = q_len.cuda()

            out = net(v, q, q_len)
            scores = metrics.get_scores(out.cpu(), a.cpu())

            if train:
                global total_iterations
                update_learning_rate(optimizer, total_iterations)

                optimizer.zero_grad()
                nll = -log_softmax(out)
                loss = (nll * a / 10).sum(dim=1).mean()
                loss.backward()
                optimizer.step()

                total_iterations += 1
            else:
                loss = np.array(0)

            loss_tracker.append(loss.item())
            acc_tracker.append(scores["accuracy"])
            pre_tracker.append(scores["precision"])
            rec_tracker.append(scores["recall"])
            f1_tracker.append(scores["F1"])
            fmt = '{:.4f}'.format
            tq.set_postfix(loss=fmt(loss_tracker.mean.value), accuracy=fmt(acc_tracker.mean.value), 
                            precision=fmt(pre_tracker.mean.value), recall=fmt(rec_tracker.mean.value), f1=fmt(f1_tracker.mean.value))

        if not train:
            return {
                "accuracy": acc_tracker.mean.value,
                "precision": pre_tracker.mean.value,
                "recall": rec_tracker.mean.value,
                "F1": f1_tracker.mean.value
            }


def main():

    cudnn.benchmark = True

    vocab = Vocab([config.json_train_path, config.json_test_path], 
                            specials=["<pad>", "<sos", "<eos>"])
    train_dataset = ViVQA(config.json_train_path, config.preprocessed_path, vocab)
    test_dataset = ViVQA(config.json_test_path, config.preprocessed_path, vocab)
    folds, test_fold = get_loader(train_dataset, test_dataset)
    k_fold = len(folds) - 1

    for k in range(k_fold):
        print(f"Stage {k+1}:")
        net = nn.DataParallel(ASAA(train_dataset.num_tokens, len(train_dataset.output_cats))).cuda()
        optimizer = optim.Adam([p for p in net.parameters() if p.requires_grad])

        tracker = Tracker()
        config_as_dict = {k: v for k, v in vars(config).items() if not k.startswith('__')}

        max_f1 = 0 # for saving the best model
        f1_test = 0
        for e in range(config.epochs):
            run(net, folds[:-1], optimizer, tracker, train=True, prefix='Training', epoch=e)
            val_returned = run(net, [folds[-1]], optimizer, tracker, train=False, prefix='Validation', epoch=e)
            test_returned = run(net, [test_fold], optimizer, tracker, train=False, prefix='Evaluation', epoch=e)

            print("+"*13)

            results = {
                'tracker': tracker.to_dict(),
                'config': config_as_dict,
                'weights': net.state_dict(),
                'eval': {
                    'accuracy': val_returned["accuracy"],
                    "precision": val_returned["precision"],
                    "recall": val_returned["recall"],
                    "f1-val": val_returned["F1"],
                    "f1-test": test_returned["F1"]

                },
                'vocab': train_dataset.vocab,
            }
        
            torch.save(results, os.path.join(config.model_checkpoint, f"model_last_stage_{k+1}.pth"))
            if val_returned["F1"] > max_f1:
                max_f1 = val_returned["F1"]
                f1_test = test_returned["F1"]
                torch.save(results, os.path.join(config.model_checkpoint, f"model_best_fold_{k+1}.pth"))

        print(f"Finished for stage {k+1}. Best F1 score: {max_f1}. F1 score on test set: {f1_test}")
        print("="*31)

        # change roles of the folds
        for i in range(k_fold):
            tmp = folds[i]
            folds[i] = folds[i-1]
            folds[i-1] = tmp

if __name__ == '__main__':
    main()
