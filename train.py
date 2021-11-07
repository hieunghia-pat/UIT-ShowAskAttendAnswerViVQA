import sys
import os.path
from statistics import mean

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from tqdm import tqdm

import config
from model.asaa import ASAA
from data_utils.vivqa import ViVQA, get_loader
from metric_utils.metrics import Metrics
from metric_utils.tracker import Tracker


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
        answ = []
        accs = []
        pres = []
        recs = []
        f1s = []

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
            nll = -log_softmax(out)
            loss = (nll * a / 10).sum(dim=1).mean()
            scores = metrics.get_scores(out.cpu(), a.cpu())

            if train:
                global total_iterations
                update_learning_rate(optimizer, total_iterations)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_iterations += 1
            else:
                # store information about evaluation of this minibatch
                _, answer = out.cpu().max(dim=1)
                answ.append(answer.view(-1))
                accs.append(scores["accuracy"])
                pres.append(scores["precision"])
                recs.append(scores["recall"])
                f1s.append(scores["F1"])

            loss_tracker.append(loss.item())
            acc_tracker.append(scores["accuracy"])
            pre_tracker.append(scores["precision"])
            rec_tracker.append(scores["recall"])
            f1_tracker.append(scores["F1"])
            fmt = '{:.4f}'.format
            tq.set_postfix(loss=fmt(loss_tracker.mean.value), accuracy=fmt(acc_tracker.mean.value), 
                            precision=fmt(pre_tracker.mean.value), recall=fmt(rec_tracker.mean.value), f1=fmt(f1_tracker.mean.value))

        if not train:
            answ = list(torch.cat(answ, dim=0))
            accs = list(torch.cat(accs, dim=0))

            return {
                "answers": answ,
                "accuracy": mean(accs),
                "precision": mean(pres),
                "recall": mean(recs),
                "F1": mean(f1s)
            }


def main():

    cudnn.benchmark = True

    dataset = ViVQA(config.json_train_path, config.preprocessed_path)

    net = nn.DataParallel(ASAA(dataset.num_tokens, len(dataset.output_cats))).cuda()
    optimizer = optim.Adam([p for p in net.parameters() if p.requires_grad])

    tracker = Tracker()
    config_as_dict = {k: v for k, v in vars(config).items() if not k.startswith('__')}

    max_f1 = 0 # for saving the best model

    folds = get_loader(dataset)
    k_fold = len(folds) - 1

    for e in range(config.epochs):
        for i in range(k_fold):
            tmp = folds[i]
            folds[i] = folds[i-1]
            folds[i-1] = tmp
        run(net, folds[:-1], optimizer, tracker, train=True, prefix='Training', epoch=e)
        returned = run(net, [folds[-1]], optimizer, tracker, train=False, prefix='Validation', epoch=e)

        results = {
            'tracker': tracker.to_dict(),
            'config': config_as_dict,
            'weights': net.state_dict(),
            'eval': {
                'answer': returned["answer"],
                'accuracy': returned["accuracy"],
                "precision": returned["precision"],
                "recall": returned["recall"],
                "f1": returned["F1"]
            },
            'vocab': dataset.vocab,
        }
    
        torch.save(results, config.model_checkpoint)
        if returned["F1"] > max_f1:
            max_f1 = returned["F1"]
            torch.save(results, config.best_model_checkpoint)


if __name__ == '__main__':
    main()
