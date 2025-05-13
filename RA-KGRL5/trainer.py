from matplotlib.pyplot import bar_label
from torch.utils.tensorboard import SummaryWriter
import os
import json
from utils import Log
from torch.utils.data import DataLoader
from ent_init_model import EntInit
from rgcn_model import RGCN
from kge_model import KGEModel
import torch
import torch.nn.functional as F
from collections import defaultdict as ddict
from datasets import KGEEvalDataset


class Trainer(object):
    def __init__(self, args):
        self.args = args

        # writer and logger
        self.name = args.name
        self.writer = SummaryWriter(os.path.join(args.tb_log_dir, self.name))
        self.logger = Log(args.log_dir, self.name).get_logger()
        self.logger.info(json.dumps(vars(args)))

        # state dir
        self.state_path = os.path.join(args.state_dir, self.name)
        if not os.path.exists(self.state_path):
            os.makedirs(self.state_path)

        # models
        self.ent_init = EntInit(args).to(args.gpu)
        self.rgcn = RGCN(args).to(args.gpu)
        self.kge_model = KGEModel(args).to(args.gpu)

    def save_checkpoint(self, step):
        state = {'ent_init': self.ent_init.state_dict(),
                 'rgcn': self.rgcn.state_dict(),
                 'kge_model': self.kge_model.state_dict()}
        # delete previous checkpoint
        for filename in os.listdir(self.state_path):
            if self.name in filename.split('.') and os.path.isfile(os.path.join(self.state_path, filename)):
                os.remove(os.path.join(self.state_path, filename))
        # save checkpoint
        torch.save(state, os.path.join(self.args.state_dir, self.name,
                                       self.name + '.' + str(step) + '.ckpt'))

    def save_model(self, best_step):
        os.rename(os.path.join(self.state_path, self.name + '.' + str(best_step) + '.ckpt'),
                  os.path.join(self.state_path, self.name + '.best'))

    def write_training_loss(self, loss, step):
        self.writer.add_scalar("training/loss", loss, step)

    def write_evaluation_result(self, results, e):
        self.writer.add_scalar("evaluation/mrr", results['mrr'], e)
        self.writer.add_scalar("evaluation/hits10", results['hits@10'], e)
        self.writer.add_scalar("evaluation/hits3", results['hits@3'], e)
        self.writer.add_scalar("evaluation/hits1", results['hits@1'], e)

    def before_test_load(self):
        state = torch.load(os.path.join(self.state_path, self.name + '.best'), map_location=self.args.gpu)
        self.ent_init.load_state_dict(state['ent_init'])
        self.rgcn.load_state_dict(state['rgcn'])
        self.kge_model.load_state_dict(state['kge_model'])

    def get_loss(self, tri, neg_tail_ent, neg_head_ent, ent_emb):
        neg_tail_score = self.kge_model((tri, neg_tail_ent), ent_emb, mode='tail-batch')
        neg_head_score = self.kge_model((tri, neg_head_ent), ent_emb, mode='head-batch')
        neg_score = torch.cat([neg_tail_score, neg_head_score])
        neg_score = (F.softmax(neg_score * self.args.adv_temp, dim=1).detach()
                     * F.logsigmoid(-neg_score)).sum(dim=1)

        pos_score = self.kge_model(tri, ent_emb)
        pos_score = F.logsigmoid(pos_score).squeeze(dim=1)

        positive_sample_loss = - pos_score.mean()
        negative_sample_loss = - neg_score.mean()
        loss = (positive_sample_loss + negative_sample_loss) / 2

        return loss

    def get_ent_emb(self, sup_g_bidir):
        self.ent_init(sup_g_bidir)
        ent_emb = self.rgcn(sup_g_bidir)

        return ent_emb

    # def evaluate(self, ent_emb, eval_dataloader, num_cand='all'):
    #     results = ddict(float)
    #     count = 0

    #     eval_dataloader.dataset.num_cand = num_cand

    #     if num_cand == 'all':
    #         for batch in eval_dataloader:
    #             pos_triple, tail_label, head_label = [b.to(self.args.gpu) for b in batch]
    #             head_idx, rel_idx, tail_idx = pos_triple[:, 0], pos_triple[:, 1], pos_triple[:, 2]

    #             # tail prediction
    #             pred = self.kge_model((pos_triple, None), ent_emb, mode='tail-batch')

    #             b_range = torch.arange(pred.size()[0], device=self.args.gpu)
    #             target_pred = pred[b_range, tail_idx]
    #             pred = torch.where(tail_label.byte(), -torch.ones_like(pred) * 10000000, pred)
    #             pred[b_range, tail_idx] = target_pred

    #             tail_ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True),
    #                                            dim=1, descending=False)[b_range, tail_idx]

    #             # head prediction
    #             pred = self.kge_model((pos_triple, None), ent_emb, mode='head-batch')

    #             b_range = torch.arange(pred.size()[0], device=self.args.gpu)
    #             target_pred = pred[b_range, head_idx]
    #             pred = torch.where(head_label.byte(), -torch.ones_like(pred) * 10000000, pred)
    #             pred[b_range, head_idx] = target_pred

    #             head_ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True),
    #                                            dim=1, descending=False)[b_range, head_idx]

    #             ranks = torch.cat([tail_ranks, head_ranks])
    #             ranks = ranks.float()
    #             count += torch.numel(ranks)
    #             results['mr'] += torch.sum(ranks).item()
    #             results['mrr'] += torch.sum(1.0 / ranks).item()

    #             for k in [1, 3, 10]:
    #                 results['hits@{}'.format(k)] += torch.numel(ranks[ranks <= k])

    #         for k, v in results.items():
    #             results[k] = v / count

    #     else:
    #         for i in range(self.args.num_sample_cand):
    #             for batch in eval_dataloader:
    #                 pos_triple, tail_cand, head_cand = [b.to(self.args.gpu) for b in batch]

    #                 b_range = torch.arange(pos_triple.size()[0], device=self.args.gpu)
    #                 target_idx = torch.zeros(pos_triple.size()[0], device=self.args.gpu, dtype=torch.int64)
    #                 # tail prediction
    #                 pred = self.kge_model((pos_triple, tail_cand), ent_emb, mode='tail-batch')
    #                 tail_ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True),
    #                                                dim=1, descending=False)[b_range, target_idx]
    #                 # head prediction
    #                 pred = self.kge_model((pos_triple, head_cand), ent_emb, mode='head-batch')
    #                 head_ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True),
    #                                                dim=1, descending=False)[b_range, target_idx]

    #                 ranks = torch.cat([tail_ranks, head_ranks])
    #                 ranks = ranks.float()
    #                 count += torch.numel(ranks)
    #                 results['mr'] += torch.sum(ranks).item()
    #                 results['mrr'] += torch.sum(1.0 / ranks).item()

    #                 for k in [1, 5, 10]:
    #                     results['hits@{}'.format(k)] += torch.numel(ranks[ranks <= k])

    #         for k, v in results.items():
    #             results[k] = v / count

    #     return results
    def evaluate(self, ent_emb, eval_dataloader, num_cand='all'):
        results = ddict(float)
        count = 0

        eval_dataloader.dataset.num_cand = num_cand

        if num_cand == 'all':
            total_batches = len(eval_dataloader)
            for batch_index, batch in enumerate(eval_dataloader):
                pos_triple, tail_label, head_label = [b.to(self.args.gpu) for b in batch]
                head_idx, rel_idx, tail_idx = pos_triple[:, 0], pos_triple[:, 1], pos_triple[:, 2]

                # tail prediction
                pred = self.kge_model((pos_triple, None), ent_emb, mode='tail-batch')
                # self.logger.info(f'Batch {batch_index}/{total_batches}: Tail prediction done.')

                b_range = torch.arange(pred.size()[0], device=self.args.gpu)
                target_pred = pred[b_range, tail_idx]
                pred = torch.where(tail_label.bool(), -torch.ones_like(pred) * 10000000, pred)
                pred[b_range, tail_idx] = target_pred

                tail_ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[b_range, tail_idx]

                # head prediction
                pred = self.kge_model((pos_triple, None), ent_emb, mode='head-batch')
                # self.logger.info(f'Batch {batch_index}/{total_batches}: Head prediction done.')

                b_range = torch.arange(pred.size()[0], device=self.args.gpu)
                target_pred = pred[b_range, head_idx]
                pred = torch.where(head_label.bool(), -torch.ones_like(pred) * 10000000, pred)
                pred[b_range, head_idx] = target_pred

                head_ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[b_range, head_idx]

                ranks = torch.cat([tail_ranks, head_ranks])
                ranks = ranks.float()
                count += torch.numel(ranks)
                results['mr'] += torch.sum(ranks).item()
                results['mrr'] += torch.sum(1.0 / ranks).item()

                for k in [1, 3, 10]:
                    results['hits@{}'.format(k)] += torch.numel(ranks[ranks <= k])

            for k, v in results.items():
                results[k] = v / count

        return results
















