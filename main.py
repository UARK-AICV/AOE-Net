import sys
import os
import argparse

from tqdm import tqdm
import pandas as pd
import torch
import torch.nn.parallel
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from models.model import EventDetection
from dataset import VideoDataSet, Collator
from loss_function import bmn_loss_func, get_mask
from post_processing import PostProcessor, getDatasetDict
from utils import ProposalGenerator

from eval_anet import evaluate_proposals as anet_evaluate_prop
from eval_thumos import evaluate_proposals as thumos_evaluate_prop

from eval_det_anet import evaluate_detections as anet_evaluate_det
from eval_det_thumos import evaluate_detections as thumos_evaluate_det

from config.defaults import get_cfg

sys.dont_write_bytecode = True


class Solver:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = EventDetection(cfg).cuda()
        self.model = torch.nn.DataParallel(self.model, device_ids=cfg.GPU_IDS)
        if cfg.MODE not in ['train', 'training']:  # TODO: add condition for resume feature.
            checkpoint = torch.load(cfg.TEST.CHECKPOINT_PATH)
            print('Loaded model at epoch %d.' % checkpoint['epoch'])
            self.model.load_state_dict(checkpoint['state_dict'])

        if cfg.MODE in ['train', 'training']:
            self.optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
            #self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
            self.train_collator = Collator(cfg, 'train')
        self.test_collator = Collator(cfg, 'test')

        self.temporal_dim = cfg.DATA.TEMPORAL_DIM
        self.max_duration = cfg.DATA.MAX_DURATION

        self.evaluate_func = None
        if cfg.DATASET == 'anet':
            if cfg.EVAL_TYPE == 'proposal':
                self.evaluate_func = anet_evaluate_prop
            elif cfg.EVAL_TYPE == 'detection':
                self.evaluate_func = anet_evaluate_det
        elif cfg.DATASET == 'thumos':
            if cfg.EVAL_TYPE == 'proposal':
                self.evaluate_func = thumos_evaluate_prop
            elif cfg.EVAL_TYPE == 'detection':
                self.evaluate_func = thumos_evaluate_det
        if self.evaluate_func is None:
            print('Evaluation function [{}] of dataset [{}] is not implemented'.format(cfg.EVAL_TYPE, cfg.DATASET))

    def train_epoch(self, data_loader, bm_mask, epoch, writer):
        cfg = self.cfg
        self.model.train()
        self.optimizer.zero_grad()
        loss_names = ['Loss', 'TemLoss', 'PemLoss Regression', 'PemLoss Classification']
        epoch_losses = [0] * 4
        period_losses = [0] * 4
        last_period_size = len(data_loader) % cfg.TRAIN.STEP_PERIOD
        last_period_start = cfg.TRAIN.STEP_PERIOD * (len(data_loader) // cfg.TRAIN.STEP_PERIOD)

        for n_iter, (env_features, agent_features, agent_masks, obj_features, obj_masks, label_confidence, label_start, label_end) in enumerate(tqdm(data_loader)):
            env_features = env_features.cuda() if cfg.USE_ENV else None
            agent_features = agent_features.cuda() if cfg.USE_AGENT else None
            agent_masks = agent_masks.cuda() if cfg.USE_AGENT else None
            obj_features = obj_features.cuda() if cfg.USE_OBJ else None
            obj_masks = obj_masks.cuda() if cfg.USE_OBJ else None

            label_start = label_start.cuda()
            label_end = label_end.cuda()
            label_confidence = label_confidence.cuda()

            confidence_map, start, end = self.model(env_features, agent_features, agent_masks, obj_features, obj_masks)

            losses = bmn_loss_func(confidence_map, start, end, label_confidence, label_start, label_end, bm_mask)
            period_size = cfg.TRAIN.STEP_PERIOD if n_iter < last_period_start else last_period_size
            total_loss = losses[0] / period_size
            total_loss.backward()

            losses = [l.cpu().detach().numpy() / cfg.TRAIN.STEP_PERIOD for l in losses]
            period_losses = [l + pl for l, pl in zip(losses, period_losses)]

            if (n_iter + 1) % cfg.TRAIN.STEP_PERIOD != 0 and n_iter != (len(data_loader) - 1):
                continue

            self.optimizer.step()
            self.optimizer.zero_grad()

            epoch_losses = [el + pl for el, pl in zip(epoch_losses, period_losses)]

            write_step = epoch * len(data_loader) + n_iter
            for i, loss_name in enumerate(loss_names):
                writer.add_scalar(loss_name, period_losses[i], write_step)
            period_losses = [0] * 4

        print(
            "BMN training loss(epoch %d): tem_loss: %.03f, pem reg_loss: %.03f, pem cls_loss: %.03f, total_loss: %.03f" % (
                epoch, epoch_losses[1] / (n_iter + 1),
                epoch_losses[2] / (n_iter + 1),
                epoch_losses[3] / (n_iter + 1),
                epoch_losses[0] / (n_iter + 1)))

    def train(self, n_epochs):
        exp_id = max([0] + [int(run.split('_')[-1]) for run in os.listdir(self.cfg.TRAIN.LOG_DIR)]) + 1
        log_dir = os.path.join(self.cfg.TRAIN.LOG_DIR, 'run_' + str(exp_id))
        if not os.path.isdir(os.path.dirname(log_dir)):
            os.makedirs(os.path.dirname(log_dir))

        writer = SummaryWriter(log_dir)
        checkpoint_dir = os.path.join(self.cfg.MODEL.CHECKPOINT_DIR, 'checkpoint_' + str(exp_id))
        assert not os.path.isdir(checkpoint_dir), 'Checkpoint directory %s has already been created.' % checkpoint_dir
        os.makedirs(checkpoint_dir)

        train_loader = torch.utils.data.DataLoader(
            VideoDataSet(self.cfg, split=self.cfg.TRAIN.SPLIT),
            batch_size=self.cfg.TRAIN.BATCH_SIZE, shuffle=True,
            num_workers=12, pin_memory=True, collate_fn=self.train_collator)

        eval_loader = torch.utils.data.DataLoader(
            VideoDataSet(self.cfg, split=self.cfg.VAL.SPLIT),
            batch_size=self.cfg.VAL.BATCH_SIZE, shuffle=False,
            num_workers=12, pin_memory=True, drop_last=False, collate_fn=self.test_collator)

        bm_mask = get_mask(self.temporal_dim, self.max_duration).cuda()
        scores = []
        for epoch in range(n_epochs):
            #print('Current LR: {}'.format(self.scheduler.get_last_lr()[0]))
            self.train_epoch(train_loader, bm_mask, epoch, writer)
            #self.scheduler.step()
            score = self.evaluate(eval_loader, self.cfg.VAL.SPLIT)

            state = {
                'epoch': epoch + 1,
                'score': score,
                'state_dict': self.model.state_dict()
            }
            if len(scores) == 0 or score > max(scores):
                torch.save(state, os.path.join(checkpoint_dir, "best_{}.pth".format(self.cfg.EVAL_SCORE)))
            torch.save(state, os.path.join(checkpoint_dir, "model_{}.pth".format(epoch + 1)))

            writer.add_scalar(self.cfg.EVAL_SCORE, score, epoch)
            scores.append(score)

    def evaluate(self, data_loader=None, split=None):
        self.inference(data_loader, split, self.cfg.VAL.BATCH_SIZE)
        score = self.evaluate_func(self.cfg)  # AUC if dataset=anet, AR@100 if dataset=thumos
        return score

    def inference(self, data_loader=None, split=None, batch_size=None):
        if not os.path.isdir('results/outputs/'):
            os.makedirs('results/outputs/')

        annotations = getDatasetDict(self.cfg.DATA.ANNOTATION_FILE, split) if self.cfg.DATASET == 'thumos' else None
        self.prop_gen = ProposalGenerator(self.temporal_dim, self.max_duration, annotations)
        self.post_processing = PostProcessor(self.cfg, split)
        if data_loader is None:
            data_loader = torch.utils.data.DataLoader(
                VideoDataSet(self.cfg, split=split),
                batch_size=batch_size, shuffle=False,
                num_workers=12, pin_memory=True, drop_last=False, collate_fn=self.test_collator)

        col_name = ["xmin", "xmax", "xmin_score", "xmax_score", "clr_score", "reg_score", "score"]
        self.model.eval()
        with torch.no_grad():
            for video_names, env_features, agent_features, agent_masks, obj_features, obj_masks in tqdm(data_loader):
                env_features = env_features.cuda() if self.cfg.USE_ENV else None
                agent_features = agent_features.cuda() if self.cfg.USE_AGENT else None
                agent_masks = agent_masks.cuda() if self.cfg.USE_AGENT else None
                obj_features = obj_features.cuda() if self.cfg.USE_OBJ else None
                obj_masks = obj_masks.cuda() if self.cfg.USE_OBJ else None

                confidence_map, start_map, end_map = self.model(env_features, agent_features, agent_masks, obj_features, obj_masks)
                confidence_map = confidence_map.cpu().numpy()
                start_map = start_map.cpu().numpy()
                end_map = end_map.cpu().numpy()

                batch_props = self.prop_gen(start_map, end_map, confidence_map, video_names)
                for video_name, new_props in zip(video_names, batch_props):
                    new_df = pd.DataFrame(new_props, columns=col_name)
                    new_df.to_feather("./results/outputs/" + video_name + ".feather")
        self.post_processing()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--cfg-file',
        default=None,
        type=str,
        help='Path to YAML config file.'
    )
    parser.add_argument(
        "opts",
        help="See slowfast/config/defaults.py for all options",
        default=None,
        nargs=argparse.REMAINDER
    )
    return parser.parse_args()


def main(args):
    cfg = get_cfg()
    if args.cfg_file:
        cfg.merge_from_file(args.cfg_file)
    if args.opts is not None:
        cfg.merge_from_list(args.opts)
    cfg.freeze()

    solver = Solver(cfg)

    if cfg.MODE in ["train", "training"]:
        solver.train(cfg.TRAIN.NUM_EPOCHS)
    elif cfg.MODE in ['validate', 'validation']:
        solver.evaluate(split=cfg.VAL.SPLIT)
    elif cfg.MODE in ['test', 'testing']:
        solver.inference(split=cfg.TEST.SPLIT, batch_size=cfg.TEST.BATCH_SIZE)


if __name__ == '__main__':
    args = get_args()
    main(args)
