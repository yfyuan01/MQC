from torch.utils.data import DataLoader, Dataset, Sampler
from pathlib import Path
from collections import defaultdict
import json
import random
from multiprocessing import Pool
import h5py
import pickle
import math
from tqdm import tqdm
import torch
import numpy as np
from copy import deepcopy
from param import parse_args
from sklearn.metrics import ndcg_score
from torch.utils.data.distributed import DistributedSampler

from transformers import T5TokenizerFast, BartTokenizer
from tokenization import VLT5TokenizerFast

project_dir = Path(__file__).resolve().parent.parent  # VLT5
workspace_dir = project_dir.parent
dataset_dir = workspace_dir.joinpath('datasets/').resolve()
# coco_dir = dataset_dir.joinpath('COCO')
vg_dir = dataset_dir
img_dir = '/ivi/ilps/personal/yyuan/img_feature_all.pkl'
# coco_feature_dir = coco_dir.joinpath('features')


class GRFineTuneDataset(Dataset):
    def __init__(self, split='train', raw_dataset=None, rank=-1, topk=-1, verbose=True, args=None, mode='train'):
        super().__init__()

        self.raw_dataset = raw_dataset
        self.topk = topk
        self.verbose = verbose
        self.args = args

        self.mode = mode

        # Loading datasets to data
        self.source = split
        if self.verbose:
            print('Data source: ', self.source)


        if self.args.tokenizer is None:
            self.args.tokenizer = self.args.backbone

        if 't5' in self.args.tokenizer:
            if self.args.use_vision:
                self.tokenizer = VLT5TokenizerFast.from_pretrained(
                    args.backbone,
                    # max_length=self.args.max_text_length,
                    do_lower_case=self.args.do_lower_case)
            else:
                self.tokenizer = T5TokenizerFast.from_pretrained(
                    args.backbone,
                    # max_length=self.args.max_text_length,
                    do_lower_case=self.args.do_lower_case)
        elif 'bart' in self.args.tokenizer:
            self.tokenizer = BartTokenizer.from_pretrained(
                args.backbone,
                # max_length=self.args.max_text_length,
                do_lower_case=self.args.do_lower_case)

            additional_special_tokens = [f'<extra_id_{i}>' for i in range(100-1, -1, -1)] + \
                    [f'<vis_extra_id_{i}>' for i in range(100-1, -1, -1)]
            special_tokens_dict = {'additional_special_tokens': additional_special_tokens}
            num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)

        if self.args.oscar_tags:
            # Load VG Classes
            vg_classes = []
            with open(vg_dir.joinpath('objects_vocab.txt')) as f:
                for obj in f.readlines():
                    vg_classes.append(obj.split(',')[0].lower().strip())
            self.vg_classes = vg_classes

        # for source in self.sources:
        source = self.source
        data_info_path = dataset_dir.joinpath(f'{source}.json')
        n_images = 0
        with open(data_info_path) as f:
            karpathy_data = json.load(f)
            data = []
            if source == 'train':
                for k in karpathy_data:
                    new_datum = {
                        'img_ids': k['img_ids'],
                        'id': k['facet_id'],
                        'sent': k['topic']+' '+k['question']+' '+k['answer'],
                        'targets': k['related_dict'],
                        'is_train': True,
                    }
                    data.append(new_datum)
                    n_images += 1
            elif source=='val':
                for k in karpathy_data:
                    new_datum = {
                        'img_ids': k['img_ids'],
                        'id': k['facet_id'],
                        'sent': k['topic']+' '+k['question']+' '+k['answer'],
                        'targets': k['related_dict'],
                        'is_train': False,
                    }
                    data.append(new_datum)
                    n_images += 1
            else:
                for k in karpathy_data:
                    new_datum = {
                        'img_ids': k['img_ids'],
                        'id': k['facet_id'],
                        'sent': k['topic']+' '+k['question']+' '+k['answer'],
                        'targets': k['related_dict'],
                        'is_train': False,
                    }
                    data.append(new_datum)
                    n_images += 1


        if self.verbose:
            print(f"{self.source} has {n_images} images")
            print(f"Loaded {len(data)} data from", split)

        self.n_gpus = torch.cuda.device_count()

        self.rank = rank
        if self.topk > 0:
            data = data[:self.topk]
            if self.verbose:
                print(f"Use only {self.topk} data")

        self.data = data

        if self.verbose:
            print("# all sentences:", len(self.data))

        # self.source_to_h5 = {}
        self.dict_topic = pickle.load(open(f'{dataset_dir}/keyword_dict.pkl','rb'))


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        out_dict = {}
        out_dict['args'] = self.args

        datum = self.data[idx]
        if 'id' in datum:
            out_dict['id'] = datum['id']
        ###### Image ######
        if self.args.use_vision:
            img_ids = datum['img_ids']
            out_dict['img_ids'] = img_ids
            # f = self.source_to_h5[source]
            f = self.img_feature_dict
            # if isinstance(f, Path):
            #     # path = self.data_source_to_h5_path[source]
            #     f = h5py.File(f, 'r')
            #     # self.split_to_h5_features[split_i] = f
            #     self.source_to_h5[source] = f

            # Normalize the boxes (to 0 ~ 1)
            out_dict['boxes'] = []
            out_dict['vis_feats'] = []
            for img_id in img_ids:
                img_h = f[img_id]['img_h']
                img_w = f[img_id]['img_w']
                boxes = f[img_id]['boxes']  # (x1, y1, x2, y2)
                boxes[:, (0, 2)] /= img_w
                boxes[:, (1, 3)] /= img_h
                np.testing.assert_array_less(boxes, 1+1e-5)
                # np.testing.assert_array_less(boxes, 1+5e-2)
                np.testing.assert_array_less(-boxes, 0+1e-5)
                boxes = torch.from_numpy(boxes)

                boxes.clamp_(min=0.0, max=1.0)

                n_boxes = len(boxes)

                # feats = np.zeros(shape=(n_boxes, 2048), dtype=np.float32)
                feats = f[img_id]['features']
                # f[f'{img_id}/features'].read_direct(feats)
                feats = torch.from_numpy(feats)

                n_boxes = min(n_boxes, self.args.max_n_boxes)
                out_dict['n_boxes'] = n_boxes
                # if not self.args.BUTD100:
                boxes = boxes[:n_boxes]
                feats = feats[:n_boxes]
                out_dict['boxes'].append(boxes)
                out_dict['vis_feats'].append(feats)
            out_dict['boxes'] = torch.mean(torch.stack(out_dict['boxes']),dim=0)
            out_dict['vis_feats'] = torch.mean(torch.stack(out_dict['vis_feats']),dim=0)

        ###### Text #####
        if self.args.no_prefix:
            input_text = ''
            input_ids = []

        else:
            if self.args.prefix is None:
                prefix = 'generative retrieval:'
            elif self.args.prefix == 'mask':
                if 'bart' in self.args.tokenizer:
                    prefix = "<mask>"

            input_tokens = [] #prefix
            input_tokens.extend(datum['sent'].split(' '))

            if self.args.oscar_tags:
                prefix = 'describe image with tags:'
                input_tokens = [prefix]
                obj_ids = f[f'{img_id}/obj_id'][()]
                for obj_id in obj_ids:
                    obj = self.vg_classes[obj_id]
                    if obj not in input_tokens:
                        input_tokens.append(obj)
            input_text = ' '.join(input_tokens)

            if 't5' in self.args.tokenizer:
                input_ids = self.tokenizer.encode(
                    input_text,
                    max_length=self.args.max_text_length, truncation=True)
            elif 'bart' in self.args.tokenizer:
                input_ids = self.tokenizer.encode(
                    input_text,
                    max_length=self.args.max_text_length, truncation=True)
            else:
                input_ids = self.tokenizer.convert_tokens_to_ids(
                    self.tokenizer.tokenize(input_text)[:self.args.max_text_length - 1] + ['[SEP]'])

        out_dict['input_text'] = input_text
        out_dict['input_ids'] = torch.LongTensor(input_ids)
        out_dict['input_length'] = len(input_ids)
        if 'targets' in datum:
            out_dict['targets'] = ' [SEP] '.join([self.dict_topic[t] for t in datum['targets']])
            targets = out_dict['targets']
        if datum['is_train']:
            sent = datum['sent'].strip()
            targets = targets.strip()
            if 't5' in self.args.tokenizer:
                target_ids = self.tokenizer.encode(targets, max_length=self.args.gen_max_length, truncation=True)
            elif 'bart' in self.args.tokenizer:
                target_ids = self.tokenizer.encode(targets, max_length=self.args.gen_max_length, truncation=True)

            assert len(target_ids) <= self.args.gen_max_length, len(target_ids)
            # out_dict['sent'] = sent
            out_dict['target_ids'] = torch.LongTensor(target_ids)
            out_dict['target_length'] = len(target_ids)




        return out_dict

    def collate_fn(self, batch):
        batch_entry = {}

        B = len(batch)

        S_W_L = max(entry['input_length'] for entry in batch)
        input_ids = torch.ones(B, S_W_L, dtype=torch.long) * self.tokenizer.pad_token_id

        if self.args.no_prefix:
            assert input_ids.size() == (B, 0)

        if self.args.use_vision:
            V_L = max(entry['n_boxes'] for entry in batch)
            # V_L = len(batch[0]['boxes'])
            feat_dim = batch[0]['vis_feats'].shape[-1]

            boxes = torch.zeros(B, V_L, 4, dtype=torch.float)
            vis_feats = torch.zeros(B, V_L, feat_dim, dtype=torch.float)
            vis_attention_mask = torch.zeros(B, V_L, dtype=torch.float)

        if 'target_ids' in batch[0]:
            T_W_L = max(entry['target_length'] for entry in batch)
            target_ids = torch.ones(B, T_W_L, dtype=torch.long) * self.tokenizer.pad_token_id

        # sentences = []

        targets = []
        img_ids = []
        img_paths = []
        input_text = []
        ids = []
        # target = []
        # reference = []
        for i, entry in enumerate(batch):
            input_ids[i, :entry['input_length']] = entry['input_ids']

            if self.args.use_vision:
                n_boxes = entry['n_boxes']
                boxes[i, :n_boxes] = entry['boxes']
                vis_feats[i, :n_boxes] = entry['vis_feats']
                vis_attention_mask[i, :n_boxes] = 1
                img_ids.append(entry['img_ids'])
                # img_paths.append(entry['img_path'])

            if 'target_ids' in entry:
                target_ids[i, :entry['target_length']] = entry['target_ids']

            if 'input_text' in entry:
                input_text.append(entry['input_text'])

            if 'id' in entry:
                ids.append(entry['id'])
            # sentences.append(entry['sent'])

            if 'targets' in entry:
                targets.append(entry['targets'])
            # if 'target' in entry:
            #     target.append(entry['target'])
            # if 'reference' in entry:
            #     reference.append(entry['reference'])
        batch_entry['input_ids'] = input_ids
        batch_entry['id'] = ids
        if 'target_ids' in batch[0]:
            word_mask = target_ids != self.tokenizer.pad_token_id
            target_ids[~word_mask] = -100
            batch_entry['target_ids'] = target_ids

        if self.args.use_vision:
            batch_entry['boxes'] = boxes
            batch_entry['vis_feats'] = vis_feats
            batch_entry['vis_attention_mask'] = vis_attention_mask
            batch_entry['img_id'] = img_ids
            batch_entry['img_paths'] = img_paths

        # batch_entry['sent'] = sentences

        batch_entry['input_text'] = input_text

        batch_entry['targets'] = targets
        batch_entry['task'] = 'generationretrieval'
        # batch_entry['target'] = target
        # batch_entry['reference'] = reference
        # print(len(batch_entry['target']))
        #         print(target)
        return batch_entry


def get_loader(args, split='karpathy_train', mode='train',
               batch_size=32, workers=4, distributed=False, gpu=0,
               topk=-1):

    verbose = (gpu == 0)

    dataset = GRFineTuneDataset(
        split,
        rank=gpu,
        topk=topk,
        verbose=verbose,
        args=args,
        mode=mode)

    if distributed:
        sampler = DistributedSampler(dataset)
    else:
        sampler = None

    if mode == 'train':
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=(sampler is None),
            num_workers=workers, pin_memory=True, sampler=sampler,
            collate_fn=dataset.collate_fn)
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=workers, pin_memory=True,
            sampler=sampler,
            shuffle=None if (sampler is not None) else False,
            collate_fn=dataset.collate_fn,
            drop_last=False)
    if verbose:
        loader.evaluator = GREvaluator()

    loader.task = 'genervation retrieval'

    return loader



class GREvaluator:
    def __init__(self):
        # import language_evaluation
        self.evaluator = ndcg_score

    def evaluate(self, predicts, answers):
        # predicts: top 10 results answers:
        p1 = 0
        p3 = 0
        p5 = 0
        p10 = 0
        for i in range(len(predicts)):
            predict = predicts[i]
            answer = answers[i].split(' [SEP] ')
            p1 += len([i for i in predict[:1] if i in answer]) / 1.0
            p3 += len([i for i in predict[:3] if i in answer]) / 3.0
            p5 += len([i for i in predict[:5] if i in answer]) / 5.0
            p10 += len([i for i in predict[:10] if i in answer]) / 10.0
        p1 = p1/len(predicts)
        p3 = p3/len(predicts)
        p5 = p5/len(predicts)
        p10 = p10/len(predicts)
            # break
        # p10 = len([i for i in answers if i in predicts[:10]]) / 10.0
        # p15 = len([i for i in answers if i in predicts[:15]]) / 15.0
        results = {'P@1':p1, 'P@5':p5,'P@3':p3, 'P@10':p10}

        # results = self.evaluator.run_evaluation(predicts, answers)

        return results
if __name__ == "__main__":
#     cudnn.benchmark = True
    args = parse_args()
    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node
    if args.local_rank in [0, -1]:
        print(args)
        comments = []
        if args.load is not None:
            ckpt_str = "_".join(args.load.split('/')[-3:])
            comments.append(ckpt_str)
        if args.comment != '':
            comments.append(args.comment)
        comment = '_'.join(comments)

        from datetime import datetime
        current_time = datetime.now().strftime('%b%d_%H-%M')

        run_name = f'{current_time}_GPU{args.world_size}'
        if len(comments) > 0:
            run_name += f'_{comment}'

        args.run_name = run_name
        args.gpu = args.local_rank
    train_loader = get_loader(
            args,
            split=args.valid, mode='val', batch_size=args.batch_size,
            distributed=False, gpu=args.gpu,
            workers=args.num_workers,
            topk=args.train_topk,
        )
    targets = []
    for i, batch in enumerate(train_loader):
        targets.extend(batch['targets'])
        # print(batch['targets'])
        # print(batch['input_text'][0])
        # print(batch['targets'][0])
        # break
    print(len(targets))
    # result = train_loader.evaluator.evaluate(references,targets)
