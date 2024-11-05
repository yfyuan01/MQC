
import torch
import torch.nn as nn
import numpy as np
from GENRE.genre.trie import MarisaTrie
import pickle


from modeling_t5 import VLT5
class VLT5GR(VLT5):
    def __init__(self, config):
        super().__init__(config)

    def train_step(self, batch):

        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)
        input_ids = batch['input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)
        # tags = batch['tag'].to(device)
        lm_labels = batch["target_ids"].to(device)

        output = self(
            input_ids=input_ids,
            vis_inputs=(vis_feats, vis_pos),
            labels=lm_labels,
            # tags = tags,
            return_dict=True
        )
        assert 'loss' in output

        lm_mask = lm_labels != -100
        B, L = lm_labels.size()

        loss = output['loss']

        loss = loss.view(B, L) * lm_mask

        loss = loss.sum(dim=1) / lm_mask.sum(dim=1).clamp(min=1)  # B

        loss = loss.mean()

        result = {
            'loss': loss
        }
        return result

    def test_step(self, batch, **kwargs):
        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)
        input_ids = batch['input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)
        # tags = batch['tag'].to(device)
        topics = pickle.load(open('../datasets/keyword_dict.pkl', 'rb'))
        topics = [[0] + self.tokenizer.encode(t) for t in topics.values()]
        trie = MarisaTrie(topics)

        output = self.generate(
            input_ids=input_ids,
            vis_inputs=(vis_feats, vis_pos),
            # tags=tags,
            num_return_sequences=10,
            prefix_allowed_tokens_fn=lambda batch_id, sent: trie.get(sent.tolist()),
            output_scores=True,
            return_dict_in_generate=True,
            **kwargs
        )
        generated_scores = output['sequences_scores']
        output = output['sequences']
        generated_sents = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        sents = []
        scores = []
        for i in range(0, len(generated_sents), 10):
            sents.append(generated_sents[i:i + 10])
            scores.append(generated_scores[i:i+10])
        result = {}
        result['pred'] = sents
        result['scores'] = scores
        return result


from modeling_bart import VLBart
class VLBartGR(VLBart):
    def __init__(self, config):
        super().__init__(config)

    def train_step(self, batch):

        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)
        input_ids = batch['input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)

        lm_labels = batch["target_ids"].to(device)

        output = self(
            input_ids=input_ids,
            vis_inputs=(vis_feats, vis_pos),
            labels=lm_labels,
            return_dict=True
        )
        assert 'loss' in output

        lm_mask = lm_labels != -100
        B, L = lm_labels.size()

        loss = output['loss']

        loss = loss.view(B, L) * lm_mask

        loss = loss.sum(dim=1) / lm_mask.sum(dim=1).clamp(min=1)  # B

        loss = loss.mean()

        result = {
            'loss': loss
        }
        return result

    def test_step(self, batch, **kwargs):
        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)
        input_ids = batch['input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)
        topics = pickle.load(open('/home/yyuan/MQC/VL-T5/datasets/keyword_dict.pkl','rb'))

        topics = [self.tokenizer.encode(t) for t in topics.values()]
        trie = MarisaTrie(topics[:2])
        print(type(self.tokenizer))
        output,scores = self.generate(
            input_ids=input_ids,
            vis_inputs=(vis_feats, vis_pos),
            num_return_sequences=1,
            prefix_allowed_tokens_fn=lambda batch_id, sent: trie.get(sent.tolist()),
            output_scores=True,
            **kwargs
        )
        print(output)
        generated_sents = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        print(generated_sents)
        result = {}
        result['pred'] = generated_sents

        return result
