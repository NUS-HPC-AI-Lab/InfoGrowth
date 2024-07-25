# -*- coding: utf-8 -*-
# @time: 10/6/2023 12:24 AM
# @Author: 坤
# @file: blipv2.py
import torch
import torch.nn as nn
from transformers import Blip2Model, Blip2ForConditionalGeneration, Blip2Processor, BertTokenizer, \
        Blip2Config

import pdb
class BLIP_v2(nn.Module):
    def __init__(self, captioner_path, max_words=30, num_beams=5):
        super(BLIP_v2, self).__init__()
        # self.captioner = Blip2ForConditionalGeneration.from_pretrained('/zhaobai46a02/huggingface/transformers/blip2-opt-2.7b')
        # self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.num_beams = num_beams
        self.max_words = max_words
        # no pretrained
        # self.init_tokenizer()
        # self.captioner_config = Blip2Config.from_pretrained(captioner_path)
        self.captioner = Blip2ForConditionalGeneration.from_pretrained(captioner_path)
        self.tokenizer = Blip2Processor.from_pretrained(captioner_path)

    def init_tokenizer(self):
        self.tokenizer = BertTokenizer.from_pretrained('/zhaobai46a02/huggingface/transformers/bert-base-uncased',
                                                       local_files_only=True)
        self.tokenizer.add_special_tokens({'bos_token': '[DEC]'})
        self.tokenizer.add_special_tokens({'additional_special_tokens': ['[ENC]']})
        self.tokenizer.enc_token_id = self.tokenizer.additional_special_tokens_ids[0]


    def generate(self, image):
        generated_ids = self.captioner.generate(image, max_length=self.max_words, num_beams=self.num_beams, do_sample=False)
        generated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        return generated_text

    def forward(self):
        pass


def blipv2(config):
    return BLIP_v2(config['caption_model'], config['max_words'], config['num_beams'])

if __name__ == "__main__":
    pass
