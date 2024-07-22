# -*- coding: utf-8 -*-
# @time: 10/5/2023 12:16 AM
# @Author: 坤
# @file: clip_pretrain.py
from torch import nn
from transformers import CLIPVisionModel, CLIPTextModel, CLIPTokenizer, CLIPModel
from .blip import init_tokenizer
from torch.nn import functional as F
import torch
import pdb

huggingface_path = '/zhaobai46a02/huggingface/transformers/clip-vit-base-patch32'

class CLIP_Pretrain(nn.Module):
    def __init__(self, **kwargs):
        # using clip in transformers
        super().__init__()
        print("creating clip")
        self.visual_encoder = CLIPVisionModel.from_pretrained(huggingface_path)
        self.text_encoder = CLIPTextModel.from_pretrained(huggingface_path)
        self.tokenizer = CLIPTokenizer.from_pretrained(huggingface_path)
        self.encoder = CLIPModel.from_pretrained(huggingface_path)

    def forward(self, image, caption, logit_scale):
        text = self.tokenizer(caption, padding='max_length', truncation=True, max_length=30,
                              return_tensors="pt").to(image.device)

        image_features = self.encoder.get_text_features(**text)
        text_features = self.encoder.get_image_features(pixel_values=image)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        sim_i2t = image_features @ text_features.t()
        sim_t2i = text_features @ image_features.t()

        batch_size = image.shape[0]

        sim_targets = torch.zeros(sim_t2i.size()).to(image.device)
        sim_targets.fill_diagonal_(1)

        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_targets, dim=1)
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_targets, dim=1)

        loss_ita = (loss_i2t + loss_t2i) / 2
        return loss_ita

    def cal_sim_for_item(self, image, caption, logit_scale=100.0):
        text = self.tokenizer(caption, padding='max_length', truncation=True, max_length=30,
                              return_tensors="pt").to(image.device)

        text_embeds = self.encoder.get_text_features(**text)
        image_embeds = self.encoder.get_image_features(pixel_values=image)

        # normalized features
        image_embeds = image_embeds / image_embeds.norm(dim=1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=1, keepdim=True)

        # calculate cosine similarity
        sim_i2t = image_embeds @ text_embeds.t()
        sim_t2i = text_embeds @ image_embeds.t()

        batch_size = image.shape[0]

        sim_targets = torch.zeros(sim_t2i.size()).to(image.device)
        sim_targets.fill_diagonal_(1)

        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_targets, dim=1)
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_targets, dim=1)

        loss_ita = (loss_i2t + loss_t2i) / 2

        sim_i2t = torch.sum(sim_i2t * sim_targets, dim=1)
        sim_t2i = torch.sum(sim_t2i * sim_targets, dim=1)
        sim = (sim_i2t + sim_t2i) / 2

        # image similarity
        image_similarity = image_embeds @ image_embeds.T
        # pdb.set_trace()

        return sim ,loss_ita, image_similarity

    def similarity(self, image, caption):
        text_embeds = self.encoder.get_text_features(**caption)
        image_embeds = self.encoder.get_image_features(pixel_values=image)

        # normalized features
        image_embeds = image_embeds / image_embeds.norm(dim=1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=1, keepdim=True)

        # calculate cosine similarity
        sim_i2t = image_embeds @ text_embeds.t()
        sim_t2i = text_embeds @ image_embeds.t()

        sim_targets = torch.zeros(sim_t2i.size()).to(image.device)
        sim_targets.fill_diagonal_(1)

        sim_i2t = torch.sum(sim_i2t * sim_targets, dim=1)
        sim_t2i = torch.sum(sim_t2i * sim_targets, dim=1)
        sim = (sim_i2t + sim_t2i) / 2

        return sim


def clip(**kwargs):
    return CLIP_Pretrain(**kwargs)
