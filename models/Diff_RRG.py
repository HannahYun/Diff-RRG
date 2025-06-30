import os
import json
os.environ["PYTHONIOENCODING"] = "utf-8"

import torch
import torch.nn as nn
import lightning.pytorch as pl
from transformers import LlamaForCausalLM, LlamaTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score
from transformers import SwinModel, Blip2Model, ResNetModel
from lightning_tools.optim import config_optimizer
from peft import get_peft_model, LoraConfig, TaskType
import pdb
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoImageProcessor, AutoModel
from torch.distributions import Gumbel

import numpy as np
import pandas as pd
import re

import open_clip
from open_clip import create_model_from_pretrained, get_tokenizer
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings

warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Progression Classifier
class ProgressionClassifier(nn.Module):
    def __init__(self, input_dim=768):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 3)  
        )
        
    def forward(self, x):
        return self.classifier(x)
    

class Diff_RRG(pl.LightningModule):
    """
    Diff_RRG model.
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(args)

        # 14 key diseases
        self.diseases = [
            "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
            "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion",
            "Lung Opacity", "Pleural Effusion",
            "Pleural Other", "Pneumonia", "Pneumothorax", "Support Devices", "No Finding",
        ]

        # Progression classifier
        self.progression_classifier = ProgressionClassifier().to(device)
        self.progression_loss = nn.CrossEntropyLoss()

        # Gumble Softmax threshold
        self.threshold = args.gumble_threshold
        self.tau = args.gumble_tau
        self.position_embedding = torch.nn.Linear(1, 768)

        # Selected Patch
        self.val_all_selected_patch_indices_list = []
        self.test_all_selected_patch_indices_list = []

        # ====================================================================== #
        # Visual Encoder #
        print("Loading BiomedCLIP visual encoder...")
        
        self.visual_encoder = open_clip.create_model('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        self.visual_encoder = self.visual_encoder.to(device)
        self.visual_encoder.eval()

        self.tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        print("BiomedCLIP visual encoder loaded successfully")

        if args.vis_use_lora:
            peft_config_visual = LoraConfig(
                                    r=args.vis_r,
                                    lora_alpha=args.vis_alpha,
                                    target_modules=["query", "value"],
                                    lora_dropout=args.lora_dropout,
                                    bias="none",
                                    modules_to_save=["classifier"],
                                )
            self.visual_encoder = get_peft_model(self.visual_encoder, peft_config_visual)
            self.visual_encoder.print_trainable_parameters()
            print('Loading vision encoder with LoRA -- Done')
        elif args.freeze_vm:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            print(f'Loading Frozen vision encoder:{args.vision_model} -- Done')
        else:
            print(f'Loading Trainable vision encoder:{args.vision_model} -- Done')

        # ====================================================================== #
        # LLM #

        print('Loading BioMistral-7b')
        self.llama_tokenizer = AutoTokenizer.from_pretrained(args.llama_model, use_fast=False)
        self.llama_tokenizer.pad_token_id = 0
        if args.low_resource:
            self.llama_model = AutoModelForCausalLM.from_pretrained(
                args.llama_model,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map="auto",
            )
        else:
            self.llama_model = AutoModelForCausalLM.from_pretrained(
                args.llama_model,
                torch_dtype=torch.bfloat16,
            )

        # ====================================================================== #
             
        if args.llm_use_lora:
            self.embed_tokens = self.llama_model.get_input_embeddings()
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, inference_mode=False, r=args.llm_r, lora_alpha=args.llm_alpha, lora_dropout=args.lora_dropout
            ) # CAUSAL_LM
            self.llama_model = get_peft_model(self.llama_model, peft_config)
            self.llama_model.print_trainable_parameters()
            print('Loading LLM LoRA Done')         
        else:
            self.embed_tokens = self.llama_model.get_input_embeddings()
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = False
            print('Loading LLM Done')


        # ====================================================================== #
        print("=============================")
        self.llama_proj = nn.Linear(768, self.llama_model.config.hidden_size).to(device)
        

        self.layer_norm = nn.LayerNorm(self.llama_model.config.hidden_size)
        self.end_sym = args.end_sym
        
        self.val_step_outputs = []
        self.test_step_outputs = []
        self.val_score = 0.0

        if args.delta_file is not None:
            state_dict = torch.load(args.delta_file, map_location=torch.device(f'cuda:{torch.cuda.current_device()}'))['model']
            self.load_state_dict(state_dict=state_dict, strict=False)
            print(f'Load checkpoint from {args.delta_file}')


    def score(self, ref, hypo):
        """
        ref, dictionary of reference sentences (id, sentence)
        hypo, dictionary of hypothesis sentences (id, sentence)
        score, dictionary of scores
        """
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Rouge(), "ROUGE_L"),
            (Meteor(), "METEOR"),
            (Cider(), "CIDEr")
        ]
        final_scores = {}
        for scorer, method in scorers:
            try:
                score, scores = scorer.compute_score(ref, hypo, verbose=0)
            except TypeError:
                score, scores = scorer.compute_score(ref, hypo)
            if type(method) == list:
                for sc, m in zip(score, method):
                    final_scores[m] = sc
            else:
                final_scores[method] = score
        return final_scores


    def encode_img(self, images):
        for image in images:
            device = image.device

            with torch.no_grad():
                image_embed = self.visual_encoder.visual.trunk.forward_features(image)
                patch_embed = image_embed[:, 1:]
        
        inputs_img = self.llama_proj(image_embed.float())
        atts_img = torch.ones(image_embed.size()[:-1], dtype=torch.long).to(device)
        
        return inputs_img, atts_img, patch_embed
    

    def prior_encode_img(self, images):
        if isinstance(images, list):
            image = images[0]
        else:
            image = images

        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        elif len(image.shape) == 4:
            pass
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}")

        device = image.device
    
        with torch.no_grad():
            image_embed = self.visual_encoder.visual.trunk.forward_features(image)
            patch_embed = image_embed[:, 1:]

            image_embed = self.llama_proj(image_embed)
            
        atts_img = torch.ones(image_embed.size()[:-1], dtype=torch.long).to(device)
        
        return image_embed, atts_img, patch_embed

    
    def prompt_wrap_cls(self, img_embeds, atts_img, prior_input_text, progression_prompts_embeds):
        batch_size = img_embeds.shape[0]
        p_before = '### User: <Img>'
        p_mid = '</Img> Generate a comprehensive and detailed diagnosis report for this chest xray image.\nPlease refer to the historical diagnosis information.\nHere is the historical diagnosis report <Report>'
        p_mid1 = '</Report> and disease progression status by classification using difference map between current image and prior image: <Disease Progression>'
        p_after = '</Disease Progression> \n### Assistant:'

        p_before_tokens = self.llama_tokenizer(
            p_before, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
        
        p_mid_tokens = self.llama_tokenizer(
            p_mid, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
        
        p_mid1_tokens = self.llama_tokenizer(
            p_mid1, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
        
        p_after_tokens = self.llama_tokenizer(
            p_after, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
        
        prior_tokens = self.llama_tokenizer(
            prior_input_text, return_tensors="pt", add_special_tokens=False, padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length).to(img_embeds.device)
        
        p_before_embeds = self.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
        p_mid_embeds = self.embed_tokens(p_mid_tokens.input_ids).expand(batch_size, -1, -1)
        p_mid_embeds1 = self.embed_tokens(p_mid1_tokens.input_ids).expand(batch_size, -1, -1)
        p_after_embeds = self.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)
        prior_embeds = self.embed_tokens(prior_tokens.input_ids)

        wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds, p_mid_embeds, prior_embeds, p_mid_embeds1, progression_prompts_embeds, p_after_embeds], dim=1)
        wrapped_atts_img = atts_img[:, :1].expand(-1, wrapped_img_embeds.shape[1])
        return wrapped_img_embeds, wrapped_atts_img


    def compute_disease_patch_similarity(self, patch_emb):
        template = 'a photo of '
        texts = [template + disease for disease in self.diseases]

        device = patch_emb.device
        text_tokens = self.tokenizer(texts, context_length=256).to(device)
        
        with torch.no_grad():
            patch_embedding = patch_emb[0, :, :]
            _, text_features, logit_scale = self.visual_encoder(None, text_tokens)

            text_emb2 = self.visual_encoder.text.transformer(text_tokens, return_dict=True)
            hidden_states = text_emb2.last_hidden_state

            text_feat = hidden_states[:, 0, :]

            text_feat = F.normalize(text_feat, dim=-1)
            patch_embedding = F.normalize(patch_embedding, dim=-1)
            
            similarity = (logit_scale * text_feat @ patch_embedding.t()).softmax(dim=-1)
        
        return similarity


    def get_disease_representation(self, image_features):
        similarity_matrix = self.compute_disease_patch_similarity(image_features)

        # Gumbel-Softmax
        gumbel_dist = Gumbel(0, 1)
        gumbel_noise = gumbel_dist.sample(similarity_matrix.shape).to(image_features.device)
        gumbel_similarity = F.softmax((similarity_matrix + gumbel_noise) / self.tau, dim=-1)
        selected_mask = gumbel_similarity > self.threshold
        selected_indices = [torch.nonzero(mask, as_tuple=True)[0] for mask in selected_mask]

        patch_features = image_features[0, :, :]
        selected_patch_embeddings = torch.zeros(14, 768, device=image_features.device)

        selected_patch_nums = []
        selected_patch_indices = []
        
        for i, indices in enumerate(selected_indices):
            if indices.numel() > 0:
                selected_patch_embeddings[i] = patch_features[indices].mean(dim=0)
            selected_patch_nums.append(len(indices))
            selected_patch_indices.append(indices.cpu().numpy().tolist())

        position_embedding = torch.arange(14, device=image_features.device).unsqueeze(1).float()
        position_embedding = self.position_embedding(position_embedding)
        selected_patch_embeddings += position_embedding

        disease_features = self.llama_proj(selected_patch_embeddings)
        disease_features = self.layer_norm(disease_features)

        return disease_features, selected_patch_embeddings, selected_patch_nums, selected_patch_indices
    

    def _get_progression_label(self, logits, selected_current_nums, selected_prior_nums):
        probs = F.softmax(logits, dim=-1)
        pred_labels = torch.argmax(probs, dim=-1) - 1
        
        for i in range(logits.shape[0]):
            for j in range(14):
                if selected_current_nums[i][j] == 0 or selected_prior_nums[i][j] == 0:
                    pred_labels[i, j] = -2
        
        return pred_labels

    def _create_progression_prompt(self, progression_labels):
        label_to_text = {
            -2: "N/A",
            -1: "worsening",
             0: "stable",
             1: "improving"
        }
        
        batch_size = progression_labels.shape[0]
        progression_prompts_list = []
        
        for b in range(batch_size):
            prompt_parts = []
            for disease_idx, disease_name in enumerate(self.diseases):
                label = progression_labels[b, disease_idx].item()
                status = label_to_text[label]
                prompt_parts.append(f"{disease_name}: {status}")
            
            progression_prompt = ", ".join(prompt_parts)
            progression_prompts_list.append(progression_prompt)
        
        progression_prompts_tokens = self.llama_tokenizer(
            progression_prompts_list,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=300
        ).to(progression_labels.device)
        
        progression_prompts_embeds = self.embed_tokens(progression_prompts_tokens.input_ids)
        return progression_prompts_embeds
    

    def forward(self, samples):
        image = samples["image"]
        img_embeds, atts_img, patch_emb = self.encode_img(image)
        img_embeds = self.layer_norm(img_embeds)

        prior_input_text = samples["prior_input_text"]

        prior_image = samples["prior_image"]
        batch_size = len(samples["input_text"])

        selected_patch_indices_list = []
        progression_logits_list = []

        for i in range(batch_size):
            prior_img_embeds, prior_atts_img, prior_patch_emb = self.prior_encode_img([prior_image[0][i]])
            prior_img_embeds = self.layer_norm(prior_img_embeds)

            current_disease_features, selected_current_patch_emb, selected_current_nums, selected_current_indices = self.get_disease_representation(patch_emb[i:i+1])
            prior_disease_features, selected_prior_patch_emb, selected_prior_nums, selected_prior_indices = self.get_disease_representation(prior_patch_emb)

            disease_difference = selected_current_patch_emb - selected_prior_patch_emb

            selected_patch_indices_list.append({
                "id": samples["id"][i],
                "current_indices": selected_current_indices,
                "prior_indices": selected_prior_indices,
                "current_nums": selected_current_nums,
                "prior_nums": selected_prior_nums
            })

            progression_logits = self.progression_classifier(disease_difference)
            progression_logits_list.append(progression_logits)

        batch_progression_logits = torch.stack(progression_logits_list, dim=0)

        selected_current_nums = [item["current_nums"] for item in selected_patch_indices_list]
        selected_prior_nums = [item["prior_nums"] for item in selected_patch_indices_list]
        
        progression_labels = self._get_progression_label(batch_progression_logits, selected_current_nums, selected_prior_nums)
        progression_prompts_embeds = self._create_progression_prompt(progression_labels)

        valid_mask = (progression_labels != -2).view(-1)
        valid_logits = batch_progression_logits.view(-1, 3)[valid_mask]
        valid_targets = torch.tensor(samples["disease_progression"]).to(batch_progression_logits.device).view(-1)[valid_mask] + 1

        if valid_logits.size(0) > 0:
            progression_loss = self.progression_loss(valid_logits, valid_targets)
        else:
            progression_loss = torch.tensor(0.0).to(batch_progression_logits.device)

        img_embeds, atts_img = self.prompt_wrap_cls(img_embeds, atts_img, prior_input_text, progression_prompts_embeds)

        self.llama_tokenizer.padding_side = "right"
        text = [t + self.end_sym for t in samples["input_text"]]

        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length,
            add_special_tokens=False
        ).to(image[0].device)

        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == 0, -100
        )

        empty_targets = (
            torch.ones([atts_img.shape[0], atts_img.shape[1]+1],
                       dtype=torch.long).to(image[0].device).fill_(-100)  # plus one for bos
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=to_regress_tokens.input_ids.dtype,
                         device=to_regress_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        atts_bos = atts_img[:, :1]

        to_regress_embeds = self.embed_tokens(to_regress_tokens.input_ids)
        inputs_embeds = torch.cat([bos_embeds, img_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_img, to_regress_tokens.attention_mask], dim=1)

        outputs = self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )

        loss = outputs.loss + 0.1 * progression_loss
        return {"loss": loss, "progression_loss": progression_loss}


    def training_step(self, batch, batch_idx):
        result = self(batch)
        self.log_dict(result, prog_bar=True)
        return result


    def save_checkpoint(self, eval_res):
        current_epoch, global_step = self.trainer.current_epoch, self.trainer.global_step
        param_grad_dic = {
            k: v.requires_grad for (k, v) in self.named_parameters() if v.requires_grad
        }
        state_dict = self.state_dict()
        for k in list(state_dict.keys()):
            if k not in param_grad_dic.keys():
                del state_dict[k]
        save_obj = {
            "model": state_dict,
            "config": self.hparams,
            "epoch": current_epoch,
            "step":global_step
        }
        os.makedirs(os.path.join(self.hparams.savedmodel_path, 'checkpoints'), exist_ok=True)
        save_to = os.path.join(
            self.hparams.savedmodel_path, 'checkpoints',
            "checkpoint_epoch{}_step{}_bleu{:3f}_cider{:3f}.pth".format(current_epoch, global_step, eval_res['Bleu_4'], eval_res['CIDEr']),
        )
        self.print("Saving checkpoint at step {} to {}.".format(global_step, save_to))
        torch.save(save_obj, save_to)
    

    def validation_step(self, samples, batch_idx):
        self.llama_tokenizer.padding_side = "right"
        to_regress_tokens = self.llama_tokenizer(
            samples['input_text'],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length,
            add_special_tokens=False
        )

        image = samples["image"]
        img_embeds, atts_img, patch_emb = self.encode_img(image)
        img_embeds = self.layer_norm(img_embeds)

        prior_input_text = samples["prior_input_text"]

        prior_image = samples["prior_image"]
        batch_size = len(samples["input_text"])


        selected_patch_indices_list = []
        progression_logits_list = []

        for i in range(batch_size):
            prior_img_embeds, prior_atts_img, prior_patch_emb = self.prior_encode_img([prior_image[0][i]])
            prior_img_embeds = self.layer_norm(prior_img_embeds)

            current_disease_features, selected_current_patch_emb, selected_current_nums, selected_current_indices = self.get_disease_representation(patch_emb[i:i+1])
            prior_disease_features, selected_prior_patch_emb, selected_prior_nums, selected_prior_indices = self.get_disease_representation(prior_patch_emb)

            disease_difference = selected_current_patch_emb - selected_prior_patch_emb

            selected_patch_indices_list.append({
                "id": samples["id"][i],
                "current_indices": selected_current_indices,
                "prior_indices": selected_prior_indices,
                "current_nums": selected_current_nums,
                "prior_nums": selected_prior_nums
            })

            progression_logits = self.progression_classifier(disease_difference)
            progression_logits_list.append(progression_logits)

        batch_progression_logits = torch.stack(progression_logits_list, dim=0)

        selected_current_nums = [item["current_nums"] for item in selected_patch_indices_list]
        selected_prior_nums = [item["prior_nums"] for item in selected_patch_indices_list]
        
        progression_labels = self._get_progression_label(batch_progression_logits, selected_current_nums, selected_prior_nums)
        progression_prompts_embeds = self._create_progression_prompt(progression_labels)

        img_embeds, atts_img = self.prompt_wrap_cls(img_embeds, atts_img, prior_input_text, progression_prompts_embeds)
        
        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=atts_img.dtype,
                         device=atts_img.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        atts_bos = atts_img[:, :1]

        inputs_embeds = torch.cat([bos_embeds, img_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_img], dim=1)

        outputs = self.llama_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            num_beams=self.hparams.beam_size,
            do_sample=self.hparams.do_sample,
            min_new_tokens=self.hparams.min_new_tokens,
            max_new_tokens=self.hparams.max_new_tokens,
            repetition_penalty=self.hparams.repetition_penalty,
            length_penalty=self.hparams.length_penalty,
            temperature=self.hparams.temperature,
        )
        hypo = [self.decode(i) for i in outputs]
        ref = [self.decode(i) for i in to_regress_tokens['input_ids']]
        ref = [re.sub(r'!+', '', r) for r in ref]
        self.val_step_outputs.append({"hypo": hypo, "ref": ref, "id": samples["id"]})

        self.val_all_selected_patch_indices_list.append(selected_patch_indices_list)
        current_epoch, global_step = self.trainer.current_epoch, self.trainer.global_step
        with open(os.path.join(self.hparams.savedmodel_path, f"val_selected_patch_indices_{current_epoch}_{global_step}.json"), "w") as f:
            json.dump(self.val_all_selected_patch_indices_list, f)
        return hypo, ref
    

    def decode(self, output_token):
        if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
            output_token = output_token[1:]
        if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
            output_token = output_token[1:]
        output_text = self.llama_tokenizer.decode(output_token, add_special_tokens=False)
        output_text = output_text.split('</s>')[0].strip()
        output_text = output_text.replace('<unk>', '')
        return output_text


    def on_validation_epoch_end(self):
        ref, hypo, ids = [], [], []
        for i in self.val_step_outputs:
            ref.extend(i['ref'])
            hypo.extend(i['hypo'])
            ids.extend(i['id'])
        ref = {k:[v] for k, v in zip(ids, ref)}
        hypo = {k:[v] for k, v in zip(ids, hypo)}
        eval_res = self.score(ref=ref,hypo=hypo)
        
        self.log_dict(eval_res, sync_dist=True, logger=True)
        
        result_folder = os.path.join(self.hparams.savedmodel_path, 'result')
        os.makedirs(result_folder, exist_ok=True)
        current_epoch, global_step = self.trainer.current_epoch, self.trainer.global_step
        json.dump(hypo, open(os.path.join(result_folder, f"result_{current_epoch}_{global_step}" + '.json'), 'w'))
        json.dump(ref, open(os.path.join(result_folder, 'refs.json'), 'w'))
        self.print(eval_res)

        val_score = 0
        for score_type, weight in zip(self.hparams.scorer_types, self.hparams.weights):
            val_score += eval_res[score_type] * weight

        if self.trainer.local_rank == 0:
            if val_score > self.val_score:
                self.save_checkpoint(eval_res)
                self.val_score = val_score
        self.val_step_outputs.clear()


    def test_step(self, samples, batch_idx):
        self.llama_tokenizer.padding_side = "right"
        to_regress_tokens = self.llama_tokenizer(
            samples['input_text'],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length,
            add_special_tokens=False
        )

        image = samples["image"]
        img_embeds, atts_img, patch_emb = self.encode_img(image)
        img_embeds = self.layer_norm(img_embeds)

        prior_input_text = samples["prior_input_text"]

        prior_image = samples["prior_image"]
        batch_size = len(samples["input_text"])


        selected_patch_indices_list = []
        progression_logits_list = []

        for i in range(batch_size):
            prior_img_embeds, prior_atts_img, prior_patch_emb = self.prior_encode_img([prior_image[0][i]])
            prior_img_embeds = self.layer_norm(prior_img_embeds)

            current_disease_features, selected_current_patch_emb, selected_current_nums, selected_current_indices = self.get_disease_representation(patch_emb[i:i+1])
            prior_disease_features, selected_prior_patch_emb, selected_prior_nums, selected_prior_indices = self.get_disease_representation(prior_patch_emb)

            disease_difference = selected_current_patch_emb - selected_prior_patch_emb

            selected_patch_indices_list.append({
                "id": samples["id"][i],
                "current_indices": selected_current_indices,
                "prior_indices": selected_prior_indices,
                "current_nums": selected_current_nums,
                "prior_nums": selected_prior_nums
            })

            progression_logits = self.progression_classifier(disease_difference)
            progression_logits_list.append(progression_logits)

        batch_progression_logits = torch.stack(progression_logits_list, dim=0)

        selected_current_nums = [item["current_nums"] for item in selected_patch_indices_list]
        selected_prior_nums = [item["prior_nums"] for item in selected_patch_indices_list]
        
        progression_labels = self._get_progression_label(batch_progression_logits, selected_current_nums, selected_prior_nums)
        progression_prompts_embeds = self._create_progression_prompt(progression_labels)

        img_embeds, atts_img = self.prompt_wrap_cls(img_embeds, atts_img, prior_input_text, progression_prompts_embeds)
        
        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=atts_img.dtype,
                         device=atts_img.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        atts_bos = atts_img[:, :1]

        inputs_embeds = torch.cat([bos_embeds, img_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_img], dim=1)

        outputs = self.llama_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            num_beams=self.hparams.beam_size,
            do_sample=self.hparams.do_sample,
            min_new_tokens=self.hparams.min_new_tokens,
            max_new_tokens=self.hparams.max_new_tokens,
            repetition_penalty=self.hparams.repetition_penalty,
            length_penalty=self.hparams.length_penalty,
            temperature=self.hparams.temperature,
        )
        hypo = [self.decode(i) for i in outputs]
        ref = [self.decode(i) for i in to_regress_tokens['input_ids']]
        ref = [re.sub(r'!+', '', r) for r in ref]
        self.test_step_outputs.append({"hypo": hypo, "ref": ref, "id": samples["id"]})

        self.test_all_selected_patch_indices_list.append(selected_patch_indices_list)
        with open(os.path.join(self.hparams.savedmodel_path, "test_selected_patch_indices.json"), "w") as f:
            json.dump(self.test_all_selected_patch_indices_list, f)

        return hypo, ref


    def on_test_epoch_end(self):
        """
        This function is called at the end of the test epoch.
        It is recommended to test on single device to ensure each sample/batch gets evaluated exactly once.
        This is helpful to make sure benchmarking for research papers is done the right way.
        Otherwise, in a multi-device setting, samples could occur duplicated when DistributedSampler is used, for eg. with strategy="ddp". 
        It replicates some samples on some devices to make sure all devices have same batch size in case of uneven inputs.
        """
        ref, hypo, ids = [], [], []
        for i in self.test_step_outputs:
            ref.extend(i['ref'])
            hypo.extend(i['hypo'])
            ids.extend(i['id'])

        ref = {k:[v] for k, v in zip(ids, ref)}
        hypo = {k:[v] for k, v in zip(ids, hypo)}
        eval_res = self.score(ref=ref,hypo=hypo)

        self.log_dict(eval_res, sync_dist=True, logger=True)

        result_folder = os.path.join(self.hparams.savedmodel_path, 'result')
        os.makedirs(result_folder, exist_ok=True)
        json.dump(hypo, open(os.path.join(result_folder, f"test_result.json"), 'w'))
        json.dump(ref, open(os.path.join(result_folder, 'test_refs.json'), 'w'))
        self.print(f"Test result of {self.hparams.delta_file}: {eval_res}")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.hparams.max_epochs, eta_min=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad()
