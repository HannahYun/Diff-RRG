
import os
import json
import re
import numpy as np
from PIL import Image
import torch.utils.data as data
import torch
from open_clip import create_model_from_pretrained, get_tokenizer

class FieldParser:
    def __init__(
            self,
            args
    ):
        super().__init__()
        self.args = args
        self.dataset = args.dataset
        
        self.visual_encoder, self.preprocess = create_model_from_pretrained(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.visual_encoder = self.visual_encoder.to(self.device)
        self.visual_encoder.eval()
        print("BiomedCLIP model loaded successfully")

        
    def _parse_image(self, img):
        img = Image.fromarray(img)
        pixel_values = self.preprocess(img)
        return pixel_values

    # from https://github.com/cuhksz-nlp/R2Gen/blob/main/modules/tokenizers.py
    def clean_report(self, report):
        # clean Iu-xray reports
        if self.dataset == "iu_xray":
            report_cleaner = lambda t: t.replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '') \
            .replace('. 2. ', '. ').replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ') \
            .replace(' 2. ', '. ').replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
            sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').
                                            replace('\\', '').replace("'", '').strip().lower())
            tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
            report = ' . '.join(tokens) + ' .'
        # clean MIMIC-CXR reports
        else:
            report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
                .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
                .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
                .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
                .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
                .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
                .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ').replace(':', ' :') \
                .strip().lower().split('. ')
            sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+()\[\]{}]', '', t.replace('"', '').replace('/', '')
                                .replace('\\', '').replace("'", '').strip().lower())
            tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
            report = ' . '.join(tokens) + ' .' 
        # report = ' '.join(report.split()[:self.args.max_txt_len])
        return report


    def parse(self, features):
        to_return = {'id': features['id']}
        report = features.get("report", "")
        report = self.clean_report(report)
        to_return['input_text'] = report

        prior_report = features.get("context_report", "")
        prior_report = self.clean_report(prior_report)
        to_return['prior_input_text'] = prior_report

        disease_progression = features.get("disease_progression", "")
        if isinstance(disease_progression, list):
            disease_progression = torch.tensor(disease_progression)
        to_return['disease_progression'] = disease_progression

        # chest x-ray images
        images = []
        for image_path in features['image_path']:
            with Image.open(os.path.join(self.args.base_dir, image_path)) as pil:
                array = np.array(pil, dtype=np.uint8)
                if array.shape[-1] != 3 or len(array.shape) != 3:
                    array = np.array(pil.convert("RGB"), dtype=np.uint8)
                image = self._parse_image(array)
                images.append(image)

        prior_images = []
        if features['context_image']:
            with Image.open(os.path.join(self.args.base_dir, features['context_image'][0])) as pil:
                array = np.array(pil, dtype=np.uint8)
                if array.shape[-1] != 3 or len(array.shape) != 3:
                    array = np.array(pil.convert("RGB"), dtype=np.uint8)
                image = self._parse_image(array)
                prior_images.append(image)

        to_return["image"] = images
        to_return["prior_image"] = prior_images

        return to_return


    def transform_with_parse(self, inputs):
        return self.parse(inputs)


class ParseDataset(data.Dataset):
    def __init__(self, args, split='train'):
        self.args = args
        self.meta = json.load(open(args.annotation, 'r'))
        self.meta = self.meta[split]
        self.parser = FieldParser(args)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, index):
        return self.parser.transform_with_parse(self.meta[index])


def create_datasets(args):
    train_dataset = ParseDataset(args, 'train')
    dev_dataset = ParseDataset(args, 'val')
    test_dataset = ParseDataset(args, 'test')
    return train_dataset, dev_dataset, test_dataset


