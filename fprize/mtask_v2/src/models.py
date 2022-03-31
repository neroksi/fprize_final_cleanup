from turtle import forward
import torch
from torch import nn, optim
import numpy as np
import re
from collections import namedtuple
from warnings import warn

from transformers import AutoTokenizer, AutoConfig, AutoModelForTokenClassification, AutoModelForSequenceClassification

from . import configs as cfg

mTaskOutputType = namedtuple('mTaskOutputType', "out_ner out_seg")

def check_if_model_has_position_embeddings(model, use_position_embeddings=None):
    has_postion_embeddings = False
    for name, param in model.named_parameters():
        if "position_embeddings" in name.lower():
            has_postion_embeddings = True
            break
    
    if use_position_embeddings is not None:
        if use_position_embeddings and not has_postion_embeddings:
            raise ValueError("Position embeddings is required but the current model has NOT.")
        
        if not use_position_embeddings and has_postion_embeddings:
            raise ValueError("Position embedding is turned off but the current model has one.")

    if has_postion_embeddings:
        warn("The current model has <position_embeddings> enabled.")
    else:
        warn("The current model has NOT <position_embeddings> enabled.")

    return has_postion_embeddings
            

def load_model(model_class=None, model=None, checkpoint_path=None, verbose=False, remove=None, match=None, **kwargs):
    DEVICE = torch.device("cpu")
    if model is None:
        assert model_class is not None

        model = model_class(**kwargs)

    model = model.to(DEVICE)
    
    if checkpoint_path is not None:
        weights_dict = torch.load(checkpoint_path, map_location=DEVICE)

        for key in list(weights_dict):
            if match and not re.search(match, key):
                weights_dict.pop(key)
            elif remove:
                key2 = re.sub(remove, "", key)
                weights_dict[key2] = weights_dict.pop(key)
        # print(weights_dict.keys())
        try:
            model.load_state_dict(weights_dict, strict=True)
        except Exception as e:
            warn(f"Model loading in strict mode failed, will try in non-strict mode\n{str(e)[:500]}")
            model.load_state_dict(weights_dict, strict=False)

        if verbose:
            print(f"Weights loaded from: '{checkpoint_path}'")

    model = model.eval()
    return model

def get_model(model_name=None, task="token_classification", num_targets=None, config=None,
    pretrained=False, use_position_embeddings=True):
    num_targets = cfg.NUM_TARGETS if num_targets is None else cfg.NUM_TARGETS
    task = task.lower()
        
    if "token" in task:
        model_instance = AutoModelForTokenClassification
        
    elif "sequence" in task:
        model_instance = AutoModelForSequenceClassification

    if use_position_embeddings and config is None:
        assert model_name
    
        config = AutoConfig.from_pretrained(model_name)

    if use_position_embeddings:
        config.position_biased_input = True
        config.relative_attention = True
    elif config:
        config.position_biased_input = False


    if not pretrained:
        assert config is not None
        model = model_instance.from_config(config)
        tokenizer = None
    else:
        assert model_name is not None

        model = model_instance.from_pretrained(model_name, config=config)
        tokenizer = AutoTokenizer.from_pretrained(model_name, config=config)
        config = AutoConfig.from_pretrained(model_name, config=config)
    
    if hasattr(model, "classifier"):
        model.classifier = nn.Linear(model.classifier.in_features, num_targets)
        
    return config, tokenizer, model


class MultiSampleDropout(nn.Module):
    def  __init__(self, n_drops=None, p_drops=None):
        super().__init__()

        self.q_0 = 0.10
        self.q_1 = 0.50 - self.q_0

        self.n_drops = n_drops or cfg.N_DROPS
        self.p_drops = (p_drops or cfg.P_DROPS) or self.gen_dropout_probas()

        self.drop_modules = nn.ModuleList([nn.Dropout(p_drop) for p_drop in self.p_drops])

       
    def gen_dropout_probas(self):
        assert self.n_drops >= 0

        if self.n_drops == 0:
            return []
        elif self.n_drops == 1:
            return [self.q_0]
        else:
            return [ self.q_0 + self.q_1 * n / (self.n_drops -1) for n in range(self.n_drops)]
    
    def forward(self, x):
        if not self.training or not self.n_drops:
            return x[:, None]

        res = []
        for drop_module in self.drop_modules:
            res.append(drop_module(x))
        res = torch.stack(res, dim=1)
        return res


class BaseModel(nn.Module):
    def __init__(self, model_name=None, num_targets=None, config=None, checkpoint_path=None,
        pretrained=False, use_position_embeddings=True):
        super().__init__()
        self.model_name = cfg.MODEL_NAME if model_name is None else model_name
        self.num_targets = cfg.NUM_TARGETS if num_targets is None else num_targets
        self.use_position_embeddings = use_position_embeddings
        
        config, tokenizer, model = get_model(model_name=model_name, task="token_classification", config=config,
                num_targets=1, pretrained=pretrained, use_position_embeddings=use_position_embeddings)
        
        self.in_features =  model.classifier.in_features
        model.classifier = nn.Identity()
        
        self.config = config
        self.tokenizer = tokenizer
        self.model = model

        if checkpoint_path is not None:
            self.model = load_model(model=self.model, checkpoint_path=checkpoint_path, verbose=True,
            remove=r"^model\.", match=r"^model\.")

        self.fc = nn.Linear(self.in_features, self.num_targets)

    def forward(self, *, input_ids, attention_mask, word_ids):
        if self.use_position_embeddings:
            assert word_ids is not None
            position_ids = word_ids
        else:
            position_ids = None

        x = self.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)["logits"]
        x = self.fc(x)
        return x

class Model(BaseModel):
    _different_lr_s = []

    def __init__(self, model_name=None, num_targets=None, config=None, checkpoint_path=None,
            pretrained=False, use_position_embeddings=True):
        super().__init__(model_name=model_name, num_targets=num_targets,
        config=config, checkpoint_path=checkpoint_path, pretrained=pretrained, use_position_embeddings=use_position_embeddings)

        self.ms_dropout = MultiSampleDropout()

        self.fc_seg = nn.Linear(self.in_features, 3)

    
    def forward(self, *, input_ids, attention_mask, word_ids):
        if self.use_position_embeddings:
            assert word_ids is not None
            position_ids = word_ids
        else:
            position_ids = None

        x = self.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)["logits"]

        x = self.ms_dropout(x)

        out_seg = self.fc_seg(x)#.transpose(1, -1)

        out_ner = self.fc(x)#.transpose(1, -1)

        out = mTaskOutputType(
            out_ner=out_ner,
            out_seg=out_seg,
        )

        return out


class NERSegmentationLoss(nn.Module):
    def __init__(self, num_iters):
        super().__init__()

        self.num_iters = num_iters
        self.num_calls = 0

        self.ner_weights = torch.tensor(cfg.CLASS_WEIGHTS, device=cfg.DEVICE, dtype=torch.float32)
        self.seg_weights = torch.tensor(cfg.SEG_CLASS_WEIGHTS, device=cfg.DEVICE, dtype=torch.float32)

    def coef(self, k):
        return max(0, np.cos(np.pi*k / (2*self.num_iters)))

    def get_losses(self, num_calls=None):
        num_calls = self.num_calls if num_calls is None else num_calls
        q = self.coef(num_calls)
        ner_loss = nn.CrossEntropyLoss(weight=(1-q) + q*self.ner_weights)
        seg_loss = nn.CrossEntropyLoss(weight=(1-q) + q*self.seg_weights)
        return ner_loss, seg_loss


    @staticmethod
    def generate_seg_target(target):
        class_blind_labels = (
            1*( ( 1 <= target) & ( target < 8) ) # In
            + 2*( ( 8 <= target) & ( target < 15) ) # Beginning
            + 1*( ( 15 <= target) & ( target < 22) ) # In
        )
        if isinstance(class_blind_labels, torch.Tensor):
            class_blind_labels = class_blind_labels.long()

        bools = (target <= 0)
        class_blind_labels[bools] = target[bools]
        return class_blind_labels

    
    def forward(self, out_ner, out_seg, target, num_calls=None):

        ner_loss, seg_loss = self.get_losses(num_calls=num_calls)
        
        seg_labels = self.generate_seg_target(target)
        seg_labels = torch.stack([seg_labels]*out_seg.size(1), dim=1)
        target = torch.stack([target]*out_ner.size(1), dim=1)

        bools = (target >= 0)
        out_ner, out_seg, target, seg_labels = out_ner[bools], out_seg[bools], target[bools], seg_labels[bools]

        l_ner = ner_loss(out_ner, target)

        l_seg = seg_loss(out_seg, seg_labels)

        l = cfg.ALPHA_NER * l_ner + cfg.ALPHA_SEG * l_seg

        self.num_calls += 1

        return l