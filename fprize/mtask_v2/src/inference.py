
import pandas as pd, numpy as np

import torch
from  torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoConfig, AutoModel

from tqdm.auto import tqdm
from warnings import warn
import math

import pickle
from pathlib import Path
import gc
from functools import partial

from .post_processing import make_sub_from_res

from .utils import slugify
from . import configs as cfg
from .dataset import RunningMax, TestDataset, gen_data_from_id, get_append_bar_chars,\
    read_from_id, collate_fn, collate_fn_list, DynamicBatchDataset
from .models import Model, check_if_model_has_position_embeddings

def load_tokenizer_and_config(config_path, tokenizer_path, is_pickle=True):
    if is_pickle:
        with open(config_path, "rb") as f:
            config = pickle.load(f)

        with open(tokenizer_path, "rb") as f:
            tokenizer = pickle.load(f)
    else:
        config = AutoConfig.from_pretrained(config_path)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    return config, tokenizer


def load_net(checkpoint_path, param):
    config = param["config"]
    use_position_embeddings = param["use_position_embeddings"]

    if use_position_embeddings:
        config.position_biased_input = True
        config.relative_attention = True
    
    net = Model(config=config, pretrained=False, use_position_embeddings=use_position_embeddings)
    
    net = net.to(cfg.DEVICE)

    check_if_model_has_position_embeddings(net, use_position_embeddings=use_position_embeddings)
    
    if checkpoint_path is not None:
        net.load_state_dict(torch.load(checkpoint_path, map_location=cfg.DEVICE))
    net = net.eval()
    return net


def get_params(
    model_name,
    minlength=None,
    num_targets=None,
    maxlen=None,
    stride=None,
    is_pickle=True,
    tokenizer_path=None,
    config_path=None,
    model_paths=None,
    models=None,
    **kwargs,
    
):
    if maxlen is None:
        if model_name == "roberta-base":
            maxlen = 512
        elif model_name == "allenai/longformer-base-4096":
            maxlen = 1024
        elif "microsoft/deberta" in model_name:
            maxlen = 1024
#             maxlen = 512
        elif model_name == "roberta-large":
            maxlen = 512
        elif model_name == "google/bigbird-roberta-base":
            maxlen = 1024
        else:
            raise ValueError(f"Unknown mondel: {model_name}")
            

    model_name_slug = slugify(model_name)
    
    if stride is None:
        stride = maxlen // (1 if maxlen > 768 else (2 if maxlen > 512 else 4))

    if  torch.cuda.is_available():
        if model_name in ["roberta-large", "microsoft/deberta-v3-large"]:
            test_batch_size = 16
        else:
            test_batch_size = 64 if maxlen > 512 else 128
            
        test_batch_size = test_batch_size // (maxlen // stride)
        test_num_workers = 2
    else:
        test_batch_size = 2
        test_num_workers = 0

    config_path = config_path or Path(f"../input/fprize-kkiller-tools/{model_name_slug}-config.pkl")
    tokenizer_path = tokenizer_path or Path(f"../input/fprize-kkiller-tools/{model_name_slug}-tokenizer.pkl")

    config, tokenizer = load_tokenizer_and_config(config_path=config_path, tokenizer_path=tokenizer_path, is_pickle=is_pickle)
    
    params = {
        "model_name": model_name,
        "model_name_slug": model_name_slug,
        "minlength": minlength or cfg.MINLENGTH,
        "maxlen": maxlen,
        "num_targets": num_targets or cfg.NUM_TARGETS,
        "stride": stride,
        "test_batch_size": test_batch_size,
        "test_num_workers": test_num_workers, 
        "config_path": config_path,
        "tokenizer_path": tokenizer_path,
        "is_pickle": is_pickle,
        "config": config,
        "tokenizer": tokenizer,
        "model_paths": model_paths,
        "models": models,
    }
    
    params.update(kwargs)
         
    return params


def copy_param_to_configs(param):
    for attr, val in param.items():
        cfg_attr = None
        if hasattr(cfg, attr):
            cfg_attr = attr
        elif hasattr(cfg, attr.upper()):
            cfg_attr = attr.upper()

        if cfg_attr is not None:
            setattr(cfg, cfg_attr, val)


def get_max_pads(attention_mask):
    bools = (attention_mask[:, 1:] == 0) & (attention_mask[:, :-1] != 0)
    max_pads = bools.to(torch.uint8).argmax(1)

    if (max_pads == 0).any(): # There at least one full batch, no padding
        max_pads = attention_mask.shape[1]
    else:
        max_pads = 1 + max_pads.max().item()

    return max_pads

@torch.no_grad()
def _predict(nets, input_ids, attention_mask, word_ids, agg="mean",
            return_output=False, dynamic_padding=True, apply_softmax=False):

    if return_output:
        assert isinstance(nets, torch.nn.Module)
        nets = [nets]

    if len(nets) > 1:
        assert apply_softmax
    
    pred, pred_seg = None, None
    for net in nets:
        o_all = net(input_ids=input_ids, attention_mask=attention_mask, word_ids=word_ids)
        o, o_seg = o_all.out_ner, o_all.out_seg

        if apply_softmax:
            o, o_seg = o.softmax(dim=-1), o_seg.softmax(dim=-1)
        
        if  agg == "max":
            pred = o if pred is None else torch.max(o, pred)
            pred_seg = o_seg if pred_seg is None else torch.max(o, pred_seg)
        elif agg == "mean":
            pred = o if pred is None else pred.add_(o)
            pred_seg = o_seg if pred_seg is None else pred_seg.add_(o_seg)
        else:
            raise ValueError(f"Unknow value `{agg}` for `agg`")

    if agg == "mean":
        pred /= len(nets)
        pred_seg /= len(nets)

    return (pred, pred_seg, o_all) if return_output else (pred, pred_seg)
    

@torch.no_grad()
def predict_eval(net, test_data, bar=False, ret_out=True, dynamic_padding=True):
    rm = RunningMax()
    rm_seg = RunningMax()
    rm_target = RunningMax()

    if ret_out:
        out_list = []

    target_list = []

    test_data = tqdm(test_data) if bar else test_data

    apply_softmax = not ret_out
    
    for i_inp, inp in  enumerate(test_data):
        batch_ids, word_ids, word_perc_pos, input_ids, attention_mask = inp[:5]

        target = inp[5] if len(inp) == 6 else None

        if isinstance(input_ids, np.ndarray):
            input_ids = torch.from_numpy(input_ids)
            attention_mask = torch.from_numpy(attention_mask)
            word_perc_pos = torch.from_numpy(word_perc_pos)

        input_ids = input_ids.to(cfg.DEVICE)#.long()
        attention_mask = attention_mask.to(cfg.DEVICE)#.long()
        word_perc_pos = word_perc_pos.to(cfg.DEVICE)
        
        preds, preds_seg, out = _predict(net, input_ids=input_ids, attention_mask=attention_mask,
            word_ids=word_perc_pos, return_output=True, dynamic_padding=dynamic_padding, apply_softmax=apply_softmax)

        if not apply_softmax:
            preds, preds_seg = preds.softmax(dim=-1), preds_seg.softmax(dim=-1)

        preds, preds_seg = preds.squeeze(1).cpu().numpy(), preds_seg.squeeze(1).cpu().numpy()

        if dynamic_padding and (preds.shape[1] < word_ids.shape[1]):
            word_ids = word_ids[:, :preds.shape[1]]
            if target is not None:
                target = target[:, :preds.shape[1]]

        if ret_out:
            out_list.append(out)

        if target is not None:
            target_list.append(target)

        bools = (word_ids >= 0)

        batch_ids = batch_ids[:, None].repeat(word_ids.shape[1], axis=1)[bools]
        word_ids = word_ids[bools]
        
        reduce = (i_inp % 200 == 0)

        rm.update(ids=np.c_[batch_ids, word_ids],
                  vals=preds[bools], reduce=reduce)

        rm_seg.update(ids = np.c_[batch_ids, word_ids],
                  vals=preds_seg[bools], reduce=reduce)

        if target is not None:
            rm_target.update(ids=np.c_[batch_ids, word_ids],
                    vals=target[bools][:, None].astype(np.int8, copy=False), reduce=reduce)
    
    if len(rm._vals):
        rm.reduce()
        rm_seg.reduce()
        if target is not None:
            rm_target.reduce()

    preds = rm.normlaize().vals

    preds_seg = rm_seg.normlaize().vals
    
    target_v2 = rm_target.vals if target is not None else None

    if dynamic_padding:
        out = out_list if ret_out else None
        target = target_list if target is not None else None
    else:
        out = tuple(map(torch.cat, zip(*out_list))) if ret_out else None
        target = torch.from_numpy(np.concatenate(target_list)).to(cfg.DEVICE) if target is not None else None

    res = {"out": out, "target": target, "preds": preds, "preds_seg": preds_seg, "target_v2": target_v2}
    return res


def predict_from_param(uuids, param, data=None, oof=False, make_sub=True, dynamic_padding=True, queue=None, 
    model_bar=False, reduce="mean"):
    copy_param_to_configs(param)

    assert not oof or not make_sub

    warn("You should sort your UUIDs for faster prediction")

    tokenizer=param["tokenizer"]
    pad_token_id=tokenizer.pad_token_id,
    special_token_ids={"bos_token_id": tokenizer.bos_token_id, "eos_token_id": tokenizer.eos_token_id}

    if data is None:
        append_bar = get_append_bar_chars(tokenizer=tokenizer)
        data = {uuid: gen_data_from_id(uuid, tokenizer=tokenizer, root=param["root"], append_bar=append_bar) for uuid  in tqdm(uuids)}

    test_data = TestDataset(uuids=uuids, data=data, pad_token_id=pad_token_id, tokenizer=tokenizer,
                root=param["root"], stride=param["stride"], special_token_ids=special_token_ids)

    sizes = [ math.ceil( ( max( len(data[uuid][0]) - test_data.maxlen + 2 , 0) + test_data.stride ) / test_data.stride ) for uuid in uuids]

    test_data = DynamicBatchDataset(test_data, batch_size=max(1, param["batch_size"] // max(1, param["num_workers"])), sizes=sizes)

    test_loader = DataLoader(test_data, batch_size=max(1, param["num_workers"]), num_workers=param["num_workers"],
                shuffle=False, collate_fn=partial(collate_fn_list, pad_token_id=pad_token_id))

    if reduce == "mean":
        preds, preds_seg = None, None
    else:
        assert reduce is None
        assert not make_sub
        results = []

    for model_path in tqdm(param["model_paths"]):
        model = load_net(model_path, param)
        res = predict_eval(model, test_loader, bar=model_bar, ret_out=False, dynamic_padding=dynamic_padding)
        
        if reduce == "mean":
            if preds is None:
                preds = res["preds"]
                preds_seg = res["preds_seg"]
            else:
                preds += res["preds"]
                preds_seg += res["preds_seg"]
        else:
            results.append((res["preds"], res["preds_seg"]))

        del model, res

        gc.collect()
        torch.cuda.empty_cache()

    if reduce == "mean":      
        preds /= len(param["model_paths"])
        preds_seg /= len(param["model_paths"])
    else:
        if queue is None:
            return results
        else:
            queue.put(results)

    if make_sub:
    
        sub = make_sub_from_res(
            uuids=uuids,
            res=preds,
            res_seg=preds_seg,
        )

        if queue is None:
            return preds, preds_seg, sub
        else:
            queue.put((preds, preds_seg, sub))

    if queue is None:
        return preds, preds_seg
    else:
        queue.put((preds, preds_seg))


def mp_predict_from_param(uuids, param, data=None, oof=False, make_sub=True, dynamic_padding=True, model_bar=False):

    import multiprocessing as mp
    
    context = mp.get_context('spawn')

    queue = mp.Queue()

    p = context.Process(
        target=predict_from_param,
        kwargs=dict(
            uuids=uuids,
            param=param,
            data=data,
            oof=oof,
            make_sub=make_sub,
            dynamic_padding=dynamic_padding,
            queue=queue,
            model_bar=model_bar,
        )
    )

    p.start()
    p.join()

    return queue.get()