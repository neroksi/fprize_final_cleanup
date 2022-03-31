
import numpy as np
from pathlib import Path

import torch
from torch import nn, optim
from  torch.utils.data import DataLoader
from  torch.nn.utils import clip_grad_norm_
from torch.cuda import amp

from transformers import AdamW, get_cosine_schedule_with_warmup

from matplotlib import pyplot as plt

import os, random, gc
import re, time, json, pickle, math
from copy import deepcopy

from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score

from sklearn.model_selection import KFold, GroupKFold

from tqdm.notebook import tqdm

from collections import defaultdict
from warnings import warn
from  functools import partial

from .dataset import Dataset, TestDataset, read_from_id, DynamicBatchDataset, collate_fn, collate_fn_list, collate_fn_train
from .models  import NERSegmentationLoss, Model, load_model as models_load_model, check_if_model_has_position_embeddings, AutoConfig
from . import configs as  cfg
from .utils import slugify, seed_everything, get_config_as_param
from .inference import predict_eval, get_max_pads
from .post_processing import to_1_7, make_sub_from_res, default_threshs, get_seg_from_ner
from .comp_metric import score_feedback_comp


average = "macro"
multi_class = "ovo" # "ovr"
zero_division = 0

def disable_tokenizer_parallelism():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_cv_score(model_path):
    log_paths = sorted(Path(model_path).glob("*.json"))
    scores = {}
    for path in log_paths:
        with open(path) as f:
            log = json.load(f)
            fold = int(str(path).split("_fold")[1][0])
            scores[f"fold_{fold}"] = max([l["f1_val"] for l in log])
            
    scores["mean_f1"] = sum(scores.values())/len(scores)
    
    return scores


def one_step(inputs, net, criterion, optimizer, scheduler=None, scaler=None):
    if len(inputs) == 4:
        word_perc_pos, input_ids, attention_mask, target = inputs
    else:
        batch_ids, word_ids, word_perc_pos, input_ids, attention_mask, target = inputs

    if isinstance(input_ids, np.ndarray):
        input_ids = torch.from_numpy(input_ids)
        attention_mask = torch.from_numpy(attention_mask)
        target = torch.from_numpy(target)
        word_perc_pos = torch.from_numpy(word_perc_pos)

    xb, yb = (input_ids, attention_mask), target

    xb, yb = (xb[0].to(cfg.DEVICE), xb[1].to(cfg.DEVICE)), yb.to(cfg.DEVICE)

    word_perc_pos = word_perc_pos.to(cfg.DEVICE)

    optimizer.zero_grad(set_to_none=True)

    use_amp = scaler is not None

    if use_amp:
        with amp.autocast():
            o = net(input_ids=xb[0], attention_mask=xb[1], word_ids=word_perc_pos)

            loss = criterion(
                *o,
                target=yb,
            )
    else:
        o = net(input_ids=xb[0], attention_mask=xb[1], word_ids=word_perc_pos)

        loss = criterion(
            *o,
            target=yb,
        )

    if use_amp:
        scaler.scale(loss).backward()
        # Unscales the gradients of optimizer's assigned params in-place
        scaler.unscale_(optimizer)

        # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
        clip_grad_norm_(net.parameters(), cfg.CLIP_GRAD_NORM)

        scaler.step(optimizer)

        # Updates the scale for next iteration.
        scaler.update()
    else:
        loss.backward()
        optimizer.step()

    
    with torch.no_grad():
      
        l = loss.item()
        if np.isnan(l):
            l = 0.

        o = o[0].mean(1)

        o = o.argmax(-1).cpu().numpy().ravel()
        yb = yb.cpu().numpy().ravel()

        sel = (yb >= 0)
        o = o[sel]
        yb = yb[sel]

        acc = accuracy_score(yb, o)

        prec = precision_score(yb, o, average=average, zero_division=zero_division)
        rec = recall_score(yb, o, average=average, zero_division=zero_division)
        f1 = f1_score(yb, o, average=average, zero_division=zero_division)
          
    if  scheduler is not None:
        scheduler.step()

    metrics = {
        "loss": l,
        "acc": acc,
        "f1": f1,
        "prec": prec,
        "rec": rec,
    }

    return metrics


def get_metrics(out, target):
    acc = accuracy_score(target, out)

    prec = precision_score(target, out, average=average, zero_division=zero_division)
    rec = recall_score(target, out, average=average, zero_division=zero_division)
    f1 = f1_score(target, out, average=average, zero_division=zero_division)

    metrics = {
        "acc": acc,
        "f1": f1,
        "prec": prec,
        "rec": rec,
    }

    return metrics

def get_macro_f1_score(uuids, df, res):
    def return_default():
        macrof1_score = -1.0
        class_scores = {}
        macrof1_score_v2 = -1.0
        class_scores_v2 = {}
        return (macrof1_score, class_scores), (macrof1_score_v2, class_scores_v2)

    preds = res["preds"]
    res_seg = res["preds_seg"]
    threshs = deepcopy(default_threshs)

    try:
        sub = make_sub_from_res(uuids=uuids, res=preds, res_seg=res_seg, threshs=threshs)

        res_seg_v2 = cfg.TRUE_SEG_COEF*res_seg + (1 - cfg.TRUE_SEG_COEF) * get_seg_from_ner(preds)

        sub_v2 = make_sub_from_res(uuids=uuids, res=preds, res_seg=res_seg_v2, threshs=threshs)
    except Exception as e:
        warn(f"EXCEPTION DURING SUB BUILDING !!!\n{e}")
        return return_default()

    if  len(sub):
        try:
            macrof1_score, class_scores = score_feedback_comp(
                sub, df, return_class_scores=True
            )

            macrof1_score_v2, class_scores_v2 = score_feedback_comp(
                sub_v2, df, return_class_scores=True
            )
        except Exception as e:
            warn(f"EXCEPTION DURING MACRO F1 COMPUTATION !!!\n{e}")
            return return_default()
    else:
        warn("empty discourse prediction, f1_score can't be computed !", UserWarning)

        return return_default()

    return (macrof1_score, class_scores), (macrof1_score_v2, class_scores_v2)


@torch.no_grad()
def evaluate(net, criterion, val_loader, df):
    net.eval()

    uuids = val_loader.dataset.dataset.uuids
    df = df[df["id"].isin(uuids)].reset_index(drop=True)
    val_loader = tqdm(val_loader, leave = False, total=len(val_loader))

    res = predict_eval(net, val_loader, ret_out=True, dynamic_padding=True)
    o = res["out"]
    y = res["target"]
    preds = res["preds"]
    target_v2 = res["target_v2"]

    (macrof1_score, class_scores), (macrof1_score_v2, class_scores_v2) = get_macro_f1_score(uuids, df=df, res=res)

    l = 0.

    for oo, yy in zip(o, y):
        l += criterion(
            *oo,
            target=torch.from_numpy(yy).to(cfg.DEVICE),
        ).item()
    
    l /= len(o)
    
    oof = {
        "out": preds.astype(np.float16),
        "out_seg": res["preds_seg"].astype(np.float16),
        "target": target_v2.astype(np.int8),
        "uuids": uuids,
    }

    metrics = {}

    metrics.update(
        iov=macrof1_score,
        iov_classes=class_scores,
        iov_v2=macrof1_score_v2,
        iov_classes_v2=class_scores_v2,
        loss=l,
    )

    o_v2 = preds.values#.argmax(1)
    target_v2 = target_v2.values[:, 0]

    o_v2 = o_v2.argmax(1)

    metrics.update(get_metrics(o_v2, target_v2))

    return oof, metrics


def finalize_epoch(*, net, criterion, val_loader, df, icount, train_metrics, **kwargs):

    metrics = deepcopy(train_metrics)
    div_factor = max(1, icount)
    for key in list(metrics):
         metrics[key] /= div_factor
         

    oof, metrics_val = evaluate(net, criterion, val_loader, df=df)

    for key, val in metrics_val.items():
        metrics[f"{key}_val"] = val

    
    print_and_save(
        oof=oof,
        metrics=metrics,
        net=net,
        criterion=criterion,
        **kwargs,
    )

    return oof, metrics

def one_epoch(*, net, criterion, optimizer, scheduler, train_loader, val_loader, df,
    schedule_each_step=False, scaler=None, save_each=None, **kwargs):
    net.train()
    icount = 0
    num_iter = len(train_loader)
    train_loader = tqdm(train_loader, leave = False)
    epoch_bar = train_loader

    metrics_format_dict = dict(
        loss="{loss:.6f}",
        acc="{acc:.3f}",
        prec="{prec:.3f}",
        rec="{rec:.3f}",
        f1="{f1:.3f}",
    )

    if save_each is not None:
        save_step = save_each*num_iter
        save_points = np.arange(save_step, num_iter, save_step).astype(int)
    else:
        save_points = []

    metrics = defaultdict(int)

    for step, inputs in  enumerate(epoch_bar):

        _metrics = one_step(inputs, net=net, criterion=criterion, optimizer=optimizer, scaler=scaler)

        if schedule_each_step:
            scheduler.step()

        for key, val in _metrics.items():
            metrics[key] += val

        icount += 1
          
        if hasattr(epoch_bar, "set_postfix") and not icount%10:
            metrics_normalized = {key: val / icount for key, val in metrics.items()}
            metrics_formated = {
                key: val.format(**metrics_normalized) for key,val in metrics_format_dict.items()
            }
            
            epoch_bar.set_postfix(**metrics_formated)

        
        if step in save_points:
            oof, metrics_temp = finalize_epoch(
                net=net,
                criterion=criterion,
                val_loader=val_loader,
                df=df,
                icount=icount,
                train_metrics=metrics,
                optimizer=optimizer,
                scheduler=scheduler,
                **kwargs,
            )

        
    if not schedule_each_step:
        scheduler.step()


    oof, metrics = finalize_epoch(
        net=net,
        criterion=criterion,
        val_loader=val_loader,
        df=df,
        icount=icount,
        train_metrics=metrics,
        optimizer=optimizer,
        scheduler=scheduler,
        **kwargs,
    )

    return oof, metrics


class AutoSave:
    def __init__(self, top_k=1, metric="f1", mode="min", root=None, name="ckpt"):
        self.top_k = top_k
        self.logs = []
        self.metric = metric
        self.mode = mode
        self.root = Path(root or cfg.MODEL_ROOT)
        assert self.root.exists()
        self.name = name

        self.top_models = []
        self.top_metrics = []

        self.slug_regex = r"[^\w_\-\.]"
        self.oof_suffix = "oof"

        self._log_path = None

    
    def slugify(self, s):
        return re.sub(self.slug_regex, "", s)

    def log(self, model, metrics, oof=None):
        metric = metrics[self.metric]
        rank = self.rank(metric)

        self.top_metrics.insert(rank+1, metric)
        if len(self.top_metrics) > self.top_k:
          self.top_metrics.pop(0)

        self.logs.append(metrics)
        self.save(model, metric, rank, metrics["epoch"], oof=oof)


    def oof_path_from_model_path(self, model_path):
        oof_name = model_path.parent / "{}_{}.pkl".format(
            model_path.stem, self.oof_suffix
        )
        return oof_name
        

    def save(self, model, metric, rank, epoch, oof=None):
        t = time.strftime("%Y%m%d%H%M%S")
        name = "{}_epoch_{:02d}_{}_{:.04f}_{}".format(self.name, epoch, self.metric, metric, t)
        name = self.slugify(name) + ".pth"
        path = self.root.joinpath(name)

        old_model = None
        self.top_models.insert(rank+1, name)
        if len(self.top_models) > self.top_k:
          old_model = self.root.joinpath(self.top_models[0])
          self.top_models.pop(0)      

        torch.save(model.state_dict(), path.as_posix())


        if oof is not None:
            with  self.oof_path_from_model_path(path).open(mode="wb") as f:
                pickle.dump(oof, f)

        if old_model is not None:
          old_model.unlink()
          old_oof = self.oof_path_from_model_path(old_model)
          if old_oof.exists():
              old_oof.unlink()

        self.to_json()

        

    def rank(self, val):
        r = -1
        for top_val in self.top_metrics:
          if val <= top_val:
            return r
          r += 1

        return r

    @property
    def log_path(self):
        if self._log_path is  None:
            t = time.strftime("%Y%m%d%H%M%S")
            name = "{}_{}_logs".format(self.name, t)
            name =  self.slugify(name) + ".json"
            self._log_path = self.root.joinpath(name)

        return self._log_path

    
    def to_json(self):

        with self.log_path.open(mode="w") as f:
            data = {
                "log": self.logs,
                "params": get_config_as_param(cfg),
            }
            json.dump(data, f, indent=2, default=str)


def fetch_optimizer(net):
    return optim.AdamW(net.parameters(), lr=cfg.OPTIMIZER_LR, weight_decay=cfg.OPTIMIZER_WEIGHT_DECAY)

def fetch_scheduler(optimizer, num_train_steps):
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * num_train_steps),
        num_training_steps=num_train_steps,
        num_cycles=1,
        last_epoch=-1,
    )
    return scheduler


def prepare_train_data(uuids, data, train_set, data_kwargs, shuffle=False, max_item=None):
    train_data = TestDataset(uuids=uuids[train_set] , data=data, is_train=True, **data_kwargs)
    max_item = 4 if (train_data.maxlen <= 512) else 2
    train_sizes = [ math.ceil( ( max( len(data[uuid][0]) - train_data.maxlen + 2 , 0) 
                            + train_data.stride ) / train_data.stride ) for uuid in train_data.uuids]
    train_sizes = [size if max_item is None else min(size, max_item) for size in train_sizes]
    train_data = DynamicBatchDataset(train_data, batch_size=max(1, cfg.TRAIN_BATCH_SIZE // max(1, cfg.TRAIN_NUM_WORKERS) ), sizes=train_sizes)
    train_loader = DataLoader(train_data, batch_size=max(1, cfg.TRAIN_NUM_WORKERS), num_workers=cfg.TRAIN_NUM_WORKERS,
    shuffle=shuffle, pin_memory=True, drop_last=False,
        collate_fn=partial(collate_fn_list, pad_token_id=data_kwargs["pad_token_id"], max_item=max_item) )

    return train_data, train_loader

def prepare_val_data(uuids, data, val_set, data_kwargs, max_item=4):
    val_set = sorted(val_set, key=lambda idx: -len(read_from_id(uuids[idx]).split()) )
    val_data = TestDataset(uuids=uuids[val_set], data=data, is_train=False, **data_kwargs)
    max_item = 4 if (val_data.maxlen <= 512) else 2
    val_sizes = [ math.ceil( ( max( len(data[uuid][0]) - val_data.maxlen + 2 , 0) 
                            + val_data.stride ) / val_data.stride ) for uuid in val_data.uuids]
    val_sizes = [size if max_item is None else min(size, max_item) for size in val_sizes]
    val_data = DynamicBatchDataset(val_data, batch_size=max(1, cfg.VAL_BATCH_SIZE // max(1, cfg.VAL_NUM_WORKERS) ), sizes=val_sizes)
    val_loader = DataLoader(val_data, batch_size=max(1, cfg.VAL_NUM_WORKERS), num_workers=cfg.VAL_NUM_WORKERS,
        collate_fn=partial(collate_fn_list, pad_token_id=data_kwargs["pad_token_id"], max_item=max_item), shuffle=False)

    return val_data, val_loader



def print_and_save(fold, oof, metrics, epoch, net, optimizer, criterion, scheduler, val_set, train_set, epochs_bar, do_save, saver):
    metrics["epoch"] = epoch
    metrics["learning_rates"] = optimizer.param_groups[0]["lr"]

    # print(metrics)

    oof["val_set"] = val_set
    oof["train_set"] = train_set
    oof["fold"] = fold
    oof["criterion"] = criterion.__class__.__name__
    oof["model"] = net.__class__.__name__
    oof["scheduler"] = scheduler.__class__.__name__

    metrics_format_dict = dict(
        loss="({loss:.6f}, {loss_val:.6f})",
        iov="(-1., {iov_val:.3f})",
        iov2="(-1., {iov_v2_val:.3f})",
        # auc="(-1., {auc_val:.3f})",
        acc="({acc:.3f}, {acc_val:.3f})",
        prec="({prec:.3f}, {prec_val:.3f})",
        rec="({rec:.3f}, {rec_val:.3f})",
        f1="({f1:.3f}, {f1_val:.3f})",
    )

    metrics_print_format = "[{epoch:02d}] loss: {loss} iov: {iov} iov2: {iov2} acc: {acc} f1: {f1} rec: {rec} prec: {prec}"

    metrics_formated = {key: val.format(**metrics) for key, val in metrics_format_dict.items()}

    epochs_bar.set_postfix(**metrics_formated)

    print(
        metrics_print_format.format(epoch=epoch, **metrics_formated)
    )

    if do_save:
        saver.log(net, metrics, oof=oof)
    


def one_fold(*, uuids, data, model_name, df, fold, train_set, val_set, epochs=20, save=True, save_root=None,
    checkpoint_paths=None, use_stride_during_train=False, use_position_embeddings=True, save_each=None,
    early_stop_epoch=None, **data_kwargs):

    model_name_slug = slugify(model_name)

    save_root = Path(save_root) or cfg.MODEL_ROOT

    saver = AutoSave(root=save_root, name=f"fprize_{model_name_slug}_fold{fold}", metric=cfg.MAIN_METRIC_NAME)
    
    checkpoint_path=(checkpoint_paths or {}).get(f"fold_{fold}")
    net = Model(model_name, checkpoint_path=checkpoint_path, pretrained=True, use_position_embeddings=use_position_embeddings)
    check_if_model_has_position_embeddings(net, use_position_embeddings=use_position_embeddings)

    try:
        net = models_load_model(model=net, checkpoint_path=checkpoint_path, verbose=True)
    except Exception as e:
        print("Second Load ERRORRRRRRRRRR:\n", str(e)[:500])

    net = net.to(cfg.DEVICE)
    
    epochs_bar = tqdm(list(range(epochs if early_stop_epoch is None else early_stop_epoch)), leave=False)

    if use_stride_during_train:
        _, train_loader = prepare_train_data(uuids=uuids, data=data, train_set=train_set, data_kwargs=data_kwargs, shuffle=False)
    else:
        train_data = Dataset(uuids=uuids[train_set] , data=data, is_train=True, **data_kwargs)
        train_loader = DataLoader(train_data, batch_size=cfg.TRAIN_BATCH_SIZE, num_workers=cfg.TRAIN_NUM_WORKERS, shuffle=True,
                pin_memory=True, drop_last=True, collate_fn=partial(collate_fn_train, pad_token_id=data_kwargs["pad_token_id"]))

    _, val_loader = prepare_val_data(uuids=uuids, data=data, val_set=val_set, data_kwargs=data_kwargs)


    num_iters = len(train_loader)*epochs
    criterion = NERSegmentationLoss(num_iters= int(0.75*num_iters))
    optimizer = fetch_optimizer(net)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=cfg.SCHEDULER_ETA_MIN, T_max=num_iters)
    schedule_each_step = True

    if cfg.USE_AMP and ("cuda" in str(cfg.DEVICE)):
        scaler = amp.GradScaler()
        warn("amp and fp16 are enabled !")
    else:
        scaler = None


    for epoch in epochs_bar:

        epochs_bar.set_description(f"--> [EPOCH {epoch:02d}]")
        net.train()

        oof, metrics = one_epoch(
            net=net,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            train_loader=train_loader,
            val_loader=val_loader,
            df=df,
            schedule_each_step=schedule_each_step,
            save_each=save_each,

            # For print and save
            fold=fold,
            epoch=epoch,
            val_set=val_set,
            train_set=train_set,
            epochs_bar=epochs_bar,
            do_save=save,
            saver=saver,
        )
        
        if use_stride_during_train:
        # We need to redefine train data with custom shuffle because data loader shuffling is disabled cause of DynamicBatchDataset
            _, train_loader = prepare_train_data(uuids=uuids, data=data, 
                                train_set=train_set[np.random.permutation(len(train_set))], data_kwargs=data_kwargs, shuffle=False)


def _train(*, uuids, data, model_name, df, epochs=20, save=True, n_splits=5, seed=None, save_root=None, suffix="", folds=None,
        checkpoint_paths=None, use_stride_during_train=False, use_position_embeddings=True, save_each=None, **data_kwargs):
    # print(get_config_as_param(cfg))

    gc.collect()
    torch.cuda.empty_cache()

    seed = cfg.SEED if seed is None else seed
    
    model_name_slug = slugify(model_name)
    save_root = save_root or cfg.MODEL_ROOT/f"{model_name_slug}{suffix}"
    save_root.mkdir(exist_ok=True, parents=True)

    seed_everything(seed)

    if isinstance(uuids, dict):
        fold_bar = []
        n_splits = max(uuids.values()) + 1

        for fold in range(n_splits):
            train_set = np.array(
                [i for i, fold_i in enumerate(uuids.values()) if fold_i != fold], dtype=np.int64
            )

            val_set = np.array(
                [i for i, fold_i in enumerate(uuids.values()) if fold_i == fold], dtype=np.int64
            )

            fold_bar.append((train_set, val_set))

        uuids = np.array(list(uuids.keys()))
    else:
        kf = KFold(n_splits=n_splits, random_state=seed, shuffle=True)
        fold_bar = list(kf.split(np.arange(len(uuids))))

    if folds:
        fold_bar = tqdm([(fold, fold_bar[fold]) for fold in folds])
    else:
        fold_bar = tqdm(enumerate(fold_bar), total=n_splits)
    
    for fold, (train_set, val_set) in fold_bar:
        
        print(f"\n############################### [FOLD {fold}  SEED {seed}]")
        fold_bar.set_description(f"[FOLD {fold}  SEED {seed}]")

        one_fold(uuids=uuids, data=data, model_name=model_name, df=df, fold=fold, train_set=train_set, val_set=val_set, epochs=epochs, 
        save=save, save_root=save_root, checkpoint_paths=checkpoint_paths, use_stride_during_train=use_stride_during_train,
        use_position_embeddings=use_position_embeddings, save_each=save_each, **data_kwargs)
      
        gc.collect()
        torch.cuda.empty_cache()


def train(uuids, data, model_name, df, epochs=20, save=True, n_splits=5, seed=None, save_root=None, suffix="", 
        folds=None, checkpoint_paths=None, use_stride_during_train=False, **data_kwargs):

    import multiprocessing as mp
    
    context = mp.get_context('spawn')

    p = context.Process(
        target=_train,
        kwargs=dict(
            uuids=uuids,
            data=data,
            model_name=model_name,
            df=df,
            epochs=epochs,
            save=save,
            n_splits=n_splits,
            seed=seed,
            save_root=save_root,
            suffix=suffix,
            folds=folds,
            checkpoint_paths=checkpoint_paths,
            use_stride_during_train=use_stride_during_train,
            **data_kwargs,
            )
        )

    p.start()
    p.join()