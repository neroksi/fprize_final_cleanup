import numpy as np, pandas as pd
from collections import Counter
from itertools import chain as it_chain
from warnings import warn

import sys
sys.setrecursionlimit(2048)

from . import configs as cfg
from .configs import to_1_7

default_threshs = {
    'span_min_score': {
        'Claim': 0.43,
        'Concluding Statement': 0.44,
        'Counterclaim': 0.37,
        'Evidence': 0.42,
        'Lead': 0.38,
        'Position': 0.38,
        'Rebuttal': 0.36
    },
    'span_min_len': {
        'Claim': 3,
        'Concluding Statement': 10,
        'Counterclaim': 7,
        'Evidence': 12,
        'Lead': 8,
        'Position': 5,
        'Rebuttal': 5
    },
    'consecutive_span_min_score': {
        'Claim': 0.45,
        'Concluding Statement': float("+Inf"),
        'Counterclaim': float("+Inf"),
        'Evidence': 0.45,
        'Lead': float("+Inf"),
        'Position': float("+Inf"),
        'Rebuttal': float("+Inf")
    },
    'consecutive_span_min_len': {
        'Claim': 3,
        'Concluding Statement': float("+Inf"),
        'Counterclaim': float("+Inf"),
        'Evidence': 14,
        'Lead': float("+Inf"),
        'Position': float("+Inf"),
        'Rebuttal': float("+Inf")
    }
}

def get_most_common(L):
    return Counter(L).most_common(1)[0][0]

def get_seg_from_ner(res_ner):
    res_seg = res_ner.copy(deep=False)
    cols = res_seg.columns

    cols = (
            1*( ( 1 <= cols) & ( cols <= cfg.NUM_PURE_TARGETS) ) # In
            + 2*( ( cfg.NUM_PURE_TARGETS < cols) & ( cols <= 2*cfg.NUM_PURE_TARGETS) ) # Beginning
            + 1*( ( 2*cfg.NUM_PURE_TARGETS < cols) & ( cols <= 3*cfg.NUM_PURE_TARGETS) ) # In
    )

    res_seg.columns = cols
    res_seg = res_seg.groupby(level=0, axis=1).sum()
    return res_seg

class SpanGetter:
    def __init__(self, seg_target, pred_target=None, step=3, max_span_len=None, min_target_score=0,
        start_checking_window=3, end_checking_window=1):
        assert step > 0
        
        if pred_target is not None:
            assert len(seg_target) == len(pred_target)
            # assert pred_target.shape[1] == cfg.NUM_TARGETS

        self.seg_target = np.asarray(seg_target).astype(np.float32, copy=False)
        self.pred_target = np.asarray(pred_target).astype(np.float32, copy=False) if pred_target is not None else None
        self.pred_target_argmax = None if self.pred_target  is None else [to_1_7(i) for i  in self.pred_target.argmax(1)]

        assert self.seg_target.shape[1] == 3
        assert self.seg_target.ndim == 2

        self.step = step
        self.max_span_len = float("+Inf") if max_span_len is None else max_span_len
        self.min_target_score = min_target_score
        self.start_checking_window = start_checking_window
        
        self.end_checking_window = end_checking_window

        self.nrows = len(seg_target)

    def find_next_start(self, pos, previous_end):
        if pos >= self.nrows-1:
            return
        
        while (pos < self.nrows) and (self.seg_target[pos].argmax() == 0):
            pos += 1
            
        if self.could_be_start(pos):
            return pos
            
        return self.find_next_start(pos=pos+1, previous_end=previous_end)
    
    def find_next_end(self, pos, previous_start):
        assert pos >= previous_start
#         i = previous_start + self.step
        if pos >= self.nrows:
            return self.nrows
        
        if self.could_be_end(pos):
            return pos
        
        if (pos - previous_start) >= self.max_span_len:
            return previous_start + self.max_span_len
            
        return self.find_next_end(pos=pos+1, previous_start=previous_start)
            
    def could_be_start(self, i):
        # assert i < len(self.seg_target)
        if i >= self.nrows-1:
            return False
        
        score = self.seg_target[i:i+self.start_checking_window].mean(0)
        
        return score.argmax() in [1, 2]
    
    def could_be_end(self, i):
        # assert i < len(self.seg_target)

        if i >= self.nrows:
            return True
        
        score = self.seg_target[max(0, i-self.end_checking_window+1, i):i+1].mean(0)
        
        return score.argmax() in [0, 2]

    def one_span(self, start=0):
        start = self.find_next_start(pos=start, previous_end=start)
        if start is None:
            return
        
        end = self.find_next_end(pos=start+self.step, previous_start=start)

        span = {
            'start': start,
            'end': end,
            'class_id': -1 if self.pred_target_argmax is None else get_most_common(self.pred_target_argmax[start:end]),
            'num_tokens': end-start
        }
        return span
    
    def all_spans(self):
        start = 0
        spans = []
        while start < self.nrows:
            span = self.one_span(start=start)
            if span is not None:
                spans.append(span)
                start = span["end"]
            else:
                break
        return spans


def get_spans_from_seg_v3(seg_target, pred_target=None):
    return SpanGetter(seg_target=seg_target, pred_target=pred_target).all_spans()


class SpanRepairer:
    def __init__(self, 
                span_min_len, span_min_score, consecutive_span_min_len,
                 consecutive_span_min_score, max_iter=None):
        self.span_min_len = span_min_len
        self.span_min_score = span_min_score
        self.consecutive_span_min_len = consecutive_span_min_len
        self.consecutive_span_min_score = consecutive_span_min_score
        
        self.max_iter = max_iter or 1_000
    
    @staticmethod
    def merge_spans(span1, span2, copy=False):
        span = span1.copy() if copy  else span1
        
        span["start"] = min(span1["start"], span2["start"])
        span["end"] = max(span1["end"], span2["end"])


        span["score"] = 0.5 * (span1["score"] + span2["score"])
        span["num_tokens"] = span["end"] - span["start"]
        
        return span
    
    def is_span_ok(self, span):
        return ( (span["num_tokens"] >= self.span_min_len[span["class"]])
                and (span["score"] >= self.span_min_score[span["class"]])
        )
        
    def is_consecutive_spans_ok(self, span, last_span):
        assert span["class"] == last_span["class"]
        
        return (
            (span["num_tokens"] >= self.consecutive_span_min_len[span["class"]])
            and (last_span["num_tokens"] >= self.consecutive_span_min_len[last_span["class"]])
            and (span["score"] >= self.consecutive_span_min_score[span["class"]])
            and (last_span["score"] >= self.consecutive_span_min_score[last_span["class"]])
        )
    
    def move_cusrsor(self, span1, span2, copy=False):
        if span1["class"] == span2["class"]:
            if self.is_span_ok(span1) and self.is_span_ok(span2) and self.is_consecutive_spans_ok(span1, span2):
                spans, num_step = (span1, ), 1
                
            else:
#                 print("MERGE")
                spans, num_step = (self.merge_spans(span1, span2, copy=copy), ), 2
        else:
            if self.is_span_ok(span1):
                spans, num_step = (span1,), 1
            else:
                spans, num_step = (), 1
        
        return spans, num_step
    
    def smart_repair(self, spans, res_list=None, copy=False):
        if res_list is None:
            res_list = []
            
        if len(spans) == 0:
            return res_list
            
        if len(spans) == 1:
            if self.is_span_ok(spans[0]):
                res_list.append(spans[0])
            
            return res_list
        
        span1, span2 = spans[0], spans[1]
        
        spans_to_append, num_step = self.move_cusrsor(span1, span2, copy=copy)
        
        assert num_step > 0
        
        res_list.extend(spans_to_append)
        
        for _ in range(num_step):
            spans.pop(0)
        
        return self.smart_repair(spans, res_list=res_list, copy=copy)
    
    def __call__(self, spans, copy=False):
        max_iter = self.max_iter or float("+Inf")
        i = 0
        
        while i < max_iter:
            n = len(spans)
            spans = self.smart_repair(spans, copy=copy)
            
            if len(spans) >= n:
                break

            i += 1
            
        return spans
        
    def repair_consecutive_spans(self, spans, copy=False):
        if len(spans) < 2:
            return spans
        
        last_span = spans[0]
        new_spans = [last_span]
        for i, span in enumerate(spans[1:], 1):
            if span["class"] == last_span["class"]:
                if self.is_consecutive_spans_ok(span, last_span):
                    new_spans.append(span)
                    
                else:
                    # print(i)
                    span = self.merge_spans(span, last_span, copy=copy)
                    
                    new_spans[-1] = span
            else:
                new_spans.append(span)
                    
            
            last_span = span
        
        return new_spans
    
    
    def repair_single_spans(self, spans, copy=False):
        if copy:
            spans = spans.copy()
            
        for i in range(len(spans)-1, -1, -1):
            span = spans[i]
            
            if not self.is_span_ok(span):
                # print(i)
                spans.pop(i)
        
        return spans


def get_targets_from_spans_v2(spans, preds, zero_margin=0.20):
    new_spans = []
    for span in spans:
        scores = preds[span["start"]:span["end"]].mean(0)
        score_zero = scores[0] + zero_margin

        class_id = scores.argmax()
        score = scores[class_id]
        if score > score_zero:
            # span = span.copy()
            span["score"] = score
            class_id = to_1_7(class_id)
            span["class"] = cfg.ID2Discourse[class_id] if class_id > 0 else ""
            span["class_id"] = class_id

            new_spans.append(span)
                
    return new_spans

def get_thresh_from_sub(sub, q=None, q_consecutive=None,):

    if q_consecutive is None:
        q_consecutive = 1.5*q

    span_min_score = sub.groupby("class")["score"].quantile(q).round(2).to_dict()
    span_min_len = sub.groupby("class")["num_tokens"].quantile(q).astype(int).to_dict()

    # sub = sub[sub["class"].isin(["Claim", "Evidence"])]
    # sub = sub[sub.duplicated(["id", "class_id"], keep=False)]

    consecutive_span_min_score = sub.groupby("class")["score"].quantile(q_consecutive, interpolation="nearest").round(2).to_dict()
    consecutive_span_min_len = sub.groupby("class")["num_tokens"].quantile(q_consecutive, interpolation="nearest").astype(int).to_dict()

    threshs = {
        "span_min_score": span_min_score,
        "span_min_len": span_min_len,
        "consecutive_span_min_score": consecutive_span_min_score,
        "consecutive_span_min_len": consecutive_span_min_len,
    }


    # For non constrained classes, we put threshs at +Inf ==> Should make sure that this is always Ok !!!
    for thresh_name in threshs:
        for class_ in cfg.Discourse2ID:
            threshs[thresh_name][class_] = threshs[thresh_name].get(class_, float("+Inf"))

            if (thresh_name in ["consecutive_span_min_score", "consecutive_span_min_len"] ) and (class_ not in ['Evidence', 'Claim']):
                threshs[thresh_name][class_] = float("+Inf")

    return threshs

def get_preds_seg_v2(uuids, res, res_seg, span_getter=None,
    target_getter=None, span_getter_kwargs=None, target_getter_kwargs=None):
    span_getter = get_spans_from_seg_v3 if span_getter is None else span_getter
    target_getter = get_targets_from_spans_v2 if target_getter is None else target_getter

    span_getter_kwargs = span_getter_kwargs or {}
    target_getter_kwargs = target_getter_kwargs or {}

    all_spans = {}

    # We're allowed to convert Starts and Ends to In
    res = res.copy()
    res.columns = [to_1_7(col) for col in res.columns]
    res = res.groupby(level=0, axis=1).sum()

    for index, df_uuid  in res.groupby(level=0):

        seg_target = res_seg.loc[index].values
        spans = span_getter(seg_target, pred_target=None, **span_getter_kwargs)
        
        spans = target_getter(spans, preds=df_uuid.values, **target_getter_kwargs)

        id_ = uuids[df_uuid.index[0][0]]
        for i in range(len(spans)-1, -1, -1): # In reverse order to avoid popping issues
            span = spans[i]
            if span["class_id"] <= 0:
                spans.pop(i)
            else:
                span["id"] = id_

        all_spans[id_] = spans

    return all_spans


def get_preds_seg_experimental(uuids, res, res_seg, span_getter=None,
    target_getter=None, span_getter_kwargs=None, target_getter_kwargs=None):
    span_getter = get_spans_from_seg_v3 if span_getter is None else span_getter
    target_getter = get_targets_from_spans_v2 if target_getter is None else target_getter

    span_getter_kwargs = span_getter_kwargs or {}
    target_getter_kwargs = target_getter_kwargs or {}

    all_spans = {}

    res_v0 = res.copy()

    # We're allowed to convert Starts and Ends to In
    res = res.copy()
    res.columns = [to_1_7(col) for col in res.columns]
    # res = res.groupby(level=0, axis=1).max()
    res = res.groupby(level=0, axis=1).sum()

    for index, df_uuid  in res.groupby(level=0):
        seg_target = res_seg.loc[index].values

        spans = span_getter(seg_target, pred_target=res_v0.loc[index].values, **span_getter_kwargs)
        
        spans = target_getter(spans, preds=df_uuid.values, **target_getter_kwargs)

        id_ = uuids[df_uuid.index[0][0]]
        for i in range(len(spans)-1, -1, -1): # In reverse order to avoid popping issues
            span = spans[i]
            if span["class_id"] <= 0:
                spans.pop(i)
            else:
                span["id"] = id_

        all_spans[id_] = spans

    return all_spans


def prune_spans(spans, threshs, max_iter=None):
    repairer = SpanRepairer(**threshs, max_iter=max_iter)
    return repairer(spans)


def add_predictionstring_to_sub(sub):
    sub["predictionstring"] = sub[["start", "end"]].apply(
        lambda span: " ".join(map(str, range(span["start"], span["end"]))), axis=1,
    )

    return sub


def make_sub_from_res(uuids, res, res_seg, span_getter=None, threshs=None, q=0.015, max_iter=None, prune=True):

    spans = get_preds_seg_v2(uuids=uuids, res=res, res_seg=res_seg, span_getter=span_getter)

    sub = pd.DataFrame(list(it_chain(*spans.values())))

    if not len(sub):
        return sub

    if not prune:
        sub = add_predictionstring_to_sub(sub)
        return sub

    if threshs is None:
        threshs = get_thresh_from_sub(sub, q=q)

    new_spans = {}
    for uuid, spans_ in spans.items():
        new_spans[uuid] = prune_spans(spans=spans_, threshs=threshs, max_iter=max_iter)

    new_spans = list(it_chain(*new_spans.values()))

    if not len(new_spans):
        warn("All spans were filtered out, so will remove span filtering and keep initial spans.")

        new_spans = list(it_chain(*spans.values()))
        
    sub = pd.DataFrame(new_spans)

    if len(sub):
        sub = add_predictionstring_to_sub(sub)

    return sub