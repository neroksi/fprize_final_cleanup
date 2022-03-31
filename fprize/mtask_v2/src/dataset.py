import pandas as pd, numpy as np
from pathlib import Path
import re, bisect

from sklearn.model_selection import GroupKFold

from torch.utils.data import Dataset as TorchDataset
from warnings import warn
from string import punctuation
from collections import Counter
import unicodedata
from itertools import chain as it_chain

puncs = re.escape(punctuation)
_split_regex = r"(\s+)"


from . import configs as cfg


def get_append_bar_chars(tokenizer):
    append_bar = set()
    for c in " \t\n\r\f\v":
        if not len(tokenizer(c, add_special_tokens=False)["input_ids"]):
            append_bar.add(c)
    return append_bar

def read_from_id(text_id, root=None):
    root = Path(root or cfg.TRAIN_ROOT)
    return unicodedata.normalize("NFKD", (root / text_id).with_suffix(".txt").read_text(encoding="utf-8") )


def read_train_df(train_csv_path=None):
    train_csv_path =  train_csv_path or cfg.TRAIN_CSV_PATH

    df = pd.read_csv(train_csv_path, encoding="utf-8")

    df["discourse_start"] = df["discourse_start"].astype(int)
    df["discourse_end"] = df["discourse_end"].astype(int)

    df["predictionstring"] = df["predictionstring"].apply(lambda x: np.fromstring(x, sep=" ", dtype=np.int16))

    df["discourse_type_id"] = df["discourse_type"].map(cfg.Discourse2ID)

    return df

def add_training_fold(df, nfolds=5):

    df["fold"] = -1

    gkf = GroupKFold(n_splits=nfolds)

    for fold, (_, val_set) in enumerate(gkf.split(np.arange(len(df)), groups=df["bucket"])):
        df.loc[df.index[val_set], "fold"] = fold
    
    return df


def prediction_string_is_contiguous(df):
    
    return ( df["predictionstring"].apply(
        lambda x: np.sum(x[1:] - x[:-1]- 1 )  if len(x) > 1 else 0) > 0).all()


def split(s):
    res = re.split(_split_regex, s)
    new_res = []
    t = ""
    
    for r in res:
        if re.match(_split_regex, r) and t:
            new_res.append(t)
            t = r
        else:
            t += r

    if not re.match(_split_regex, r):
        if t:
            new_res.append(t)
    else:
        new_res[-1] += r

    res = [r for r in res if len(r)]
    return new_res

def isupper(s):
    return re.search("[A-Z]", s)

def char_span_to_word_span(char_span, token_lens, tokens=None):
   
    start, end = char_span
    
    assert start >= 0
    assert start < end
    
    n = len(token_lens)
    
    
    word_start_1 = bisect.bisect_left(token_lens, start)
    word_start_2 = bisect.bisect_right(token_lens, start)

    word_start = None


    if word_start is None:
        e1 = abs(start - (token_lens[word_start_1-1] if word_start_1 > 0 else 0))
        e2 = abs(start - (token_lens[word_start_2-1] if word_start_2 > 0 else 0))
        
        word_start = word_start_1 if e1 < e2 else word_start_2
        
    
    word_end_1 = bisect.bisect_left(token_lens, end)
    word_end_2 = bisect.bisect_right(token_lens, end)
    
    e1 = abs(end - (token_lens[word_end_1] if word_end_1 < n else token_lens[-1]))
    e2 = abs(end - (token_lens[word_end_2] if word_end_2 < n else token_lens[-1]))
    
    word_end = word_end_1 if e1 < e2 else word_end_2
    word_end = min(word_end+1, n)
    
    word_span = (word_start, word_end)

    return word_span


def split_is_ok(text):
    text = text.strip()
    text2 = split(text)

    if  text != "".join(text2):
        return False

    for i, (t1, t2) in enumerate(zip(text.split(), text2)):
        if t1 != t2.strip():
            return False
    
    return True


def get_word_ids_from_tokens(tokens):
    word_ids = []
    i = 0
    for t in tokens:
        if re.search("\s", t):
            i += 1
        word_ids.append(i)

    word_ids = np.array(word_ids, dtype=np.int16)
    return word_ids


def get_most_common(L):
    return Counter(L).most_common(1)[0][0]

def group_tokens(group_ids, tokens=None, target=None, target_agg_func=None):

    target_agg_func = get_most_common if target_agg_func is None else target_agg_func

    assert (tokens is not None) or (target is not None)

    if len(group_ids) < 2:
        return group_ids, tokens, target
    
    new_group_ids = []
    new_tokens = None if tokens is None else []
    new_target = None if target is None else []

    start = 0
    i = 0
    while i < len(group_ids):
        if group_ids[i] != group_ids[start]:
            end = i
            new_group_ids.append(group_ids[start])
            if  tokens is not None:
                new_tokens.append("".join(tokens[start:end]))
            if target is not None:
                # new_target.append(get_most_common(target[start:end]))
                new_target.append(target_agg_func(target[start:end]))
                
            start = end
        
        i += 1
    
    if start < i:
        end = i
        new_group_ids.append(group_ids[start])
        if  tokens is not None:
            new_tokens.append("".join(tokens[start:end]))
        if target is not None:
            # new_target.append(get_most_common(target[start:end]))
            new_target.append(target_agg_func(target[start:end]))

    new_group_ids = np.array(new_group_ids, dtype=np.int16)
    if target is not None:
        new_target = np.array(new_target, dtype=np.int16)
    return new_group_ids, new_tokens, new_target


def word_perc_pos_from_ids(word_ids):
    word_perc_pos = (word_ids[1:] != word_ids[:-1]).cumsum()
    word_perc_pos = np.insert(word_perc_pos, 0, 0)
    word_perc_pos = cfg.MAXLEN * word_perc_pos / (1+word_perc_pos.max())
    word_perc_pos = np.minimum(word_perc_pos, cfg.MAXLEN - 1)
    word_perc_pos = word_perc_pos.round().astype(np.int64)
    return word_perc_pos


def gen_data_from_id(uuid, tokenizer, append_bar, df=None, root=None):
    """
    No  word grouping is done!
    """

    text = read_from_id(uuid, root=root).strip()
    nchars = len(text)
    tokens = split(text)

    word_ids = get_word_ids_from_tokens(tokens)

    offsets = np.cumsum([len(t) for t in tokens])

    ntokens = len(tokens)

    compute_target = df is not None

    target = None
    if compute_target:
        n_discourse_types = len(cfg.Discourse2ID)

        data = df.loc[df["id"] == uuid]

        data = data[["discourse_type_id", "discourse_start", "discourse_end"]].values

        target = np.zeros(ntokens, dtype=np.int16) # Here, target is in Word dim
        for irow, row in enumerate(data):

            class_id, start, end = row

            start, end = min(start, nchars), min(end, nchars)
            if start >= end:
                warn(f"The {irow}'th span for <{uuid}> is empty")
                continue

            start, end = char_span_to_word_span((start, end), token_lens=offsets, tokens=tokens)

            if start < end:
                target[start] = class_id + n_discourse_types # Beginning <B>

                if start < end-1:
                    target[start+1:end] = class_id # In <I>

    if len(append_bar):
        tokens = [f"|" if t in append_bar else t for t in tokens]

    input_ids = tokenizer(tokens, add_special_tokens=False)["input_ids"] # Word dim

    token_sizes = [len(t) for t  in input_ids] # Word dim
    
    token_sizes.insert(0, 0) # Word dim + 1
    token_sizes = np.array(token_sizes, dtype=np.int16).cumsum() # Word dim + 1

    word_ids = word_ids.repeat(token_sizes[1:] - token_sizes[:-1])

    word_perc_pos = word_perc_pos_from_ids(word_ids)

    input_ids = np.concatenate(input_ids) # In Token dim

    if compute_target:
        target = target.repeat(token_sizes[1:] - token_sizes[:-1]) # Repeat for words tokenized into several tokens => Token dim

        assert target.shape == input_ids.shape
    
        return word_ids, word_perc_pos, token_sizes, input_ids, target
    else:
        return word_ids, word_perc_pos, token_sizes, input_ids



class Dataset(TorchDataset):
    def __init__(self, uuids, data, pad_token_id, mask_token_id, maxlen=None, is_train=True, p_mask_size=None, 
        p_mask_freq=None, special_token_ids=None):
        self.uuids = uuids
        self.data = data
        self.maxlen = maxlen or cfg.MAXLEN
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        self.is_train = is_train
        self.p_mask_freq = cfg.P_MASK_FREQ if p_mask_freq is None else p_mask_freq
        self.special_token_ids = special_token_ids
        
    def __len__(self):
        return len(self.uuids)

    def mask_input_ids(self, input_ids):
        
        p_mask_size = np.random.uniform(cfg.P_MASK_SIZE_LOW, cfg.P_MASK_SIZE_HIGH)
        
        n_masked = int(p_mask_size * len(input_ids))
        
        if n_masked > 0:
            index = np.random.choice(len(input_ids), size=n_masked, replace=False)
            input_ids[index] = self.mask_token_id

        return input_ids

    @property
    def get_special_tokens_pad_widths(self):
        pad_width, constant_values = [0, 0], [-666, -666]
        if self.special_token_ids["bos_token_id"] is not None:
            pad_width[0] = 1
            constant_values[0] = self.special_token_ids["bos_token_id"]
        
        if self.special_token_ids["eos_token_id"] is not None:
            pad_width[1] = 1
            constant_values[1] = self.special_token_ids["eos_token_id"]

        return pad_width, constant_values

    def add_special_tokens(self, input_ids, target):
       
        pad_width, constant_values = self.get_special_tokens_pad_widths

        if sum(pad_width):
            input_ids = np.pad(input_ids, pad_width=pad_width,constant_values=constant_values)

            target = np.pad(target, pad_width=pad_width, constant_values=cfg.PYTORCH_CE_IGNORE_INDEX)

        return input_ids, target

    
    def truncate(self, word_perc_pos, input_ids, target):
        input_ids = input_ids.astype(np.int64, copy=False)
        target = target.astype(np.int64, copy=False)

        seq_len = self.maxlen-2
        if self.is_train and np.random.random() < cfg.FORCE_TRUNC_FREQ:
            seq_len = min(seq_len, cfg.MIN_SEQ_LEN)

        if len(input_ids) > seq_len:
            if self.is_train:
                # start = 0

                if np.random.random() < cfg.P_RANDOM_START:
                    start = np.random.choice(len(input_ids) - seq_len)
                elif np.random.random() < cfg.P_START_AT_SEQ_BEGINNING:
                    start = 0
                else:
                    start = len(input_ids) - seq_len
            else:
                start = 0

            input_ids = input_ids[start: start + seq_len ]
            target = target[start: start + seq_len]
            word_perc_pos = word_perc_pos[start: start + seq_len + 2]
        
        if self.is_train and np.random.rand() < self.p_mask_freq:
            input_ids = self.mask_input_ids(input_ids.copy())

        
        input_ids, target = self.add_special_tokens(input_ids=input_ids, target=target)

        if len(word_perc_pos) < len(input_ids):
            word_perc_pos = np.concatenate([word_perc_pos, [word_perc_pos[-1]]*(len(input_ids) - len(word_perc_pos))  ])
       
        masks = np.ones_like(input_ids)

        return word_perc_pos, input_ids, masks, target
        
    
    def __getitem__(self, idx):
        uuid = self.uuids[idx]
        
        word_ids, word_perc_pos, token_sizes, input_ids, target = self.data[uuid]

        return self.truncate(word_perc_pos=word_perc_pos, input_ids=input_ids, target=target)




class TestDataset(Dataset):
    def __init__(self, uuids, pad_token_id, mask_token_id=None, maxlen=None, stride=None, is_train=False,
        special_token_ids=None, data=None, **kwargs):
        self.uuids = uuids
        # self.data = data
        self.maxlen = maxlen or cfg.MAXLEN
        self.stride = stride or (self.maxlen // cfg.STRIDE_MAX_LEN_RATIO if self.maxlen < 1024 else 1024)
        # print("self.stride:", self.stride)
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        self.is_train = is_train
        self.special_token_ids = special_token_ids
        self.data = data
        self.kwargs = kwargs

        self.p_mask_freq = cfg.P_MASK_FREQ
        
    def __len__(self):
        return len(self.uuids)


    def add_special_tokens(self, word_ids, input_ids, target=None):

        pad_width, constant_values = self.get_special_tokens_pad_widths
        
        if sum(pad_width) :
            word_ids = np.pad(word_ids, pad_width=pad_width, constant_values=-1)

            input_ids = np.pad(input_ids, pad_width=pad_width, constant_values=constant_values)

            if target is not None:
                target = np.pad(target, pad_width=pad_width, constant_values=cfg.PYTORCH_CE_IGNORE_INDEX)

        return (word_ids, input_ids, target) if target is not None else (word_ids, input_ids)
    
    
    def truncate(self, word_ids, word_perc_pos, input_ids, target=None):

        if target is not None:
            target = target.astype(np.int64, copy=False)

        input_ids_list = []
        masks_list = []
        token_ids_list = []
        word_perc_pos_list = []

        target_list = []

        if self.is_train and np.random.rand() < self.p_mask_freq:
            input_ids = self.mask_input_ids(input_ids.copy())


        for start in range(0, max(len(input_ids) - self.maxlen + 2 , 0) + self.stride, self.stride):
            end = min(start + self.maxlen-2, len(input_ids))
            start = max(end - self.maxlen + 2, 0)

            word_id, input_id = word_ids[start:end], input_ids[start:end]
            if target is not None:
                t = target[start:end]
                word_id, input_id, t = self.add_special_tokens(word_ids=word_id, input_ids=input_id, target=t)
                target_list.append(t)
            else:
                word_id, input_id = self.add_special_tokens(word_ids=word_id, input_ids=input_id)

            w = word_perc_pos[start:end+2]
            if len(w) < len(input_id):
                w = np.concatenate([ w, [w[-1]]*(len(input_id)- len(w)) ])

            word_perc_pos_list.append(w)
            input_ids_list.append(input_id)
            token_ids_list.append(word_id)
            masks_list.append(np.ones(len(input_id)))

        input_ids = input_ids_list
        masks = masks_list
        word_ids = token_ids_list
        word_perc_pos = word_perc_pos_list
        if target is not None:
            target = target_list

        return (word_ids, word_perc_pos, input_ids, masks)  if  target is None else (
                    word_ids, word_perc_pos, input_ids, masks, target)
    
    def __getitem__(self, idx):
        uuid = self.uuids[idx]
        
        res = gen_data_from_id(uuid, **self.kwargs) if self.data is None else self.data[uuid]

        if len(res) == 4:
            word_ids, word_perc_pos, token_sizes, input_ids = res
            target = None
        else:
            word_ids, word_perc_pos, token_sizes, input_ids, target = res # Word_ids are ignored here, but not in inference

        return (idx, token_sizes, *self.truncate(word_ids=word_ids, word_perc_pos=word_perc_pos, input_ids=input_ids, target=target))


class DynamicBatchDataset(TorchDataset):
    def __init__(self, dataset: TestDataset, batch_size: int, sizes: list):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.sizes  = sizes
        self.spans = self.get_batch_spans()

    def get_batch_spans(self):
        sizes = self.sizes
        batch_size = self.batch_size

        if len(sizes) < 2:
            return [(0, len(sizes))]
        
        spans = []
        s = 0
        i = 0
        start = 0
        while (i < len(sizes)):
            s += sizes[i]
            
            if s > batch_size:
                end = max(start+1, i)
                spans.append((start, end))
                i = end
                start = i
                s = 0
            else:
                i += 1
                
        if not spans or spans[-1][1] < len(sizes):
            spans.append((start, len(sizes)))
        return spans

    def __len__(self):
        return len(self.spans)

    def __getitem__(self, idx):
        span = self.spans[idx]
        samples = []
        for idx in range(span[0], span[1]):
            samples.append(self.dataset[idx])
            
        return samples


def pad_to_bach_maxlen(batch_ids, token_ids, word_perc_pos, input_ids, masks, pad_token_id, target=None, max_item=None):
    
    batch_item_sizes = [len(x) if max_item is None else min(len(x), max_item) for x  in token_ids]

    batch_size = sum(batch_item_sizes)
    batch_maxlen = max([len(xx) for x in token_ids for xx in x])

    batch_ids = np.array(batch_ids).repeat(batch_item_sizes)

    shape = (batch_size, batch_maxlen)
    new_token_ids = np.full(shape, -1, dtype=np.int16)
    new_word_perc_pos = np.zeros(shape, dtype=np.int64)
    new_input_ids = np.full(shape, pad_token_id, dtype=np.int64)
    new_masks = np.zeros(shape, dtype=np.int64)

    compute_target =  target is not None
    if compute_target:
        new_target = np.full(shape, cfg.PYTORCH_CE_IGNORE_INDEX, dtype=np.int64)
    else:
        target = [[[]]*size for size in batch_item_sizes]

    pos = 0
    for token_id_list, word_perc_list, input_id_list, mask_list, target_list in zip(token_ids, word_perc_pos, input_ids, masks, target):
        for i, (token_id, word_perc, input_id, mask, t) in enumerate(zip(token_id_list, word_perc_list, input_id_list, mask_list, target_list)):

            if max_item is not None and i >= max_item:
                break

            new_token_ids[pos, :len(token_id)] = token_id
            new_word_perc_pos[pos, :len(word_perc)] = word_perc
            new_input_ids[pos, :len(input_id)] = input_id
            new_masks[pos, :len(mask)] = mask

            if compute_target:
                new_target[pos, :len(t)] = t

            pos += 1

    return (batch_ids, new_token_ids, new_word_perc_pos, new_input_ids, new_masks, *( (new_target, ) if compute_target else ( ) ) )


def collate_fn(inputs, pad_token_id, max_item=None):

    inputs = tuple(zip(*inputs))

    assert len(inputs) in [6, 7], f"{len(inputs)} is an Unknown number of elements"

    batch_ids, token_pos, token_ids, word_perc_pos, input_ids, masks = inputs[:6]

    target = inputs[6] if len(inputs) == 7 else None
    
    return pad_to_bach_maxlen(batch_ids=batch_ids, token_ids=token_ids, word_perc_pos=word_perc_pos, input_ids=input_ids,
                masks=masks, target=target, pad_token_id=pad_token_id, max_item=max_item)



def collate_fn_train(inputs, pad_token_id):

    inputs = tuple(zip(*inputs))

    assert len(inputs) == 4 , f"{len(inputs)} is an Unknown number of elements"

    word_perc_pos, input_ids, masks, target = inputs

    batch_size = len(input_ids)
    batch_maxlen = max([len(x) for x in input_ids])

    shape = (batch_size, batch_maxlen)

    new_word_perc_pos = np.full(shape, -1, dtype=np.int64)
    new_input_ids = np.full(shape, pad_token_id, dtype=np.int64)
    new_masks = np.zeros(shape, dtype=np.int64)
    new_target = np.full(shape, cfg.PYTORCH_CE_IGNORE_INDEX, dtype=np.int64)

    for pos, (word_id, input_id, mask, t ) in enumerate(zip(word_perc_pos, input_ids, masks, target)):
        new_word_perc_pos[:, :len(word_id)] = word_id
        new_input_ids[pos, :len(input_id)] = input_id
        new_masks[pos, :len(mask)] = mask
        new_target[pos, :len(t)] = t

    return new_word_perc_pos, new_input_ids, new_masks, new_target


def collate_fn_list(inputs, pad_token_id, max_item=None):
    return collate_fn(it_chain(*inputs), pad_token_id=pad_token_id, max_item=max_item)


class RunningMax:
    """
    Will track the max for each position over running windows.
    """
    def __init__(self):
        self.vals = None
        self._ids = []
        self._vals = []
        
    def append(self, ids, vals, reduce=False):
        self._ids.append(ids)
        self._vals.append(vals)
        return self
    
    def reset(self):
        self._ids = []
        self._vals = []
        return self
    
    def reduce(self):
        ids = np.concatenate(self._ids)
        vals = np.concatenate(self._vals)
        
        vals = pd.DataFrame(vals, index=tuple(zip(*ids.T)))
        
        if self.vals is not None:
            vals = pd.concat([self.vals, vals], axis=0, sort=False)
        
        self.vals = vals.groupby(level=(0, 1)).max()
        
        self.reset()

        return self

    def update(self, ids, vals, reduce=False):
        self.append(ids, vals)
        if reduce:
            self.reduce()
        return self

    def normlaize(self):
        self.vals /= self.vals.sum(1).values[:, None]
        return self