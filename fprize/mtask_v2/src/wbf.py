from ensemble_boxes import weighted_boxes_fusion
import numpy as np, pandas as pd
from  tqdm.auto import tqdm
from  warnings import warn
from . import configs as cfg


def to_rectangle(box_seg, norm=None):
    box_rect = np.zeros((len(box_seg), 4), dtype=np.float32)
    box_rect[:, [0, 2]] = box_seg
    
    if norm:
        box_rect /= norm
        
    box_rect[:, -1] = 1
    
    return box_rect


def fusion_boxes_for_uuid(uuid, boxes_list, scores_list, labels_list, weights, iou_thr=0.333,
        skip_box_thr=0.001, default_box_idx=None):
    
    Z = 1e-3 + max(map(np.max, boxes_list))

    if default_box_idx is None:
        default_box_idx = np.argmax(weights)
    
    new_boxes_list = []
    for box in boxes_list:
        new_boxes_list.append(to_rectangle(box, norm=Z))
        
    boxes_list = new_boxes_list

    boxes, scores, labels = weighted_boxes_fusion(
        boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr, )
    
    if not len(boxes):
        boxes, scores, labels = boxes_list[default_box_idx], scores_list[default_box_idx], labels_list[default_box_idx]
        
    boxes = (boxes * Z).round().astype(int)
    labels = labels.astype(int)
    
    sub = pd.DataFrame(
        {
            "id": uuid,
            "class_id": labels,
            "class": [cfg.ID2Discourse[class_] for class_ in labels],
            "score": scores,
            "start": boxes[:, 0],
            "end": boxes[:, 2],
            "predictionstring":  [" ".join(map(str, range(start, end))) for start, end in
                                  zip(boxes[:, 0], boxes[:, 2])],
        }
    )
    
    sub = sub.sort_values(["start", "end"])

    return sub

def get_uuids(subs):
    assert len(subs)

    uuids = set(subs[0]["id"])

    all_uuids = set(uuids)
    common_uuids = set(uuids)

    for sub in subs[1:]:
        uuids = set(sub["id"])

        all_uuids = all_uuids.union(uuids)
        common_uuids = common_uuids.intersection(uuids)
    
    common_uuids = sorted(common_uuids)
    all_uuids = sorted(all_uuids)

    return all_uuids, common_uuids


def fusion_boxes_for_subs(subs, weights=None, bar=True, iou_thr=0.333,
        skip_box_thr=0.001, default_box_idx=None, **kwargs):

    assert len(subs)

    if weights is None:
        weights = np.ones(len(subs))
        weights /= weights.sum()

    all_uuids, common_uuids = get_uuids(subs)

    other_uuids = sorted(set(all_uuids).difference(common_uuids))

    if len(other_uuids):
        warn(f"{len(other_uuids)} ids are not present in all the dataframes, this could be problematic !")

    res = []

    if bar:
        all_uuids = tqdm(all_uuids)

    for uuid in all_uuids:
        boxes_list, scores_list, labels_list = [], [], []

        valid_weights = []
        for w, sub in zip(weights, subs):
            temp = sub.query(f"id == '{uuid}'")

            if not len(temp):
                continue

            boxes_list.append(temp[["start", "end"]].values)
            scores_list.append(temp["score"].values)
            labels_list.append(temp["class_id"].values)
            valid_weights.append(w)

        sub = fusion_boxes_for_uuid(
            uuid=uuid,
            boxes_list=boxes_list,
            scores_list=scores_list,
            labels_list=labels_list,
            weights=valid_weights,
            iou_thr=iou_thr,
            skip_box_thr=skip_box_thr,
            default_box_idx=default_box_idx,
            **kwargs,
        )

        res.append(sub)
        
    sub = pd.concat(res, axis=0, sort=False)
    sub.reset_index(drop=True, inplace=True)

    return sub