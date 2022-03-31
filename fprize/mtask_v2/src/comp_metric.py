import pandas as pd
import numpy as np

def calc_overlap(set_pred, set_gt):
    """
    Calculates the overlap between prediction and
    ground truth and overlap percentages used for determining
    true positives.
    """
    # Length of each and intersection
    try:
        len_gt = len(set_gt)
        len_pred = len(set_pred)
        inter = len(set_gt & set_pred)
        overlap_1 = inter / len_gt
        overlap_2 = inter/ len_pred
        return (overlap_1, overlap_2)
    except:  # at least one of the input is NaN
        return (0, 0)

def score_feedback_comp_micro(pred_df, gt_df, discourse_type):
    """
    A function that scores for the kaggle
        Student Writing Competition
        
    Uses the steps in the evaluation page here:
        https://www.kaggle.com/c/feedback-prize-2021/overview/evaluation
    """
    gt_df = gt_df.loc[gt_df['discourse_type'] == discourse_type, 
                      ['id', 'predictionstring']].reset_index(drop=True)
    pred_df = pred_df.loc[pred_df['class'] == discourse_type,
                      ['id', 'predictionstring']].reset_index(drop=True)
    pred_df['pred_id'] = pred_df.index
    gt_df['gt_id'] = gt_df.index
    pred_df['predictionstring'] = [set(pred.split(' ')) for pred in pred_df['predictionstring']]
    gt_df['predictionstring'] = [set(pred.split(' ')) for pred in gt_df['predictionstring']]
    
    # Step 1. all ground truths and predictions for a given class are compared.
    joined = pred_df.merge(gt_df,
                           left_on='id',
                           right_on='id',
                           how='outer',
                           suffixes=('_pred','_gt')
                          )
    overlaps = [calc_overlap(*args) for args in zip(joined.predictionstring_pred, 
                                                     joined.predictionstring_gt)]
    
    # 2. If the overlap between the ground truth and prediction is >= 0.5, 
    # and the overlap between the prediction and the ground truth >= 0.5,
    # the prediction is a match and considered a true positive.
    # If multiple matches exist, the match with the highest pair of overlaps is taken.
    joined['potential_TP'] = [(overlap[0] >= 0.5 and overlap[1] >= 0.5) \
                              for overlap in overlaps]
    joined['max_overlap'] = [max(*overlap) for overlap in overlaps]
    joined_tp = joined.query('potential_TP').reset_index(drop=True)
    tp_pred_ids = joined_tp\
        .sort_values('max_overlap', ascending=False) \
        .groupby(['id','gt_id'])['pred_id'].first()

    # 3. Any unmatched ground truths are false negatives
    # and any unmatched predictions are false positives.
    fp_pred_ids = set(joined['pred_id'].unique()) - set(tp_pred_ids)

    matched_gt_ids = joined_tp['gt_id'].unique()
    unmatched_gt_ids = set(joined['gt_id'].unique()) -  set(matched_gt_ids)

    # Get numbers of each type
    TP = len(tp_pred_ids)
    FP = len(fp_pred_ids)
    FN = len(unmatched_gt_ids)
    #calc microf1
    my_f1_score = TP / (TP + 0.5*(FP+FN))
    return my_f1_score

def score_feedback_comp(pred_df, gt_df, return_class_scores=False):
    if isinstance(gt_df["predictionstring"].iloc[0], np.ndarray):
        gt_df["predictionstring"] = gt_df["predictionstring"].apply(lambda x: " ".join(x.astype(str)))
        
    class_scores = {}
    for discourse_type in gt_df.discourse_type.unique():
        class_score = score_feedback_comp_micro(pred_df, gt_df, discourse_type)
        class_scores[discourse_type] = class_score
    f1 = np.mean([v for v in class_scores.values()])
    if return_class_scores:
        return f1, class_scores
    return f1



def f1_from_joined(joined):
    # Step 1. all ground truths and predictions for a given class are compared.
    
    overlaps = [calc_overlap(*args) for args in zip(joined.predictionstring_pred, 
                                                     joined.predictionstring_gt)]
    
    # 2. If the overlap between the ground truth and prediction is >= 0.5, 
    # and the overlap between the prediction and the ground truth >= 0.5,
    # the prediction is a match and considered a true positive.
    # If multiple matches exist, the match with the highest pair of overlaps is taken.
    joined['potential_TP'] = [(overlap[0] >= 0.5 and overlap[1] >= 0.5) \
                              for overlap in overlaps]
    joined['max_overlap'] = [max(*overlap) for overlap in overlaps]
    joined_tp = joined.query('potential_TP').reset_index(drop=True)
    tp_pred_ids = joined_tp\
        .sort_values('max_overlap', ascending=False) \
        .groupby(['id','gt_id'])['pred_id'].first()

    # 3. Any unmatched ground truths are false negatives
    # and any unmatched predictions are false positives.
    fp_pred_ids = set(joined['pred_id'].unique()) - set(tp_pred_ids)

    matched_gt_ids = joined_tp['gt_id'].unique()
    unmatched_gt_ids = set(joined['gt_id'].unique()) -  set(matched_gt_ids)

    # Get numbers of each type
    TP = len(tp_pred_ids)
    FP = len(fp_pred_ids)
    FN = len(unmatched_gt_ids)
    #calc microf1
    my_f1_score = TP / (TP + 0.5*(FP+FN))
    return my_f1_score