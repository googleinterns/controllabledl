from sklearn.metrics import accuracy_score, roc_curve, auc, roc_auc_score, precision_score, recall_score, precision_recall_curve, accuracy_score

def get_metrics(y_true, y_pred, y_score):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    return acc, prec, recall, fpr, tpr, roc_auc

def get_correct_results(out, label_Y):
    y_pred_tag = torch.round(out)    # Binary label
    return (y_pred_tag == label_Y).sum().float()