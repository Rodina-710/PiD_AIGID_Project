import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch 
import torch.nn as nn
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay,

    average_precision_score,  
    roc_auc_score,             
    roc_curve,
    precision_recall_curve,
)


from src import stream_data
from src import model as model_manager 
from src import pid as pid_module 

# Device 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model 
model, shard , epoch , optimizer = model_manager.get_model(device)
model.to(device)
model.eval()

print (f"\n[INFO] Loaded model from shard {shard}, epoch {epoch}")

# Prediciton Function 
def predict_batch (images) :
    """
    images: list of PIL images
    """
    processed =[]

    for img in images :
        # Apply PiD once 
        residual = pid_module.apply_pid_algorithm(img)

        tensor = torch.from_numpy(residual).permute(2,0,1).float() /255.0
        processed.append(tensor)

    x = torch.stack(processed).to(device)

    with torch.no_grad() :
        outputs = model(x)
        preds = torch.argmax(outputs, dim =1)

        # softmax gives calibrated probabilities per class
        # we take column 1 = probability the image is FAKE
        # this is what AP and AUC-ROC need — NOT the hard 0/1 label
        probs = torch.softmax(outputs, dim=1)[:, 1]

    return preds.cpu().numpy(), probs.cpu().numpy()   


# Evaluation LOOP

def evaluate() :
    y_trus = []
    y_pred = [] 
    y_prob  = []   # raw fake-probabilities    → used for AP and AUC-ROC

    print("\n[INFO] Starting evaluation on test shards...\n")

    for batch_idx , (images , labels) in enumerate (
        stream_data.get_test_batch(start_shard=1 , batch_size=64)
    ) :
        
        preds , probs = predict_batch(images) 

        y_pred.extend(preds)
        y_prob.extend(probs)   # ← collect probabilities every batch
        y_trus.extend(labels)

        if batch_idx % 10 == 0:
            print(f"[EVAL] Processed {batch_idx * 64} samples...")

        if batch_idx == 200 :
            break 
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

# Metrics

    acc = accuracy_score(y_true , y_pred)
    prec = precision_score (y_true , y_pred)
    rec = recall_score (y_true , y_pred) 
    cm = confusion_matrix(y_true , y_pred)

    print("\n" + "=" * 52)
    print("  EVALUATION RESULTS")
    print("=" * 52)
    print(f"  Accuracy        : {acc  * 100:.2f}%")
    print(f"  Precision       : {prec * 100:.2f}%")
    print(f"  Recall          : {rec  * 100:.2f}%")
    print("-" * 52)
    print(f"  Avg Precision   : {ap:.4f}   ← primary research metric")
    print(f"  AUC-ROC         : {auc:.4f}  ← primary benchmark metric")
    print("=" * 52)
 
    print("\n  Confusion Matrix  (rows=actual, cols=predicted)")
    print("  Labels: 0=Real  1=Fake\n")
    print(cm)
     

    _save_confusion_matrix(y_true, y_pred)
    _save_roc_curve(y_true, y_prob, auc)
    _save_pr_curve(y_true, y_prob, ap)
    print("\n[INFO] Plots saved to images/")
 
    return {
        "accuracy":  acc,
        "precision": prec,
        "recall":    rec,
        "ap":        ap,   
        "auc_roc":   auc,
        "confusion_matrix": cm,
    } 


# Plot Helpers 

def _save_confusion_matrix(y_true, y_pred):
    """
    Visual confusion matrix.
    Blue = correct  |  lighter blue = errors
    Tells you at a glance whether errors are FP or FN.
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Real", "Fake"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("images/confusion_matrix.png", dpi=120)
    plt.close()
 
 
def _save_roc_curve(y_true, y_prob, auc):
    """
    ROC curve: True Positive Rate vs False Positive Rate.
    The closer the curve hugs the top-left corner → the better.
    Diagonal dashed line = random guessing (AUC = 0.5).
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, color="#534AB7", lw=2,
             label=f"AUC-ROC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Random (0.5)")
    plt.xlabel("False Positive Rate  (real images wrongly flagged)")
    plt.ylabel("True Positive Rate  (fakes correctly caught)")
    plt.title("ROC Curve — PiD Detector")
    plt.legend()
    plt.tight_layout()
    plt.savefig("images/roc_curve.png", dpi=120)
    plt.close()
 
 
def _save_pr_curve(y_true, y_prob, ap):
    """
    Precision-Recall curve.
    High area = model stays precise even at high recall.
    AP is the area under this curve — the main reported metric.
    """
    prec_vals, rec_vals, _ = precision_recall_curve(y_true, y_prob)
    plt.figure()
    plt.plot(rec_vals, prec_vals, color="#993C1D", lw=2,
             label=f"AP = {ap:.4f}")
    plt.xlabel("Recall  (fraction of fakes caught)")
    plt.ylabel("Precision  (fraction of fake labels that are correct)")
    plt.title("Precision-Recall Curve — PiD Detector")
    plt.legend()
    plt.tight_layout()
    plt.savefig("images/pr_curve.png", dpi=120)
    plt.close()
 


# RUN 

if __name__ == "__main__":
    evaluate ()