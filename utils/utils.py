import tensorflow as tf
from tensorflow.keras import backend as K
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,average_precision_score
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
def dice_loss(smooth=1.0):
 
    def dice_loss_fixed(y_true, y_pred):
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])
        
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        denominator = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
        
        return 1.0 - (2.0 * intersection + smooth) / (denominator + smooth)

    return dice_loss_fixed
    
def dice_loss_fixed(y_true, y_pred):
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])
        
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        denominator = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
        
        return 1.0 - (2.0 * intersection + 1) / (denominator + 1)
    
def combined_loss(y_true, y_pred):
    return tf.keras.losses.BinaryCrossentropy()(y_true, y_pred) + dice_loss_fixed(y_true, y_pred)

def iou_loss(smooth=1e-6):
    def loss(y_true, y_pred):
        # Flatten the tensors
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])
        
        # Intersection and union
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
        
        # Soft IoU
        iou = (intersection + smooth) / (union + smooth)
        return 1.0 - iou
    return loss

class PrecisionScore(tf.keras.metrics.Metric):
    def __init__(self, name='precision_score', **kwargs):
        super(PrecisionScore, self).__init__(name=name, **kwargs)
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_bin = tf.cast(y_pred > 0.5, tf.int32)
        y_true_int = tf.cast(y_true, tf.int32)

        def compute_precision(y_true_np, y_pred_np):
            y_true_np = y_true_np.numpy().flatten()
            y_pred_np = y_pred_np.numpy().flatten()
            return np.array(precision_score(y_true_np, y_pred_np, zero_division=1), dtype=np.float32)

        prec = tf.py_function(compute_precision, [y_true_int, y_pred_bin], tf.float32)
        prec = tf.reshape(prec, [])
        self.total.assign_add(prec)
        self.count.assign_add(1.0)

    def result(self):
        return self.total / (self.count + tf.keras.backend.epsilon())

    def reset_state(self):
        self.total.assign(0.0)
        self.count.assign(0.0)


class RecallScore(tf.keras.metrics.Metric):
    def __init__(self, name='recall_score', **kwargs):
        super(RecallScore, self).__init__(name=name, **kwargs)
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_bin = tf.cast(y_pred > 0.5, tf.int32)
        y_true_int = tf.cast(y_true, tf.int32)
        
        def compute_recall(y_true_np, y_pred_np):
            y_true_np = y_true_np.numpy().flatten()
            y_pred_np = y_pred_np.numpy().flatten()
            rec = recall_score(y_true_np, y_pred_np, zero_division=1)
            return np.array(rec, dtype=np.float32)
            
        rec_val = tf.py_function(compute_recall, [y_true_int, y_pred_bin], tf.float32)
        rec_val = tf.reshape(rec_val, [])
        self.total.assign_add(rec_val)
        self.count.assign_add(1.0)
        
    def result(self):
        return self.total / (self.count + tf.keras.backend.epsilon())
        
    def reset_state(self):
        self.total.assign(0.0)
        self.count.assign(0.0)

class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_bin = tf.cast(y_pred > 0.5, tf.int32)
        y_true_int = tf.cast(y_true, tf.int32)
        
        def compute_f1(y_true_np, y_pred_np):
            y_true_np = y_true_np.numpy().flatten()
            y_pred_np = y_pred_np.numpy().flatten()
            f1_val = f1_score(y_true_np, y_pred_np, zero_division=1)
            return np.array(f1_val, dtype=np.float32)
            
        f1_val = tf.py_function(compute_f1, [y_true_int, y_pred_bin], tf.float32)
        f1_val = tf.reshape(f1_val, [])
        self.total.assign_add(f1_val)
        self.count.assign_add(1.0)
        
    def result(self):
        return self.total / (self.count + tf.keras.backend.epsilon())
        
    def reset_state(self):
        self.total.assign(0.0)
        self.count.assign(0.0)


def specificity(y_true, y_pred):
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    y_pred = K.cast(y_pred > 0.5, dtype=tf.float32)

    tn = K.sum((1 - y_true) * (1 - y_pred))
    fp = K.sum((1 - y_true) * y_pred)
    specificity = tn / (tn + fp + K.epsilon())
    return specificity

class AdaptiveEarlyStopping(tf.keras.callbacks.Callback):
    def __init__(self, monitor='val_IoU', min_delta=0.0001, patience=10, freeze_patience=4):
        super(AdaptiveEarlyStopping, self).__init__()
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.freeze_patience = freeze_patience
        self.wait = 0
        self.freeze_wait = 0
        self.best = -np.Inf
        self.previous = -np.Inf 

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current_value = logs.get(self.monitor)
        if current_value is None:
            print(f"Warning: Metric {self.monitor} is not available.")
            return
        if self.previous == -np.Inf:
            self.previous = current_value
            return
        if current_value > self.best + self.min_delta:
            self.best = current_value
            self.wait = 0  
            self.freeze_wait = 0
        else:
            self.wait += 1
            if abs(current_value - self.previous) <= self.min_delta:
                self.freeze_wait += 1
                if self.freeze_wait == 1:  
                    print(f"Validation IoU change is below {self.min_delta}, reducing patience to {self.freeze_patience}.")
                print(f"{min(self.patience - self.wait,self.freeze_patience - self.freeze_wait)} more epochs till EarlyStop!")
            else:
                self.freeze_wait = 0 
        if self.wait >= self.patience or self.freeze_wait >= self.freeze_patience:
            print(f"Training stopped early")
            self.model.stop_training = True
        self.previous = current_value

def calculate_metrics(ytrue, ypred):
    ypred_bin = (ypred > 0.5).astype(np.float32)
    ytrue = ytrue.numpy()
    if isinstance(ytrue, tf.Tensor):
        ytrue = ytrue.numpy()
    if isinstance(ypred, tf.Tensor):
        ypred = ypred.numpy()
    if isinstance(ypred_bin, tf.Tensor):
        ypred_bin = ypred_bin.numpy()
    ytrue = ytrue.astype(np.uint8)
    ypred_bin = ypred_bin.astype(np.uint8)
    
    ssim_score = ssim(ytrue, ypred_bin, data_range=1)
    
    intersection = np.logical_and(ytrue, ypred_bin).sum()
    
    binary_iou = tf.keras.metrics.BinaryIoU()
    binary_iou.reset_state()
    binary_iou.update_state(ytrue, ypred_bin)
    iou_score = binary_iou.result().numpy()
    
    dice_loss = 1 - (2 * intersection + 1) / (ytrue.sum() + ypred_bin.sum() + 1)
    ytrue_flat, ypred_bin_flat = ytrue.flatten(), ypred_bin.flatten()
    acc = accuracy_score(ytrue_flat, ypred_bin_flat)
    recall = recall_score(ytrue_flat, ypred_bin_flat,zero_division=1)
    prec = precision_score(ytrue_flat, ypred_bin_flat,zero_division=1)
    f1 = (2 * prec * recall) / (prec + recall + 1e-6)
    spec = specificity(ytrue_flat, ypred_bin_flat)
    ypred_flat = ypred.flatten()
    if np.sum(ytrue_flat) == 0:
        if np.sum(ypred_bin_flat) == 0:
            ap_score = 1.0
        else:
            ap_score = 0.0
    else:
        ap_score = average_precision_score(ytrue_flat, ypred_flat)
    return ssim_score, ap_score, iou_score, dice_loss, f1,recall,prec,acc,spec

def eval_model(tmodel,data,name,printres=False):
    ssims,aps,ious,dls,f1s,recs,precs,accs,specs = [],[],[],[],[],[],[],[],[]
    scores = dict()
    for batch_idx, (X_batch, Y_batch) in tqdm(enumerate(data)):
        for xseq,ytrue in zip(X_batch,Y_batch):
            ypred = np.squeeze(tmodel.predict(xseq[None,...],verbose=0))
            ssimv,ap, iou, dice,f1,rec,prec,acc,spec = calculate_metrics(ytrue, ypred)
            ssims.append(ssimv)
            if ap>=0:
                aps.append(ap)
            ious.append(iou)
            dls.append(dice)
            f1s.append(f1)
            recs.append(rec)
            precs.append(prec)
            accs.append(acc)
            specs.append(spec)
    scores['SSIM'] = np.mean(ssims)
    scores['mAP'] = np.mean(aps)
    scores['IoU'] = np.mean(ious)
    scores['Dice Loss'] = np.mean(dls)
    scores['F1-Score'] = np.mean(f1s)
    scores['Recall'] = np.mean(recs)
    scores['Precision'] = np.mean(precs)
    scores['Accuracy'] = np.mean(accs)
    scores['Specificity'] = np.mean(specs)
    if printres:
        print(f"Mean Accuracy for {name} set = {scores['Accuracy']:.4f}")
        print(f"Mean SSIM for {name} set = {scores['SSIM']:.4f}")
        print(f"Mean IoU for {name} set = {scores['IoU']:.4f}")
        print(f"Mean Dice Loss for {name} set = {scores['Dice Loss']:.4f}")
        print(f"Mean Specificity for {name} set = {scores['Specificity']:.4f}")
        print(f"Mean F1-Score for {name} set = {scores['F1-Score']:.4f}")
        print(f"Mean Recall for {name} set = {scores['Recall']:.4f}")
        print(f"Mean Precision for {name} set = {scores['Precision']:.4f}")
        print(f"Mean Average Precision for {name} set = {scores['mAP']:.4f}\n")
    return scores

def plot_history(history,iou):
    metrics = ['loss', 'IoU', 'accuracy','f1_score','recall_score','precision_score','specificity']
    titles = ['Dice Loss', 'IoU', 'Accuracy','F1-Score','Recall Score','Precision Score','Specificity']
    
    fig, axes = plt.subplots(len(metrics), 1, figsize=(6, 4*len(metrics)))
    for i, metric in enumerate(metrics):
        ax = axes[i]
        ax.plot(history.history[metric], label=f'Training {titles[i]}')
        ax.plot(history.history[f'val_{metric}'], label=f'Validation {titles[i]}')
        ax.set_title(titles[i])
        ax.set_xlabel('Epochs')
        ax.set_ylabel(titles[i])
        ax.legend()
        ax.grid(True)
    plt.suptitle(f'Training & Validation Curves for Best ConvLSTM Model - Val_IoU={iou:.3f}', fontsize=12,y=1.0002)
    plt.tight_layout()
    plt.show()