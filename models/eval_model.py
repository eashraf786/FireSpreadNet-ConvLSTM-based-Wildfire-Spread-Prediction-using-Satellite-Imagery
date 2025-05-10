from utils import eval_model, plot_history, calculate_metrics
from DataLoader import WildfireDataset
from utils import dice_loss_fixed,F1Score,PrecisionScore,RecallScore,specificity
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pickle
from tqdm import tqdm

with open('/workspace/eashraf/Dissertation2/Other files/traininghistory_full.pkl','rb') as f:
    history = pickle.load(f)

val_iou = max(history.history['val_iou'])
plot_history(history,val_iou)

tmodel = tf.keras.models.load_model("./models/convlstm_model_AllFE_epoch22_valIoU_0.69197_D24.keras",
                                   custom_objects = {
    'dice_loss_fixed': dice_loss_fixed,  # Make sure dice_loss is defined in your current scope.
    'F1Score': F1Score,
    'PrecisionScore': PrecisionScore,
    'RecallScore': RecallScore,
    'specificity': specificity  # Ensure specificity is defined/imported.
})

BASE_DIR = "/workspace/eashraf/Dissertation2/WildfireSpreadTSNumpy"
dataset = WildfireDataset(base_dir=BASE_DIR,batch_size=16,sequence_length=10,random_seed=185,target_size=(128, 128))
train_dataset = dataset.get_tf_dataset('train')
val_dataset = dataset.get_tf_dataset('val')
test_dataset = dataset.get_tf_dataset('test')

model = tf.keras.models.load_model("/workspace/eashraf/Dissertation/Models/best_tuned_convlstm_model_full_0.7566.keras")
sc_test = eval_model(model,test_dataset,'Test',printres=True)
sc_val = eval_model(model,val_dataset,'Validation',printres=True)
sc_train = eval_model(model,train_dataset,'Train',printres=True)

metrics = ['Accuracy', 'SSIM','Specificity', 'IoU', 'Dice Loss', 'F1-Score', 'Recall', 'Precision', 'mAP']

train_values = [sc_train[metric] for metric in metrics]
val_values = [sc_val[metric] for metric in metrics]
test_values = [sc_test[metric] for metric in metrics]

x = np.arange(len(metrics)) * 1.5
width = 0.38  

fig, ax = plt.subplots(figsize=(18, 10))

rects1 = ax.bar(x - width, train_values, width, label='Train')
rects2 = ax.bar(x, val_values, width, label='Validation')
rects3 = ax.bar(x + width, test_values, width, label='Test')

ax.set_ylabel('Metric Value',fontsize=16)
ax.set_title('Metric Scores for Train, Validation, and Test Sets',fontsize=18)
ax.set_xticks(x)
ax.set_xticklabels(metrics, rotation=45,fontsize=16)
ax.set_yticklabels([0,0.2,0.4,0.6,0.8,1],fontsize=14)
ax.legend(fontsize=16)

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10.5)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

fig.tight_layout()
plt.show()

for batch_idx, (X_batch, Y_batch) in tqdm(enumerate(test_dataset)):
    for xseq,ytrue in zip(X_batch,Y_batch):
        fig,axes = plt.subplots(3, 5, figsize=(10, 6))
        axes = axes.flatten()
        for i in range(14):
            axes[i].imshow(xseq[i,:,:,-1])
            axes[i].set_title(f"Active Fire t-{14-i}")
        axes[-1].axis('off')
        plt.tight_layout()
        plt.show()
        ypred = np.squeeze(tmodel.predict(xseq[None,...]))
        ypred = (ypred > 0.5).astype(np.float32)
        print(np.all((ypred == 0) | (ypred == 1)))
        fig, axes = plt.subplots(1, 2, figsize=(10, 4)) 
        axes[0].imshow(ytrue)
        axes[0].set_title("True Active Fire")
        axes[1].imshow(ypred)
        axes[1].set_title("Predicted Active Fire")
        plt.tight_layout()
        plt.show()
        ssim_val, auc_val, iou_val, dice_val,f1 ,rec,prec,acc,spec= calculate_metrics(ytrue, ypred)
        print(f"Accuracy: {acc:.4f}, SSIM: {ssim_val:.4f}, IoU: {iou_val:.4f}, Dice: {dice_val:.4f}") 
        print(f"F1: {f1:.4f}, Recall: {rec:.4f}, Precision: {prec:.4f}, Specificity: {spec:.4f}")
    if batch_idx==2:
        break