from DataLoader import WildfireDataset
from buildmodel import build_convlstm_model
from utils import AdaptiveEarlyStopping
import tensorflow as tf
import os,pickle

BASE_DIR = "/workspace/eashraf/Dissertation2/WildfireSpreadTSNumpy"
OUTPUT_DIR = "/workspace/eashraf/Dissertation2/Models"
dataset = WildfireDataset(base_dir=BASE_DIR,batch_size=24,sequence_length=14,random_seed=185,target_size=(128, 128))
train_dataset = dataset.get_tf_dataset('train')
val_dataset = dataset.get_tf_dataset('val')
test_dataset = dataset.get_tf_dataset('test')
input_shape = (dataset.sequence_length, *dataset.target_size,23)
model = build_convlstm_model(input_shape)
print(model.summary())
mpath = os.path.join(OUTPUT_DIR,f"convlstm_model_AllFE_epoch{{epoch:02d}}_valIoU_{{val_IoU:.5f}}_DGX.keras")
adaptive_early_stopping = AdaptiveEarlyStopping(monitor='val_IoU')
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(filepath=mpath, save_best_only=True, monitor='val_IoU',verbose=1,mode='max'),
    adaptive_early_stopping,
    #tf.keras.callbacks.EarlyStopping(monitor='val_IoU', patience=20, restore_best_weights=True,mode='max'),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_IoU', factor=0.5, patience=5,mode='max',min_lr=5e-8),
]
history = model.fit(train_dataset, validation_data=val_dataset, epochs=300, callbacks=callbacks)
vs = 'full_DGX'
with open(f'/workspace/eashraf/Dissertation2/Other files/traininghistory_{vs}.pkl','wb') as f:
    pickle.dump(history.history,f)