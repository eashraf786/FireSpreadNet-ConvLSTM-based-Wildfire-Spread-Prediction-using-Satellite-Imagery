import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from utils import PrecisionScore, F1Score, RecallScore, specificity, dice_loss

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    def build_convlstm_model(input_shape):
        model = models.Sequential()
        model.add(layers.ConvLSTM2D(
            filters=32,
            kernel_size=(3, 3),
            padding='same',
            return_sequences=True,
            activation='relu',
            input_shape=input_shape
        ))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.2))
        model.add(layers.ConvLSTM2D(
            filters=32,
            kernel_size=(3, 3),
            padding='same',
            return_sequences=True,
            activation='relu'
        ))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.2))
        model.add(layers.ConvLSTM2D(
            filters=32,
            kernel_size=(1, 1),
            padding='same',
            return_sequences=False,
            activation='relu'
        ))
        model.add(layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same'))
        
        model.compile(optimizer=Adam(learning_rate=1e-3),loss=dice_loss(),metrics=['accuracy', tf.keras.metrics.BinaryIoU(name='IoU'),
                                                                              F1Score(name='f1_score'),PrecisionScore(name='precision_score'),
                                                                              RecallScore(name='recall_score'),specificity])
        return model