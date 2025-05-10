import os,random
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from glob import glob

class WildfireDataset:
    def __init__(self, base_dir, sequence_length, batch_size, target_size, random_seed=42):
        """
        Initialize the wildfire dataset loader
        
        Args:
            base_dir: Directory containing year folders (2018, 2019, 2020, 2021)
            sequence_length: Number of days in the input sequence
            batch_size: Batch size for training
        """
        self.base_dir = base_dir
        self.sequence_length = sequence_length
        self.batch_size = batch_size  
        self.target_size = target_size
        
        self.fire_events = self._get_all_fire_events()
        self._split_data(seed=random_seed)
    
    def _get_all_fire_events(self):
        fire_events = []
        for year in ['2018', '2019', '2020', '2021']:
            year_dir = os.path.join(self.base_dir, year)
            if os.path.exists(year_dir):
                fire_dirs = [os.path.join(year_dir, d) for d in os.listdir(year_dir) if os.path.isdir(os.path.join(year_dir, d))]
                fire_events.extend(fire_dirs)
        return fire_events
        
    def _split_data(self, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42):
        random.seed(seed)
        train_events, temp_events = train_test_split(self.fire_events, train_size=train_ratio, random_state=seed)
        val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
        val_events, test_events = train_test_split(temp_events, train_size=val_ratio_adjusted, random_state=seed)
        self.train_events = train_events
        self.val_events = val_events
        self.test_events = test_events
        print(f"Train: {len(train_events)} events, Validation: {len(val_events)} events, Test: {len(test_events)} events")
        print("Few Train EventIDs:")
        print(train_events[:5],'\n')

    def get_tf_dataset(self, subset):
        fire_events = {
            'train': self.train_events,
            'val': self.val_events,
            'test': self.test_events
        }[subset]
        dataset = tf.data.Dataset.from_generator(lambda: self._data_generator(fire_events), 
                                                 output_signature=(
                                                     tf.TensorSpec(shape=(self.sequence_length, *self.target_size,23), dtype=tf.float32),
                                                     tf.TensorSpec(shape=self.target_size, dtype=tf.float32)
                                                 ))
        return dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

    def _data_generator(self, fire_events):
        for event_dir in fire_events:
            event_dir = event_dir.replace('input/WildfireSpreadTS','working/WildfireSpreadTSNumpy')
            sequences = [(sorted(glob(os.path.join(event_dir, "*.npy")))[i:i+self.sequence_length], 
                          sorted(glob(os.path.join(event_dir, "*.npy")))[i+self.sequence_length]) 
                          for i in range(len(glob(os.path.join(event_dir, "*.npy"))) - self.sequence_length)
                        ]
            for input_files, target_file in sequences:
                yield self._load_sequence(input_files, target_file)
        
    def _load_sequence(self, input_files, target_file):
        inputs = [np.load(file) for file in input_files]
        target = np.load(target_file)[:,:,-1]
        inputs = np.stack(inputs, axis=0)
        return inputs, target