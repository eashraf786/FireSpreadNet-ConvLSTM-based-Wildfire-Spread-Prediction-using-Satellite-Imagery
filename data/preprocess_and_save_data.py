import os,cv2,rasterio
import numpy as np
from tqdm import tqdm
from glob import glob

def process_and_save_tiff(tiff_path, target_dir, target_size=(128, 128)):
    base_name = os.path.basename(tiff_path).replace('.tif', '.npy')
    target_path = os.path.join(target_dir, base_name)
    if os.path.exists(target_path):
        return
    with rasterio.open(tiff_path) as src:
        data = src.read()
        target_height, target_width = target_size
        num_bands = data.shape[0]
        resized_data = np.zeros((num_bands, target_height, target_width), dtype=data.dtype)
        for band_idx in range(num_bands):
            band_data = data[band_idx]
            mask = np.isnan(band_data)
            band_data_clean = np.copy(band_data)
            band_data_clean[mask] = 0  
            resized_band = cv2.resize(band_data_clean, (target_width, target_height), 
                                    interpolation=cv2.INTER_NEAREST)
            if band_idx==22:
                resized_band = (resized_band > 0).astype(np.float32)
            else:
                resized_band = cv2.normalize(resized_band, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            resized_data[band_idx] = resized_band
        resized_data = np.stack(resized_data, axis=-1) 
    os.makedirs(target_dir, exist_ok=True)
    np.save(target_path, resized_data)

base_dir = '/kaggle/input/WildfireSpreadTS'
save_base_dir = '/kaggle/working/WildfireSpreadTSNumpy'
years = ['2018', '2019', '2020', '2021']
evc = 1
tifs = 0
event_dirs = []
for year in years:
    year_dir = os.path.join(base_dir, year)
    event_dirs.extend([os.path.join(year_dir, d) for d in os.listdir(year_dir) if os.path.isdir(os.path.join(year_dir, d))])
print(len(event_dirs),"total fire events")
nseq = 0
target_size = (128,128)
for event_dir in tqdm(event_dirs):
    relative_path = os.path.relpath(event_dir, base_dir)
    target_event_dir = os.path.join(save_base_dir, relative_path)
    tiff_files = glob(os.path.join(event_dir, "*.tif"))
    if len(tiff_files)>14:
        nseq += len(tiff_files)-14
    for tiff_file in tiff_files:
        process_and_save_tiff(tiff_file, target_event_dir, target_size=target_size)
print("No. of total sequences =",nseq)