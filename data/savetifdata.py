import datetime,tqdm,rasterio,os
import matplotlib.pyplot as plt
from remotezip import RemoteZip
zip_url = "https://zenodo.org/record/8006177/files/WildfireSpreadTS.zip?download=1"
d1 = datetime.date(2018,4,4)
d2 = datetime.date(2018,4,28)
diff = d2 - d1
yr = 2018
evid = 'fire_21690102'
for i in tqdm.tqdm(range(diff.days + 1)):
	dt = str(d1 + datetime.timedelta(i))
	target_file = f"{yr}/{evid}/{dt}.tif"
	try:
		fn = f"WildfireSpreadTS tiff files/{yr}_{evid}_2/Tiff files/{dt}.tif"
		dir = f"WildfireSpreadTS tiff files/{yr}_{evid}_2/Tiff files"
		if not os.path.exists(dir):
			os.makedirs(dir)
		with RemoteZip(zip_url) as zip_file:
			with zip_file.open(target_file) as tif:
				data = tif.read()
				with open(fn, "wb") as out_file:
					out_file.write(data)
		print(f"Successfully downloaded and extracted the file - {fn}!")
	except Exception as e:
		print("An error occurred:", e)
	
	with rasterio.open(fn) as dataset:
		all_bands = dataset.read()
		band_names = dataset.descriptions
	im = plt.imshow(all_bands[-1])
	plt.title(band_names[-1])
	plt.axis('off')
	""" fig, axes = plt.subplots(5, 5, figsize=(20, 20))
	axes = axes.flatten()
	num_bands = all_bands.shape[0]
	for i in range(5*5):
		ax = axes[i]
		if i < num_bands:
			im = ax.imshow(all_bands[i])
			ax.set_title(band_names[i])
			ax.axis('off') 
		else:
			ax.axis('off')
	plt.tight_layout() """
	dir2 = f'WildfireSpreadTS tiff files/{yr}_{evid}_2/Subplots'
	if not os.path.exists(dir2):
		os.makedirs(dir2)
	plt.savefig(f'{dir2}/{dt} - bands_viz.png')
	plt.close()