import os

from tqdm import tqdm

train_samples = 5
n_points = 50

from src.geometry import AirfoilGeometrySampler

airfoil_sampler = AirfoilGeometrySampler(n_points=n_points)
for i in tqdm(range(train_samples)):

    dir_name = str(i).zfill(6)
    dir_to_save = os.path.join(os.pardir, "data", "geometry", "train", dir_name)
    print(dir_to_save)
    filename = "geometry"

    airfoil_sampler.create_airfoil_geometry(dir_to_save=dir_to_save, filename=filename)
