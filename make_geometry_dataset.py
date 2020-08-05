import os

from tqdm import tqdm

train_samples = 5
validation_samples = 2
test_samples = 2
n_points = 50

DATA_DIR = os.path.join("data", "geometry")

from src.geometry import AirfoilGeometrySampler


def create_airfoil_profiles(directory: str, n_samples: int) -> None:
    """
	Creates a set of airfoil profiles in .geo and .stl formats in the given directory.

	:param directory: Path to the directory, in which the profiles should be saved.
	:param n_samples: Number of samples in the dataset.
	"""

    airfoil_sampler = AirfoilGeometrySampler(n_points=n_points)
    for i in tqdm(range(n_samples)):
        subdir_name = str(i).zfill(6)
        dir_to_save = os.path.join(directory, subdir_name)
        filename = "geometry"

        airfoil_sampler.create_airfoil_geometry(
            dir_to_save=dir_to_save, filename=filename
        )


if __name__ == "__main__":

    print("Creating train set!")
    training_data_dir = os.path.join(DATA_DIR, "train")
    create_airfoil_profiles(directory=training_data_dir, n_samples=train_samples)
    print("Creating validation set!")
    validation_data_dir = os.path.join(DATA_DIR, "validation")
    create_airfoil_profiles(directory=validation_data_dir, n_samples=validation_samples)
    print("Creating test set!")
    test_data_dir = os.path.join(DATA_DIR, "test")
    create_airfoil_profiles(directory=test_data_dir, n_samples=test_samples)