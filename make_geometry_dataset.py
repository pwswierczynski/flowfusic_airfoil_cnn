import argparse
import os

from tqdm import tqdm

from src.geometry import AirfoilGeometrySampler


def parse_arguments() -> argparse.Namespace:
    """ Command line arguments parser """

    parser = argparse.ArgumentParser(
        description="Define number of samples and the discretization of the dataset."
    )

    parser.add_argument(
        "--train_samples",
        type=int,
        default=10,
        help="Number of sample airfoil geometries in the training set.",
    )
    parser.add_argument(
        "--validation_samples",
        type=int,
        default=10,
        help="Number of sample airfoil geometries in the validation set.",
    )
    parser.add_argument(
        "--test_samples",
        type=int,
        default=10,
        help="Number of sample airfoil geometries in the test set.",
    )
    parser.add_argument(
        "--n_discretization_points",
        type=int,
        default=50,
        help="Number of points in the discretization of the upper and lower profiles.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.join("data", "geometry"),
        help="Top directory of the geometry dataset.",
    )
    arguments = parser.parse_args()

    return arguments


def create_airfoil_profiles(directory: str, n_samples: int, n_points: int) -> None:
    """
    Creates a set of airfoil profiles in .geo and .stl formats in the given directory.

    :params
    directory (str): Path to the directory, in which the profiles should be saved.
    n_samples (int): Number of samples in the dataset.
    n_points (int): Number of points used in the discretization of the upper and
        lower curve of the airfoil. The total discretization of the airfoil uses thus
        2 * n_points - 2 points.
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
    args = parse_arguments()

    print("Creating train set!")
    training_data_dir = os.path.join(args.data_dir, "train")
    create_airfoil_profiles(
        directory=training_data_dir,
        n_samples=args.train_samples,
        n_points=args.n_discretization_points,
    )
    print("Creating validation set!")
    validation_data_dir = os.path.join(args.data_dir, "validation")
    create_airfoil_profiles(
        directory=validation_data_dir,
        n_samples=args.validation_samples,
        n_points=args.n_discretization_points,
    )
    print("Creating test set!")
    test_data_dir = os.path.join(args.data_dir, "test")
    create_airfoil_profiles(
        directory=test_data_dir,
        n_samples=args.test_samples,
        n_points=args.n_discretization_points,
    )
