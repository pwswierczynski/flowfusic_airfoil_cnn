import os
import pickle

import numpy as np

from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
from typing import Dict, List, Tuple
from tqdm import tqdm


class SimulationDataset(Dataset):
    """
    Dataset class responsible for reading the geometry and simulation data obtained
    with OpenFOAM.

    :params
    base_dir: path to the directory containing the current dataset. This could be
        training, validation or test dataset. Default in '../data/train/'.
    geometry_filename: name of PNG files, in which the geometries are described.
        Default is 'flow_geo.png'.
    simulation_filename:  name of a pickle file, in which the simulation results are
        stored. All simulated files have to have 'U' and 'p' keys for the simulated
        velocity and pressure respectively. Default is 'flow.p'
    np_shape: shape of the domain. Default in (100, 300)
    geometry_bounds: geometry PNG file may contain a layer of empty pixels around the
        domain. The computational domain is contained inside the rectangle
        described with four coordinates: (left, top, right, bottom).
        Default is (151, 319, 997, 601).
    """

    def __init__(
        self,
        base_dir: str = "../data/train",
        geometry_filename: str = "flow_geo.png",
        simulation_filename: str = "flow.p",
        np_shape: Tuple[int, int] = (100, 300),
        geometry_bounds: Tuple[int, int, int, int] = (151, 319, 997, 601),
    ) -> None:

        # Root directory of the dataset
        self.base_dir = base_dir

        # Filenames
        assert geometry_filename.endswith(
            "png"
        ), "Geometry must be given as a PNG file!"
        self.geometry_filename = geometry_filename
        self.simulation_filename = simulation_filename

        # We assume that all samples have the shame shape.
        self.np_shape = np_shape

        # Configuration for reading geometry files
        self.geometry_bounds = geometry_bounds

        # Number of samples in the dataset
        self.data_names = self._get_data_names()
        self.len_dataset = len(self.data_names)

    def _is_valid(self, path_to_object: str) -> bool:
        """
        Function verifying if the simulation files are valid and were saved correctly.
        This serves as a security layer for the correctness of the dataset class
        and prevents errors during the training.

        :params:
        path_to_object: path to the simulation's directory.

        :returns:
        is_valid: True if the files are in correct format and false otherwise.
        """
        # Check if geometry is correctly saved as a PNG file
        try:
            # Load geometry
            path_to_geometry = os.path.join(path_to_object, self.geometry_filename)

            # Preprocessing geometry
            raw_geometry = Image.open(path_to_geometry)
            _ = raw_geometry.crop(self.geometry_bounds)
        except UnidentifiedImageError:
            return False
        except OSError:
            return False

        # Check if the simulation results are correctly saved as a valid pickle file
        try:
            path_to_simulation = os.path.join(path_to_object, self.simulation_filename)

            with open(path_to_simulation, "rb") as simulation_file:
                pickle.load(simulation_file)
        except EOFError:
            return False
        except pickle.UnpicklingError:
            return False

        return True

    def _get_data_names(self) -> List[str]:
        """
        Obtains names of data points.
        This is done by checking folders in the base_dir.
        It assumes that base_dir contains only folders with data.

        This function is also a second security layer in case saving the data
        during the simulation process failed for some samples.

        :returns
        data_names - list of sample data points
        """
        data_names: list = []
        print("Checking validity of the data...")
        for file in tqdm(os.listdir(self.base_dir)):
            path_to_object = os.path.join(self.base_dir, file)
            if os.path.isdir(path_to_object) and self._is_valid(path_to_object):
                data_names.append(file)

        print("Only correct data samples will be used!")
        return data_names

    def _get_geometry_array(self, data_directory: str) -> np.ndarray:
        """
        Function reading the geometry information of the current sample.
        This function applies necessary preprocessing of the data
        by cropping and resizing provided image.

        :params
        data_directory: path to the current simulation sample

        :returns
        geometry_array: numpy array, in which 0 denotes the empty domain and
            1 region occupied by the simulated geometry of an airfoil
        """

        # Load geometry
        path_to_geometry = os.path.join(data_directory, self.geometry_filename)

        # Preprocessing geometry
        raw_geometry = Image.open(path_to_geometry)
        cropped_geometry = raw_geometry.crop(self.geometry_bounds)

        # PIL swaps axis compared to numpy arrays
        geometry_shape = (self.np_shape[1], self.np_shape[0])
        resized_geometry = cropped_geometry.resize(
            geometry_shape, resample=Image.NEAREST
        )
        geometry_array_flat = np.array(resized_geometry)[..., 0]

        # Geometry can be described using binary values only
        binary_geometry = (geometry_array_flat > 128).astype(np.float32)
        geometry_array = np.expand_dims(binary_geometry, axis=2)

        # Permute axis so that they are compatible with PyTorch DataLoader.
        # Required order is Channel x Width x Height
        geometry_array = np.transpose(geometry_array, (2, 0, 1))

        return geometry_array

    def _get_flow_array(self, data_directory: str) -> np.ndarray:
        """
        Function reading the simylated velocity and pressure of the current sample.

        :params
        data_directory: path to the current simulation sample

        :returns
        flow_array: numpy array, in which first two channels contain the simulated
            velocity and the third channel contains the simulated pressure
        """

        path_to_simulation = os.path.join(data_directory, self.simulation_filename)

        with open(path_to_simulation, "rb") as simulation_file:

            simulation_data = pickle.load(simulation_file)
            simulated_velocity = np.array(simulation_data["U"]).reshape(
                (*self.np_shape, 3)
            )

            # Get only the first two channels containing x- and y- velocity components
            # Note that the first channel is mirrored in the pickle file!
            velocity_array = simulated_velocity[::-1, :, :2]
            pressure_array = np.array(simulation_data["p"]).reshape(*self.np_shape, 1)
            pressure_array = pressure_array[::-1, ...]

            flow_array = np.concatenate([velocity_array, pressure_array], axis=2)

        # Permute axis so that they are compatible with PyTorch DataLoader.
        # Required order is Channel x Width x Height
        flow_array = np.transpose(flow_array, (2, 0, 1))

        return flow_array

    def __len__(self) -> int:
        """ Returns the size of the dataset """
        return self.len_dataset

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """ Returns chosen element from the dataset """

        # Directory with the current sample
        data_directory = os.path.join(self.base_dir, self.data_names[idx])

        geometry_array = self._get_geometry_array(data_directory=data_directory)
        flow_array = self._get_flow_array(data_directory=data_directory)

        # Making sure that velocity and pressure are zero inside the geometry
        flow_array = (1 - geometry_array) * flow_array

        sample = {"geometry": geometry_array, "flow": flow_array}

        return sample
