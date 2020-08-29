"""
TODO:
    write __len__ function
    write__getitem function
    Add default image shape
    Pickle reader
    delete VtkDataset
"""

import glob
import os
import pickle

import numpy as np

from PIL import Image
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple
from vtk import vtkXMLMultiBlockDataReader
from vtk.util.numpy_support import vtk_to_numpy


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
        geometry_bounds: Tuple[int] = (151, 319, 997, 601),
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
        data_names = []
        for object in os.listdir(self.base_dir):
            path_to_object = os.path.join(self.base_dir, object)
            if os.path.isdir(path_to_object):
                data_names.append(object)
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
        geometry_array = np.array(resized_geometry)[..., 0]

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
            velocity_array = simulated_velocity[..., :2]
            pressure_array = np.array(simulation_data["p"]).reshape(self.np_shape)

            flow_array = np.concatenate([velocity_array, pressure_array], axis=2)

        return flow_array

    def __len__(self) -> int:
        """ Returns the size of the dataset """
        return self.len_dataset

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:

        # Directory with the current sample
        data_directory = os.path.join(self.base_dir, self.data_names[idx])

        geometry_array = self._get_geometry_array(data_directory=data_directory)
        flow_array = self._get_flow_array(data_directory=data_directory)

        sample = {"geometry": geometry_array, "flow": flow_array}

        return sample
