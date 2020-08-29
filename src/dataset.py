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
    def __init__(
        self,
        base_dir: str = "../data/train",
        np_shape: Tuple[int, int] = (100, 300),
        geometry_filename: str = "flow_geo.png",
        simulation_filename: str = "flow.p",
        geometry_bounds: Tuple[int] = (151, 319, 997, 601),
    ) -> None:

        # Root directory of the dataset
        self.base_dir = base_dir

        # Filenames
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


class VtkDataset(Dataset):
    def __init__(
        self, base_dir: str = "../data", np_shape: Optional[Tuple[int]] = None
    ) -> None:

        # Root directory of the dataset
        self.base_dir = base_dir

        # List of all simulation's config xml files
        path_to_xmls = os.path.join(self.base_dir, "xml_runs")
        self.xml_files = [
            file for file in os.listdir(path_to_xmls) if file.endswith("xml")
        ]

        # Data shape. If not provided, will be read in the first iteration.
        # We assume that all samples have the shame shape.
        self.np_shape = np_shape

    def _read_point_data_from_vtk(self, filename: str):
        """Reads point_data from the provided VTK file
        :param
        filename: (str) name of the VTK file containing the data.

        :return
        point_data: (vtkPointData) Points stored in the VTK container
        """

        # TODO: Do we need to instantiate a new reader in every call?
        reader = vtkXMLMultiBlockDataReader()
        reader.SetFileName(filename)
        reader.Update()
        data = reader.GetOutput()
        data_iterator = data.NewIterator()
        img_data = data_iterator.GetCurrentDataObject()

        # If the data shape is not yet provided, read it out from the file.
        if self.np_shape is None:
            img_shape = img_data.GetDimensions()
            self.np_shape = (img_shape[1], img_shape[0], 1)

        point_data = img_data.GetPointData()

        return point_data

    def _read_numpy_from_points(
        self, point_data, ndims: int = 1, array_idx: int = 0
    ) -> np.ndarray:
        """Function reading data as numpy array in a desired shape
        :param
        point_data: (vtkPointData) vtk point data
        ndims: (int) dimensionality of the quantity, e.g. pressure has dimension 1,
            whereas velocity has dimension 2.
        array_idx: (int) index of the chosen array in the point_data container.

        :return
        numpy_array: (np.ndarray) point data in the desired form and shape
            turned into a numpy array.
        """

        array_data = point_data.GetArray(array_idx)
        np_array = vtk_to_numpy(array_data)

        data_shape = (self.np_shape[0], self.np_shape[1], ndims)
        numpy_array = np_array.reshape(data_shape)

        return numpy_array

    def __len__(self) -> int:
        """ Returns the size of the dataset """
        return len(self.xml_files)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:

        # Choose one of the configuration files
        path_to_file = self.xml_files[idx]

        # Find all other directories associated with this simulation
        filename = os.path.basename(path_to_file)
        basename = os.path.splitext(filename)[0]
        data_folder_name = "runlog_" + basename.split("_")[1]
        dir_name = os.path.join(self.base_dir, "simulation_data", data_folder_name)

        # get needed filenames
        geometry_file = os.path.join(dir_name, "vtkData", "geometry_iT0000000.vtm")
        steady_flow_file = glob.glob(dir_name + "/vtkData/data/*.vtm")[0]

        # Read the VTK data as a numpy array
        point_data = self._read_point_data_from_vtk(filename=geometry_file)
        geometry_array = self._read_numpy_from_points(
            point_data=point_data, ndims=1, array_idx=0
        )

        point_data = self._read_point_data_from_vtk(filename=steady_flow_file)
        velocity_array = self._read_numpy_from_points(
            point_data=point_data, ndims=2, array_idx=0
        )
        pressure_array = self._read_numpy_from_points(
            point_data=point_data, ndims=1, array_idx=1
        )
        steady_flow_array = np.concatenate([velocity_array, pressure_array], axis=2)

        # Permute axis so that they are compatible with PyTorch DataLoader.
        # Required order is Channel x Width x Height
        geometry_array = np.transpose(geometry_array, (2, 0, 1))
        steady_flow_array = np.transpose(steady_flow_array, (2, 0, 1))

        sample = {"geometry": geometry_array, "flow": steady_flow_array}

        return sample
