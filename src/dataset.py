import glob
import os

import numpy as np

from torch.utils.data import Dataset
from typing import Optional, Tuple
from vtk import vtkXMLMultiBlockDataReader
from vtk.util.numpy_support import vtk_to_numpy


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
        """ Reads point_data from the provided VTK file
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
        """ Function reading data as numpy array in a desired shape
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

    def __len__(self):
        """ Returns the size of the dataset """
        return len(self.xml_files)

    def __getitem__(self, idx: int):

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

        sample = {"geometry": geometry_array, "flow": steady_flow_array}

        return sample
