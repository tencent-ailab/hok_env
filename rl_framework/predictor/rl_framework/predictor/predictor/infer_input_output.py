# -*- coding: utf-8 -*-
import logging


class InferData(object):
    """An object of InferData class is used to describe
    input or output tensor for an inference request.

    Parameters
    ----------
    name : str
        The name of input/output whose data will be described by this object
    dims : list
        The shape of the associated input/output.
    data_type : str
        The datatype of the associated input/output.
    data: numpy array
        The data of the associated input/output.
    """

    def __init__(self, name, dims, data_type=None, data=None):
        self.name = name
        self.dims = dims
        self.data_type = data_type
        self.data = None
        if data is not None:
            self.set_data(data)

    def get(self):
        """Get all attributes of the tensor.

        Returns
        ------
        str
            The tensor name.
        list
            The tensor shape.
        str
            The tensor datatype.
        numpy array
            The tensor data in numpy array format.
        """
        return self.name, self.dims, self.data_type, self.data

    def get_name(self):
        """Get the name of the tensor.

        Returns
        ------
        str
            The tensor name.
        """
        return self.name

    def set_data(self, data):
        """Set the tensor data from the specified numpy array for
        input/output associated with this object.

        Parameters
        ----------
        data : numpy array
            The tensor data in numpy array format

        Raises
        ------
        Exception
            If failed to reshape data with dims.
        """
        try:
            if self.data_type is not None:
                self.data = data.reshape(self.dims).astype(self.data_type)
            else:
                self.data = data.reshape(self.dims)
        except Exception:
            logging.error(
                "can not convert data shape from {} to {}".format(
                    str(data.shape), str(self.dims)
                )
            )
            raise

    def get_data(self):
        """Get the tensor data in numpy array format.

        Returns
        ------
        numpy array
            The tensor data in numpy array format
        """
        return self.data


class InferInput(InferData):
    """An object of InferInput class is used to describe
    input tensor for an inference request.

    Parameters
    ----------
    name : str
        The name of input whose data will be described by this object
    dims : list
        The shape of the associated input.
    data_type : str
        The datatype of the associated input.
    data: numpy array
        The data of the associated input.
    """

    def __init__(self, name, dims, data_type=None, data=None):
        super(InferInput, self).__init__(name, dims, data_type, data)


class InferOutput(InferData):
    """An object of InferOutput class is used to describe
    output tensor for an inference request.

    Parameters
    ----------
    name : str
        The name of output whose data will be described by this object
    dims : list
        The shape of the associated output.
    data_type : str
        The datatype of the associated output.
    data: numpy array
        The data of the associated output.
    """

    def __init__(self, name, dims, data_type=None, data=None):
        super(InferOutput, self).__init__(name, dims, data_type, data)
