import os
import urllib.request

import h5py
from tqdm import tqdm


def set_dataset_path(path):
    global DATASET_PATH
    DATASET_PATH = path
    os.makedirs(path, exist_ok=True)


set_dataset_path(
    os.environ.get("D4RL_DATASET_DIR", os.path.expanduser("~/.hok/datasets"))
)


def get_keys(h5file):
    keys = []

    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            keys.append(name)

    h5file.visititems(visitor)
    return keys


def filepath_from_url(dataset_url):
    _, dataset_name = os.path.split(dataset_url)
    dataset_filepath = os.path.join(DATASET_PATH, dataset_name)
    return dataset_filepath


def download_dataset_from_url(dataset_url):
    dataset_filepath = filepath_from_url(dataset_url)
    if not os.path.exists(dataset_filepath):
        print("Downloading dataset:", dataset_url, "to", dataset_filepath)
        urllib.request.urlretrieve(dataset_url, dataset_filepath)
    if not os.path.exists(dataset_filepath):
        raise IOError("Failed to download dataset from %s" % dataset_url)
    return dataset_filepath


class OfflineEnv:
    """
    Base class for offline 1v1 envs.
    Args:
        dataset_url: URL pointing to the dataset.
        ref_max_score: Maximum score (for score normalization)
        ref_min_score: Minimum score (for score normalization)
    """

    def __init__(
        self, dataset_url=None, ref_max_score=None, ref_min_score=None, **kwargs
    ):
        super(OfflineEnv, self).__init__(**kwargs)
        self.dataset_url = self._dataset_url = dataset_url
        self.ref_max_score = ref_max_score
        self.ref_min_score = ref_min_score

    def get_normalized_score(self, score):
        if (self.ref_max_score is None) or (self.ref_min_score is None):
            raise ValueError("Reference score not provided for env")
        return (score - self.ref_min_score) / (self.ref_max_score - self.ref_min_score)

    @property
    def dataset_filepath(self):
        return filepath_from_url(self.dataset_url)

    def get_dataset(self, h5path=None):
        if h5path is None:
            if self._dataset_url is None:
                raise ValueError("Offline env not configured with a dataset URL.")
            h5path = download_dataset_from_url(self.dataset_url)

        data_dict = {}
        with h5py.File(h5path, "r") as dataset_file:
            for k in tqdm(get_keys(dataset_file), desc="load datafile"):
                try:  # first try loading as an array
                    data_dict[k] = dataset_file[k][:]
                except ValueError:  # try loading as a scalar
                    data_dict[k] = dataset_file[k][()]

        # Run a few quick sanity checks
        for key in [
            "frame_no",
            "observation",
            "legal_action",
            "action",
            "reward",
            "sub_action",
            "done",
        ]:
            assert key in data_dict, "Dataset is missing key %s" % key
        n_samples = data_dict["observation"].shape[0]
        if data_dict["frame_no"].shape == (n_samples, 1):
            data_dict["frame_no"] = data_dict["frame_no"][:, 0]
        assert data_dict["frame_no"].shape == (
            n_samples,
        ), "Frame_no has wrong shape: %s" % (str(data_dict["frame_no"].shape))
        if data_dict["reward"].shape == (n_samples, 1):
            data_dict["reward"] = data_dict["rewards"][:, 0]
        assert data_dict["reward"].shape == (
            n_samples,
        ), "Reward has wrong shape: %s" % (str(data_dict["reward"].shape))
        if data_dict["done"].shape == (n_samples, 1):
            data_dict["done"] = data_dict["done"][:, 0]
        assert data_dict["done"].shape == (n_samples,), "Done has wrong shape: %s" % (
            str(data_dict["done"].shape)
        )

        return data_dict

    def get_dataset_chunk(self, chunk_id, h5path=None):
        """
        Returns a slice of the full dataset.
        Args:
            chunk_id (int): An integer representing which slice of the dataset to return.
        Returns:
            A dictionary containing observtions, actions, rewards, and terminals.
        """
        if h5path is None:
            if self._dataset_url is None:
                raise ValueError("Offline env not configured with a dataset URL.")
            h5path = download_dataset_from_url(self.dataset_url)

        dataset_file = h5py.File(h5path, "r")

        if "virtual" not in dataset_file.keys():
            raise ValueError("Dataset is not a chunked dataset")
        available_chunks = [
            int(_chunk) for _chunk in list(dataset_file["virtual"].keys())
        ]
        if chunk_id not in available_chunks:
            raise ValueError(
                "Chunk id not found: %d. Available chunks: %s"
                % (chunk_id, str(available_chunks))
            )

        load_keys = [
            "frame_no",
            "observation",
            "legal_action",
            "action",
            "reward",
            "sub_action",
            "done",
        ]
        data_dict = {
            k: dataset_file["virtual/%d/%s" % (chunk_id, k)][:] for k in load_keys
        }
        dataset_file.close()
        return data_dict
