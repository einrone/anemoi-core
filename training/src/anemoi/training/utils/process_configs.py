from collections import defaultdict
from copy import deepcopy

import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf


class ProcessConfigs:
    SENTINEL = object()
    TEMPORARY = defaultdict(dict)
    TEMP = dict()

    def __init__(
        self,
        base_config: DictConfig,
        hectometric: bool = False,
    ) -> None:
        """Initialize the ProcessConfigs with the base configuration.

        args:
            base_config (DictConfig): The base configuration object.
            hectometric (bool): Flag indicating if hectometric processing is needed.

        Returns
        -------
            None
        """
        OmegaConf.resolve(base_config)
        self.config = OmegaConf.to_container(base_config, resolve=True)
        self.struct = self.config
        self.struct_train = self.config["dataloader"]["hectometric_dataset_training"]
        self.struct_val = self.config["dataloader"]["hectometric_dataset_validation"]
    
    def _findcutoutkeys(self, cutout, key, hecto_dirs) -> dict:
        """Recursively search through the cutout structure
        to find a specific key and duplicate the subdictionary with the keys dataset_names.

        args:
            cutout (dict): The cutout configuration structure.
            key (str): The key to find.
            hecto_dirs (dict): The dictionary of hectometric directories.

        Returns
        -------
            dict: The modified cutout structure with the duplicated subdictionary.
        """

        def recurse(obj):
            
            if isinstance(obj, dict):
                if key in obj and "data" in obj.get(key, {}):
                    template = deepcopy(obj[key]["data"])
                    #obj[key].pop("data")
                    for k, v in hecto_dirs.items():
                        tmp = deepcopy(template)
                        if "dataset_config" in tmp:
                            self._findcutoutnulls(tmp["dataset_config"], {"dataset": v[0]})
                            self._inject_date(tmp, v[1], v[2])
                            obj[key][k] = tmp
                        else:
                            obj[key][k] = tmp
                    obj[key].pop("data", None)  # Remove the original "data" key after processing
                else:
                    # Otherwise, keep traversing deeper
                    for k, v in list(obj.items()):
                        recurse(v)

            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    if isinstance(item, dict) and key in item and "data" in item.get(key, {}):
                        # Replace the entire element if it has dataset=None
                        # obj[i].pop(key)
                        template = deepcopy(obj[i][key]["data"])
                        for k, v in hecto_dirs.items():
                            tmp = deepcopy(item[key]["data"])
                            if "dataset_config" in tmp:
                                self._findcutoutnulls(tmp["dataset_config"], {"dataset": v[0]})
                                self._inject_date(tmp, v[1], v[2])
                                obj[i][key][k] = tmp
                            else:
                                obj[i][key][k] = tmp
                        obj[i][key].pop("data", None)  # Remove the original "data" key after processing
                    else:
                        recurse(item)

        recurse(cutout)
        return cutout

    def _findcutoutnulls(self, cutout, replacement: dict) -> dict:
        """Recursively search through the cutout structure
        to find any dicts where "dataset" and other keys
        is explicitly None, and replace that dict with
        the provided replacement dict.

        args:
            cutout (dict): The cutout configuration structure.
            replacement (dict): The replacement dictionary to use.

        Returns
        -------
            dict: The modified cutout structure with replacements made.
        """

        def recurse(obj):
            if isinstance(obj, dict):
                # If this dict explicitly has dataset=None
                if obj.get("dataset", self.SENTINEL) is None:
                    # Replace the whole dict content with a deep copy of the replacement
                    # obj.clear()
                    # keep attributes at the end
                    obj.update(replacement.copy())
                else:
                    # Otherwise, keep traversing deeper
                    for k, v in list(obj.items()):
                        recurse(v)

            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    if isinstance(item, dict) and item.get("dataset", self.SENTINEL) is None:
                        # Replace the entire element if it has dataset=None
                        obj[i].update(replacement.copy())
                    else:
                        recurse(item)

        recurse(cutout)
        return cutout

    def _inject_date(self, struct, start, end):
        """Inject values into the config where None exists,
        based on a dictionary mapping.
        Example: {"cutout[0].dataset": "some_path"}
        """
        assert start is not None and end is not None, "Start and end dates must be provided."
        assert start <= end, "Start date must be less than or equal to end date."

        struct["start"] = start
        struct["end"] = end

        return struct

    def process_folder(self, folder_path, phase):
        """Process all files in a given folder, replacing cutout nulls with regional datasets
        and injecting dates based on the file names.

        args:
            folder_path (str): The path to the folder containing the files to process.

        Returns
        -------
            None
        """
        import os

        hecto_dirs = {}
        for lines in os.listdir(folder_path):
            # print(f"Processing file: {lines}")
            filename = lines.strip("\n")
            key = filename.split(".")[0]

            splitted = lines.split("_")

            start, end = splitted[1:3]
            start = f"{start[:4]}-{start[4:6]}-{start[6:8]}"
            end = f"{end[:4]}-{end[4:6]}-{end[6:8]}"
            # print(filename)
            directory = base_path + filename
            hecto_dirs[key] = (directory, start, end)
        return hecto_dirs

    def process_text_file_hecto(self, name):
        base_path = self.config["dataloader"]["hectometric_dataset_base_path"]
        with open(name) as f:
            ls = f.readlines()
            hecto_dirs = {}
            for lines in ls:
                filename = lines.strip("\n")
                key = filename.split(".")[0]
                splitted = lines.split("_")
                # if lines.endswith("v2.zarr"):
                # end = splitted[-1].split("-")[0]
                # start = splitted[-2]
                # else:
                start, end = splitted[1:3]
                start = f"{start[:4]}-{start[4:6]}-{start[6:8]}"
                end = f"{end[:4]}-{end[4:6]}-{end[6:8]}"
                directory = base_path + filename
                hecto_dirs[key] = (directory, start, end)
            return hecto_dirs

    def update(self):
        """Update the base configuration with the processed temporary structures
        containing information of each regional domain.

        args:
            None
        returns:
            DictConfig: The updated configuration object.

        """
        hecto_dirs_train = self.process_text_file_hecto(self.struct_train)
        # print("hecto_dirs_train", hecto_dirs_train)
        self.config = self._findcutoutkeys(self.config, "datasets", hecto_dirs_train) #FILLS UP SELF.TEMP
        dataset_names = list(hecto_dirs_train.keys())
        self.config["model"]["encoders"]["multi-domain"]["datasets"] = dataset_names
        self.config["model"]["decoders"]["multi-domain"]["datasets"] = dataset_names
        # print("Config after preprocessing", self.config)
        # print("DATALOADER CONFIG:")
        # print(self.config["dataloader"]["training"]["datasets"])
        return OmegaConf.create(self.config)


@hydra.main(
    version_base=None,
    config_path="/pfs/lustrep4/scratch/project_465000527/buurmans/DE_330_WP14/Anemoi/MD-PR/anemoi-core/training/src/anemoi/training/config",
    config_name="graph_from_file.yaml",
)
def main(config: DictConfig) -> None:
    pc = ProcessConfigs(base_config=config, hectometric=True)
    # pc.process
    config = pc.update()
    # print("config after processing", config)
    # print("DATALOADER CONFIG:")
    # print(config["dataloader"])
    # print("DATA CONFIG:")
    # print(config["data"])


if __name__ == "__main__":
    main()
