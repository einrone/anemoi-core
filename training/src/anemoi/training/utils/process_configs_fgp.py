from collections import defaultdict
from copy import deepcopy

import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf


class ProcessConfigs:
    SENTINEL = object()
    TEMPORARY = defaultdict(dict)
    DATATMP = defaultdict(dict)

    def __init__(
        self,
        base_config: DictConfig,
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
        self.struct = self.config["system"]["input"]["template"]
        self.data_struct = self.config["data"]
        if hectometric:
            print(self.struct)
            self.struct_train = self.config["dataloader"]["hectometric_dataset_training"]
            self.struct_val = self.config["dataloader"]["hectometric_dataset_validation"]
            print(self.struct_train)
            print(self.struct_val)
        else:
            self.regional = self.config["dataloader"]["regional_datasets"]
        self.TEMPORARY["training"] = {}
        self.TEMPORARY["validation"] = {}

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

    def process_text_file_hecto(self, name, phase):
        base_path = self.config["dataloader"]["hectometric_dataset_base_path"]
        with open(name) as f:
            print("reading file from .txt")
            ls = f.readlines()
            for lines in ls:
                filename = lines.strip("\n")
                key = filename.split(".")[0]

                splitted = lines.split("_")

                start, end = splitted[1:3]
                start = f"{start[:4]}-{start[4:6]}-{start[6:8]}"
                end = f"{end[:4]}-{end[4:6]}-{end[6:8]}"
                print(filename)
                directory = base_path + filename
                if filename.endswith("v2.zarr"):
                    print("dataset v2")
                    import os

                    directory = directory
                    print(directory)
                    # tot_filename = os.listdir(directory)[0]
                    d_filedir = (
                        "/leonardo_work/DestE_340_26/users/mvangind/hecto_decumulation/hecto_original/" + filename
                    )
                    d = os.listdir(d_filedir)[0]
                    print("file directory:", d_filedir)
                    print("base_path:", base_path)
                    print("d:", d)
                    self.TEMPORARY[phase][key] = {
                        "dataset_config": self._findcutoutnulls(
                            deepcopy(self.struct),
                            replacement={"dataset": base_path + d},
                        ),
                    }
                else:
                    print("dataset v3")
                    self.TEMPORARY[phase][key] = {
                        "dataset_config": self._findcutoutnulls(
                            deepcopy(self.struct),
                            replacement={"dataset": directory},
                        ),
                    }
                self.TEMPORARY[phase][key] = self._inject_date(
                    self.TEMPORARY[phase][key],
                    start=start,
                    end=end,
                )
                self.data_config = self.config["data"]
                self.DATATMP["dataset"] = {}
                if key not in self.DATATMP["dataset"]:
                    self.DATATMP["datasets"][key] = self.data_struct["datasets"]["data"]

    def process_single_file_hecto(self, name, phase):
        print("PROCESS SINGLE FILE HECTO")
        filename = name
        print(filename)
        name = name.split("/")[6]
        print(name)
        key = name.split(".")[0]

        splitted = name.split("_")
        print(splitted)
        start, end = splitted[1:3]
        print(start, end)
        start = f"{start[:4]}-{start[4:6]}-{start[6:8]}"
        end = f"{end[:4]}-{end[4:6]}-{end[6:8]}"
        self.TEMPORARY[phase][key]["dataset_config"] = self._findcutoutnulls(
            deepcopy(self.struct),
            replacement={"dataset": filename},
        )
        self.TEMPORARY[phase][key] = self._inject_date(
            self.TEMPORARY[phase][key],
            start=start,
            end=end,
        )

    @property
    def process(self):
        """Process the configurations to replace
        cutout nulls with regional datasets
        and inject dates

        args:
            None
        returns:
            None
        """
        print("PROCESS IN PROGRESS")

        if self.hectometric:
            for name, phase in [
                (self.struct_train, "training"),
                (self.struct_val, "validation"),
            ]:
                if name.endswith(".txt"):
                    print("PROCESS TEXT FILE HECTO")
                    self.process_text_file_hecto(name, phase)
                else:
                    print("PROCESS SINGLE FILE HECTO")
                    self.process_single_file_hecto(name, phase)
        else:
            for phase in ["training", "validation"]:
                for region, args in self.regional.items():
                    self.TEMPORARY[phase][region]["dataset_config"] = self._findcutoutnulls(
                        deepcopy(self.struct),
                        replacement=args,
                    )

                    # print(self.TEMPORARY[phase][region])
                    self.TEMPORARY[phase][region] = self._inject_date(
                        self.TEMPORARY[phase][region],
                        start=self.struct[f"{phase}_periods"][region]["start"],
                        end=self.struct[f"{phase}_periods"][region]["end"],
                    )

    def update(self):
        """Update the base configuration with the processed temporary structures
        containing information of each regional domain.

        args:
            None
        returns:
            DictConfig: The updated configuration object.

        """
        for phase, structs in self.TEMPORARY.items():
            self.config["dataloader"][phase]["datasets"] = structs
        self.config["data"]["datasets"] = self.DATATMP["datasets"]
        return OmegaConf.create(self.config)


@hydra.main(
    version_base=None,
    config_path="/leonardo_work/DestE_340_26/users/sbuurman/MD-PR/forked_PR/anemoi-core/training/src/anemoi/training/config/",
    config_name="hectometric_finetuning_lowres.yaml",
)
def main(config: DictConfig) -> None:
    pc = ProcessConfigs(base_config=config, hectometric=True)
    pc.process
    config = pc.update()


if __name__ == "__main__":
    main()
