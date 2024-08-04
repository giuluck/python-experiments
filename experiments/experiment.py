import hashlib
import itertools
import json
import logging
import math
import os
import pickle
import re
import shutil
import time
from abc import abstractmethod
from typing import Any, Dict, List, Iterable, Set, Union, Optional

import pandas as pd
import yaml
from tqdm import tqdm

from items.item import Item


class Experiment:
    """A custom experiment which contains multiple runs obtained from factorial design."""

    class Run:
        """A single run of the experiment."""

        def __init__(self,
                     key: str,
                     experiment: 'Experiment',
                     signature: Dict[str, Any],
                     execution_time: str,
                     elapsed_time: float):
            """
            :param key:
                The key (hash) of the run.

            :param experiment:
                The experiment to which the run belongs.

            :param signature:
                The signature of the run, containing either primitive type or `Item` types.

            :param execution_time:
                A timestamp which indicates when was the run executed.

            :param elapsed_time:
                The time (in seconds) that it took to execute the run.
            """
            self._key: str = key
            self._experiment: 'Experiment' = experiment
            self._signature: Dict[str, Any] = signature
            self._execution_time: str = execution_time
            self._elapsed_time: float = elapsed_time
            self._results: Dict[str, Any] = {}
            self._missing_files: Set[str] = {'main', *{filename for filename in experiment.files.values()}}

        @property
        def signature(self) -> Dict[str, Any]:
            """The signature of the run, containing either primitive type or `Item` types."""
            return self._signature.copy()

        @property
        def execution_time(self) -> str:
            """A timestamp which indicates when was the run executed."""
            return self._execution_time

        @property
        def elapsed_time(self) -> float:
            """The time (in seconds) that it took to execute the run."""
            return self._elapsed_time

        def results(self, complete: bool = False) -> Dict[str, Any]:
            """Returns the results of the run.

            :param complete:
                If True, loads all the results from all the files, otherwise returns just the already loaded results.

            :return:
                A dictionary containing the (partial or complete) results of the run.
            """
            if complete:
                for filename in list(self._missing_files):
                    self._load(filename)
            return self._results

        def __getitem__(self, item: str) -> Any:
            value = self._results.get(item)
            if value is None:
                filename = self._experiment.files.get(item)
                filename = 'main' if filename is None else filename
                content = self._load(filename=filename)
                value = content[item]
            return value

        def _load(self, filename: str) -> Dict[str, Any]:
            # get the folder of the run, load the results, and update the results dictionary accordingly
            folder = self._experiment.run_folder(key=self._key)
            filepath = os.path.join(folder, f'{filename}.pkl')
            with open(filepath, 'rb') as file:
                content = pickle.load(file=file)
            self._results.update(content)
            return content

    files: Dict[str, str] = {}
    """Dictionary which associates a filename to each result key. E.g., if the routine returns keys {'alpha', 'beta'} 
    then one might indicate {'alpha': 'file_alpha', 'beta': 'file_beta'} to store each of the keys in a different file.
    If a key is returned by the routine but is not present in this dictionary, it is stored in the 'main' file."""

    @staticmethod
    @abstractmethod
    def routine(**signature: Any) -> Dict[str, Any]:
        """Defines the routine of the run of any experiment based on the run signature.

        :param signature:
            The signature of the run.

        :return:
            A dictionary containing the results of the run.
        """
        pass

    # noinspection PyUnusedLocal
    @staticmethod
    def superset(super_signature: Dict[str, Any], **signature: Any) -> bool:
        """Checks whether the given run configuration is from a superset run of the given signature. A superset run is
        a run that has different parameters than the given one but still contains all the necessary information. For
        example, a run where a machine learning model is trained for 200 epochs is a superset run of a run where the
        same model is trained for 100 epochs (but not vice versa).

        :param super_signature:
            The configuration of the run to check as superset.

        :param signature:
            The signature of the run taken into consideration as subset.

        :return:
            Whether the run is superset or not of the signature.
        """
        return False

    @staticmethod
    def sha256(obj: Any) -> str:
        """Computes a stable hash of an object using the SHA256 algorithm.

        :param obj:
            The hashed object.

        :return:
            The computed hash.
        """
        algorithm = hashlib.sha256()
        buffer = json.dumps(obj, sort_keys=True).encode()
        algorithm.update(buffer)
        return algorithm.hexdigest()

    def __init__(self, name: str, folder: str = 'results', check_superset: bool = False):
        """
        :param name:
            The name of the experiment.

        :param folder:
            The folder where the experiment will be stored.

        :param check_superset:
            Whether to check for the presence of superset runs based on the 'similar' static function. A superset run
            is a run that has different parameters than the given one but still contains all the necessary information.
            For example, a run where a machine learning model is trained for 200 epochs is a superset run of a run
            where the same model is trained for 100 epochs (but not vice versa).
        """
        self._name: str = name
        self._folder: str = folder
        self._check_superset: bool = check_superset

    @property
    def output_file(self) -> str:
        """The name of the experiment output file."""
        return os.path.join(self._folder, f"{self._name}.yaml")

    def run_folder(self, key: str) -> str:
        """The folder where the result files of a run will be stored, given its key."""
        return os.path.join(self._folder, self._name, key)

    def execute(self, overwrite: bool = True, verbose: bool = False, **parameters: Any) -> List[Run]:
        """Executes all the runs of the experiment.

        :param overwrite:
            Whether to overwrite existing runs or to raise an error if they are already present.

        :param verbose:
            Whether to print information about each run or simply keeping track of the progress with a progress bar.

        :param parameters:
            A dictionary associating to each parameter name either a single value or an iterable of such.
            In case of iterable, the values will be used to get the factorial design.

        :return:
            The list of executed runs.
        """
        assert len(parameters) > 0, "There must be at least one parameter to define the experiment."
        output = []
        # a dictionary <key: configuration> of the previously executed runs
        runs = self._load()
        # the number of total runs and the respective iterable object
        signatures = [list(val) if isinstance(val, Iterable) else [val] for val in parameters.values()]
        total = math.prod([len(val) for val in signatures])
        # the list of parameters names and the values obtained through cartesian product
        names = list(parameters.keys())
        signatures = enumerate(itertools.product(*signatures))
        signatures = signatures if verbose else tqdm(signatures, total=total, desc='Fetching Runs')
        # iterate over every signature to execute
        for i, values in signatures:
            # check whether the run has been already executed by looking for its key
            signature = {k: v for k, v in zip(names, values)}
            signature_dump = {k: v.configuration if isinstance(v, Item) else v for k, v in signature.items()}
            key = self.sha256(obj=signature_dump)
            run = runs.get(key)
            # in case there is no run, or the run is outdated, try to look for a superset run
            # in case the search is, browse all the loaded runs until a matching one is found
            if run is None or self._outdated(signature=signature, run=run):
                run = None
                if self._check_superset:
                    for super_key, super_run in runs.items():
                        super_signature = super_run['signature'].copy()
                        is_superset = self.superset(super_signature=super_signature, **signature_dump)
                        if is_superset and not self._outdated(signature=signature, run=super_run):
                            key = super_key
                            run = super_run
                            break
            # in case there is still no run (no exact match or no superset run), build the result by executing it
            # then overwrite the output file with the configuration of all the executed runs
            # (dump the file before writing to check that it is yaml-compliant)
            if run is None:
                if verbose:
                    print(flush=True)
                    print(f'Running Experiment {i + 1} of {total}:')
                    for parameter, value in signature.items():
                        print(f'  > {parameter.upper()}: {value}')
                    print(end='', flush=True)
                execution_time = str(pd.Timestamp.now())
                elapsed_time = self._run(key=key, signature=signature, overwrite=overwrite)
                runs[key] = dict(signature=signature_dump, execution_time=execution_time, elapsed_time=elapsed_time)
                self._store(runs=runs)
            else:
                execution_time = run['execution_time']
                elapsed_time = run['elapsed_time']
            # build a run instance using the obtained information and:
            #  - append the run itself to the output list
            #  - insert or overwrite the configuration in the list of loaded runs
            output_run = Experiment.Run(
                key=key,
                experiment=self,
                signature=signature,
                execution_time=execution_time,
                elapsed_time=elapsed_time
            )
            output.append(output_run)
        # return the list of runs
        return output

    @staticmethod
    def _outdated(signature: Dict[str, Any], run: Dict[str, Any]) -> bool:
        # check if the execution time of the run is outdated with respect to any of its items component
        run = pd.Timestamp(run['execution_time'])
        for item in signature.values():
            if isinstance(item, Item) and pd.Timestamp(item.last_edit()) > run:
                return True
        return False

    def _load(self) -> Dict[str, Dict[str, Any]]:
        # if there is already a file containing information about previous runs load it, otherwise make the folder
        runs = {}
        filepath = self.output_file
        if os.path.isfile(filepath):
            with open(filepath, 'r') as file:
                runs = yaml.safe_load(file)
        else:
            os.makedirs(self._folder, exist_ok=True)
        return runs

    def _store(self, runs: Dict[str, Any]) -> None:
        # dump the runs as a yaml file
        dump = yaml.dump(runs, indent=2, default_flow_style=False, sort_keys=False)
        # code to insert a blank like between every top level object in yaml, taken from:
        # https://stackoverflow.com/questions/75535768/how-to-add-blank-lines-before-list-blocks-with-python-yaml-dump-pyyaml
        main_keys = re.compile(r"(\n\w+)")
        next_blocks = re.compile(r"(?<!:)(\n {0,6}- )")
        double_newline = lambda m: f"\n{m.group(1)}"
        dump, _ = re.subn(main_keys, double_newline, dump)
        dump, _ = re.subn(next_blocks, double_newline, dump)
        # write the output file with the configurations
        with open(self.output_file, 'w') as file:
            file.write(dump)

    def _run(self, key: str, signature: Dict[str, Any], overwrite: bool) -> float:
        # run the routine to get the results and store the elapsed time
        start = time.time()
        results = self.routine(**signature)
        elapsed_time = time.time() - start
        # check whether the results folder for this run exists already:
        #  - if there is no folder, create it
        #  - if there is a folder, but it is possible to overwrite, remove the folder and log a warning
        #  - if there is a folder, but it is not possible to overwrite, raise an error
        folder = self.run_folder(key=key)
        if not os.path.isdir(folder):
            os.makedirs(folder, exist_ok=False)
        elif overwrite:
            shutil.rmtree(folder)
            os.makedirs(folder, exist_ok=False)
            logging.info(f'Overwriting results for run "{key}"')
        else:
            raise FileExistsError(f'Results already exist for run "{key}".')
        # assign each result to the respective result file (or to the 'main' file if they are not paired)
        result_files = {'main': {}}
        result_files.update({filename: {} for filename in self.files.values()})
        for key, value in results.items():
            filename = self.files.get(key, 'main')
            result_files[filename][key] = value
        # for each filename and result dictionary, store the pickle file
        # (dump the file before writing to check that it is pickle-compliant)
        for filename, result in result_files.items():
            dump = pickle.dumps(result)
            filepath = os.path.join(folder, f'{filename}.pkl')
            with open(filepath, 'wb') as file:
                file.write(dump)
        # return a dictionary containing the configuration of the run
        return elapsed_time

    def clear(self,
              *conditions: str,
              older: Union[None, str, pd.Timestamp] = None,
              force: bool = False,
              verbose: bool = True) -> None:
        """Clears the results based on some given conditions.

        :param conditions:
            Strings of type "<item>[:<subkey>:<subsubkey>:...] = <value>" where <item> is the name of the item to check,
            potentially adding various subkeys separated by ':' to refine the search, and <value> is a single value
            which must match the item/subkey that is found in the signature of the run.

        :param older:
            A datetime element before which all the runs will be removed (ignores the condition if None).

        :param force:
            Whether to remove all the runs automatically or ask for confirmation.

        :param verbose:
            Whether to print additional information about the retrieval.
        """
        # retrieve the runs from the experiment file, and retrieve all the names of the subfolders related to the runs
        runs = self._load()
        filepath = os.path.join(self._folder, self._name)
        if not os.path.isdir(filepath):
            return
        keys = list(os.listdir(filepath))
        if verbose:
            print(f"Retrieved {len(keys)} experiments.")
        # create a dictionary of output runs, and a list of folders to be removed
        outputs = {}
        folders = []
        older = None if older is None else pd.Timestamp(older)
        # iterate over each subfolder
        for key in keys:
            # retrieve the run which corresponds to the key
            folder = self.run_folder(key=key)
            run = runs.get(key)
            # if the key has to be removed append its folder to the list, otherwise include it in the dictionary
            if self._remove(run=run, older=older, conditions=conditions):
                folders.append(folder)
            else:
                outputs[key] = run
        # if not force ask for confirmation (if the answer is not 'y' or 'yes', abort)
        # otherwise print a message if necessary
        if not force:
            msg = f"\nAre you sure you want to remove {len(keys) - len(outputs)} experiments, "
            msg += f"leaving {len(outputs)} experiments left? (Y/N) "
            choice = input(msg)
            if choice.lower() not in ['y', 'yes']:
                if verbose:
                    print(f"\nClearing procedure aborted\n")
                return
            elif verbose:
                print(f"\nClearing procedure started\n")
        elif verbose:
            print(f"Removing {len(keys) - len(outputs)} ({len(outputs)} left)\n")
        # remove all the subfolders of the deleted runs, then store the output file
        for folder in tqdm(folders, desc="Removing Folders"):
            shutil.rmtree(folder)
        self._store(runs=outputs)

    @staticmethod
    def _remove(run: Dict[str, Any], older: Optional[pd.Timestamp], conditions: Iterable[str]) -> bool:
        # remove a key if:
        #  - there is no associated run (i.e., the folder is orphan)
        #  - the run is too old and all the conditions are true
        if run is None:
            return True
        if older is not None and pd.Timestamp(run['execution_time']) > older:
            return False
        signature = run['signature']
        for condition in conditions:
            # split between <item> and <value> using the '=' symbol
            item, value = condition.split('=')
            item = str(item).strip()
            value = str(value).strip()
            # find the keys and subkeys of the items by splitting using the ':' symbol
            # then browse through the signature using the keys/subkeys
            keys = item.split(':')
            parameter = signature
            for key in keys:
                parameter = parameter[key]
            # compare the string version of the retrieved parameter with the value
            if str(parameter) != value:
                return False
        return True
