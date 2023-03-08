import glob
import os
from typing import List, Union

import pandas as pd


class FileHandler:
    def __init__(self, dir_name: str, extension: str, exceptions: List[str] = None):
        self.dir_name = dir_name
        self.extension = extension
        self.processed_files = []
        self.exceptions = self._set_exceptions(exceptions)

    def _set_exceptions(self, exceptions: Union[None, List[str]]) -> List[str]:
        if exceptions is None:
            return []

        return [
            os.path.join(os.path.curdir, self.dir_name, f'{exception}.{self.extension}')
            for exception in exceptions
            ]

    def retrieve_all_files(self) -> List[str]:
        all_files_without_exceptions = glob.glob(os.path.join(os.path.curdir, self.dir_name, f'*.{self.extension}'))
        return list(filter(lambda file: file not in self.exceptions, all_files_without_exceptions))

    def retrieve_files_to_process(self) -> List[str]:
        all_files = self.retrieve_all_files()
        return list(filter(lambda file: file not in self.processed_files, all_files))

    def tick_processed(self, process_list: List[str]):
        self.processed_files.extend(process_list)

    def run(self):
        # Retrieve to_process list
        to_process = self.retrieve_files_to_process()

        # Do the action for every file in the list
        for filename in to_process:
            self.execute(filename)

        # Add the processed items to the list
        self.tick_processed(to_process)

    def execute(self, filename: str):
        print(f'executing task to {filename}!')

class CombineFeaturesFileHandler(FileHandler):
    def __init__(self, dir_name: str, master_file: str, exceptions: List[str] = None):
        if exceptions is None:
            exceptions = []
        exceptions.append(master_file)

        self.master_file = master_file
        extension = "csv"
        super().__init__(dir_name, extension, exceptions)

    def master_file_full_path(self):
        return os.path.join(os.path.curdir, self.dir_name, f"{self.master_file}.{self.extension}")

    def exist_master_file(self) -> bool:
        return os.path.exists(self.master_file_full_path())

    def execute(self, filename: str, verbal: bool = False):
        f = pd.read_csv(filename)

        if not self.exist_master_file():
            updated_master_file = f
        else:
            master_file = pd.read_csv(self.master_file_full_path(), index_col=0)
            updated_master_file = pd.concat([master_file, f], ignore_index=True)

        # Write to disc
        updated_master_file.to_csv(self.master_file_full_path())

        # Remove file
        os.remove(filename)

        if verbal:
            print(f"Added {filename}!")
