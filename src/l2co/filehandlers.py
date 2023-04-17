#                                                                       Modulesset
# =============================================================================

# Standard
import glob
import os
from typing import List, Union

# Third-party
import pandas as pd

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


class FileHandler:
    def __init__(self, dir_name: str, extension: str, exceptions: List[str] = None):
        """Filehandler class that tracks newly created files and manages post-processing actions

        Parameters
        ----------
        dir_name
            filedirectory (relative or absolute) to track
        extension
            the file-exentsion to filter newly created files (without the .)
        exceptions, optional
            files that are exceptions and should not be tracked, by default None
        """
        self.dir_name = dir_name
        self.extension = extension
        self.exceptions = self._set_exceptions(exceptions)
        self._create_empty_error_and_processed_list()

    def _create_empty_error_and_processed_list(self):
        self.processed_files = []
        self.error_files = []

    def _set_exceptions(self, exceptions: Union[None, List[str]]) -> List[str]:
        """Convert exception filenames to full paths

        Parameters
        ----------
        exceptions
            list of filenames to except

        Returns
        -------
            list of full path filenames to except
        """
        if exceptions is None:
            return []

        return [
            os.path.join(os.path.curdir, self.dir_name, f'{exception}.{self.extension}')
            for exception in exceptions
        ]

    def retrieve_all_files(self) -> List[str]:
        """Retrieve all files in the directory that pass the extension criteria

        Returns
        -------
            list of filenames in the directory to track
        """
        all_files_without_exceptions = glob.glob(os.path.join(os.path.curdir, self.dir_name, f'*.{self.extension}'))
        return list(filter(lambda file: file not in self.exceptions, all_files_without_exceptions))

    def retrieve_files_to_process(self) -> List[str]:
        """Retrieve all the files from the directory that need to be processed

        Returns
        -------
            list of filenames that need to be processed
        """
        all_files = self.retrieve_all_files()
        return list(filter(lambda file: file not in self.processed_files, all_files))

    def tick_processed(self, processed_file: str, errorcode: int):
        """Move the processed files to the processed file list

        Parameters
        ----------
        processed_file
            file that has been processed
        errorcode
            errorcode of the job. If non-zero, the file is also placed in the error_files list!
        """
        self.processed_files.append(processed_file)
        if errorcode != 0:
            self.error_files.append(processed_file)

    def run(self):
        """Main script to call. Runs one sweep down the directory
        to check for files to process and processes them sequentially
        """
        # Retrieve list of files to process
        to_process = self.retrieve_files_to_process()

        # Execute the action for every file in the list
        # Tick the processed files as processed
        for filename in to_process:
            errorcode = self.execute(filename)
            self.tick_processed(filename, errorcode)

    def execute(self, filename: str) -> int:
        """Script the needs to be executed for newly created files

        Parameters
        ----------
        filename
            filename of the newly created file

        Returns
        -------
            errorcode after completion. 0 for success and 1 for fail
        """
        return 0


class CombineFeaturesFileHandler(FileHandler):
    def __init__(self, dir_name: str, master_file: str, exceptions: List[str] = None):
        if exceptions is None:
            exceptions = []
        exceptions.append(master_file)

        self.master_file = master_file
        extension = "csv"
        super().__init__(dir_name=dir_name, extension=extension, exceptions=exceptions)

    def master_file_full_path(self) -> str:
        return os.path.join(os.path.curdir, self.dir_name, f"{self.master_file}.{self.extension}")

    def does_master_file_exist(self) -> bool:
        return os.path.exists(self.master_file_full_path())

    def execute(self, filename: str, verbal: bool = False) -> int:
        try:
            csv_file = pd.read_csv(filename)

            if not self.does_master_file_exist():
                updated_master_file = csv_file
            else:
                master_file = pd.read_csv(self.master_file_full_path(), index_col=0)
                updated_master_file = pd.concat([master_file, csv_file], ignore_index=True)

            # Overwrite masterfile to disc
            updated_master_file.to_csv(self.master_file_full_path())

            # Remove file from disc
            os.remove(filename)

            if verbal:
                print(f"Added {filename}!")

            return 0

        except Exception:
            if verbal:
                print(f"ERROR with handling {filename}!")

            return 1
