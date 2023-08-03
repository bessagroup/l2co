#                                                                       Modulesset
# =============================================================================

from copy import deepcopy
# Standard
from pathlib import Path
from typing import List, Union

# Third-party
import pandas as pd

# Local
from l2co._logging import logger

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


class FileHandler:
    def __init__(self, dir_name: Path, extension: str, exceptions: List[Path] = None):
        """Filehandler class that tracks newly created files
        and manages post-processing actions

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
        self.processed_files: List[Path] = []
        self.error_files: List[Path] = []

    def _set_exceptions(self, exceptions: Union[None, List[Path]]) -> List[Path]:
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

        return [self.dir_name / exception.with_suffix(self.extension) for exception in exceptions]

    def retrieve_tracked_files(self) -> List[Path]:
        """Retrieve all files in the directory that pass the extension criteria

        Returns
        -------
            list of filenames in the directory to track
        """

        file_list = list(self.dir_name.glob(f"*{self.extension}"))
        return [file_path for file_path in file_list if file_path not in self.exceptions]

    def retrieve_files_to_process(self) -> List[Path]:
        """Retrieve all the files from the directory that need to be processed

        Returns
        -------
            list of filenames that need to be processed
        """
        all_files = self.retrieve_tracked_files()
        return list(filter(lambda file: file not in self.processed_files, all_files))

    def tick_processed(self, processed_file: Path, errorcode: int):
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
        """Main script to call. Runs one sweep down the directory to check for files
        to process and processes them sequentially
        """
        # Retrieve list of files to process
        to_process = self.retrieve_files_to_process()

        # Execute the action for every file in the list
        # Tick the processed files as processed
        for filename in (f for f in to_process):
            errorcode = self.execute(filename)
            self.tick_processed(filename, errorcode)

    def execute(self, filename: Path) -> int:
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


class CombineMarginOfVictoryFileHandler(FileHandler):
    def __init__(self, dir_name: Path, master_file: Path, exceptions: List[Path] = None, remove_files: bool = True):
        self.remove_files = remove_files
        self.master_file = master_file.with_suffix('.csv')

        if exceptions is None:
            exceptions = []
        exceptions.append(self.master_file)

        self.n_tasks = 0
        super().__init__(dir_name=dir_name, extension=".csv", exceptions=exceptions)

    def execute(self, filename: Path, verbal: bool = False) -> int:
        try:
            csv_file = pd.read_csv(filename, index_col=0)

            if not self.master_file.exists():
                mean_margin_of_victory = csv_file
            else:
                mean_margin_of_victory = pd.read_csv(self.master_file, index_col=0)

                # Create a copy
                temp_mean_margin_of_victory = deepcopy(mean_margin_of_victory)

                # multiply all the values of sum_margin_of_victory by n_tasks
                temp_mean_margin_of_victory *= self.n_tasks

                # Add the csv_file
                temp_mean_margin_of_victory += csv_file

                # Divide by n_tasks + 1
                temp_mean_margin_of_victory /= (self.n_tasks + 1)

                # Some checks
                # Check if NaN values are present
                if temp_mean_margin_of_victory.isnull().values.any():
                    logger.debug(f"NaN values are present in df_mov from {filename.stem}")
                    return 1

                # Everything good ?
                # Overwrite sum_margin_of_victory with temporary and increment n_tasks
                mean_margin_of_victory = temp_mean_margin_of_victory
                self.n_tasks += 1

            # Overwrite masterfile to disk
            mean_margin_of_victory.to_csv(self.master_file)

            # Remove file from disk
            if self.remove_files:
                filename.unlink()

            logger.info(f"Added {filename.stem} to {self.master_file.stem} !")
            return 0

        except Exception:
            logger.error(f"ERROR with handeling {filename.stem}")
            return 1


class CombineFeaturesFileHandler(FileHandler):
    def __init__(self, dir_name: Path, master_file: Path, exceptions: List[Path] = None):
        if exceptions is None:
            exceptions = []
        exceptions.append(master_file.with_suffix('.csv'))

        self.master_file = master_file.with_suffix('.csv')
        super().__init__(dir_name=dir_name, extension='.csv', exceptions=exceptions)

    def master_file_full_path(self) -> Path:
        return self.dir_name / self.master_file
        # return os.path.join(os.path.curdir, self.dir_name, f"{self.master_file}.{self.extension}")

    def does_master_file_exist(self) -> bool:
        return self.master_file_full_path().exists()
        # return os.path.exists(self.master_file_full_path())

    def execute(self, filename: Path) -> int:
        try:
            csv_file = pd.read_csv(filename)

            if not self.does_master_file_exist():
                updated_master_file = csv_file
            else:
                master_file = pd.read_csv(self.master_file_full_path(), index_col=0)
                updated_master_file = pd.concat([master_file, csv_file], ignore_index=True)

            # Overwrite masterfile to disc
            updated_master_file.to_csv(self.master_file_full_path())

            # Remove file from disk
            filename.unlink()
            return 0

        except Exception:
            logger.error(f"ERROR with handeling {filename.stem}")
            return 1
