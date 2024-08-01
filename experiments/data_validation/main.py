import argparse
from pathlib import Path

from f3dasm import ExperimentData

from l2co import create_entire_dataset

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', type=str,
                    help="Path to the experiment data", required=True)

args = parser.parse_args()
path = Path(args.path)

xarr, experimentdata = create_entire_dataset(ExperimentData.from_file(path))
xarr.to_netcdf(path / 'entire_dataset.nc')
experimentdata.store()
