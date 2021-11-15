import yaml
from pathlib import Path
from utils import experiment_manager

# set the paths
HOME = 'C:/Users/shafner/repos/DDA_UrbanChangeDetection'  # '/home/shafner/DDA_UrbanChangeDetection'
DATASET = 'C:/Users/shafner/datasets/spacenet7_s1s2_dataset_v3'  # '/storage/shafner/continuous_urban_change_detection/spacenet7_s1s2_dataset_v3'
DATASET_STHLM = 'C:/Users/shafner/datasets/stockholm_timeseries_dataset'  # '/storage/shafner/continuous_urban_change_detection/stockholm_timeseries_dataset'
OUTPUT = 'C:/Users/shafner/dda_urban_change_detection/output'  # '/storage/shafner/dda_urban_change_detection_output'


# TODO: define return type as cfgnode from experiment manager
def load_paths():
    C = experiment_manager.CfgNode()
    C.HOME = HOME
    C.DATASET = DATASET
    C.DATASET_STHLM = DATASET_STHLM
    C.OUTPUT = OUTPUT
    return C.clone()


def home_path() -> Path:
    dirs = load_paths()
    return Path(dirs.HOME)


def spacenet7s1s2_dataset_path() -> Path:
    dirs = load_paths()
    return Path(dirs.SPACENET7S1S2_DATASET)


def output_path() -> Path:
    dirs = load_paths()
    return Path(dirs.OUTPUT)


def setup_directories():
    dirs = load_paths()

    # inference dir
    inference_dir = Path(dirs.OUTPUT) / 'inference'
    inference_dir.mkdir(exist_ok=True)

    # evaluation dirs
    evaluation_dir = Path(dirs.OUTPUT) / 'evaluation'
    evaluation_dir.mkdir(exist_ok=True)
    quantiative_evaluation_dir = evaluation_dir / 'quantitative'
    quantiative_evaluation_dir.mkdir(exist_ok=True)
    qualitative_evaluation_dir = evaluation_dir / 'qualitative'
    qualitative_evaluation_dir.mkdir(exist_ok=True)

    # testing
    testing_dir = Path(dirs.OUTPUT) / 'testing'
    testing_dir.mkdir(exist_ok=True)

    # saving networks
    networks_dir = Path(dirs.OUTPUT) / 'networks'
    networks_dir.mkdir(exist_ok=True)


if __name__ == '__main__':
    setup_directories()
