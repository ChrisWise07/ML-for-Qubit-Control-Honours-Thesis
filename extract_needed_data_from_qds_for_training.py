import zipfile
import os
import pickle
import multiprocessing as mp

DISTORTION = True
KEYS_TO_SAVE = [
    "pulses",
    "Vo_operator",
    "expectations",
    "U0",
    "pulse_parameters",
]
if DISTORTION:
    KEYS_TO_SAVE.append("distorted_pulses")

NOISE_PROFILE = "N3N6"

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
NEW_DIR_PATH = f"{ROOT_DIR}/G_1q_XY_XZ_{NOISE_PROFILE}_reduced"
QDATA_DIR_PATH = f"{ROOT_DIR}/G_1q_XY_XZ_{NOISE_PROFILE}_D.zip"


def process_file(file):
    try:
        with zipfile.ZipFile(QDATA_DIR_PATH, mode="r") as fzip:
            with fzip.open(file, "r") as f:
                data = pickle.load(f)
                output_data = {key: data[key] for key in KEYS_TO_SAVE}
                output_file_path = os.path.join(NEW_DIR_PATH, f"{file}.pkl")
                with open(output_file_path, "wb") as output_file:
                    pickle.dump(output_data, output_file)
    except zipfile.BadZipFile:
        print(f"BadZipFile: {file}")
    except pickle.UnpicklingError:
        print(f"UnpicklingError: {file}")
    except ValueError:
        print(f"ValueError: {file}")


def get_simulation_parameters():
    with zipfile.ZipFile(QDATA_DIR_PATH, mode="r") as fzip:
        with fzip.open(f"G_1q_XY_XZ_{NOISE_PROFILE}_D_ex_0", "r") as f:
            data = pickle.load(f)
            simulation_parameters = str(data["sim_parameters"])
            with open("simulation_parameters.txt", "w") as f:
                f.write(simulation_parameters)


def extract_needed_data():
    if not os.path.exists(NEW_DIR_PATH):
        os.makedirs(NEW_DIR_PATH)

    with zipfile.ZipFile(QDATA_DIR_PATH, mode="r") as fzip:
        original_file_names = fzip.namelist()

    files_in_new_dir = os.listdir(NEW_DIR_PATH)

    files_to_process = [
        file_name
        for file_name in original_file_names
        if file_name + ".pkl" not in files_in_new_dir
    ]

    print(f"Number of files to process: {len(files_to_process)}")
    print(f"Number of cores: {os.cpu_count()}")

    with mp.Pool(os.cpu_count() - 2) as pool:
        pool.map(process_file, files_to_process)


if __name__ == "__main__":
    get_simulation_parameters()
    extract_needed_data()
