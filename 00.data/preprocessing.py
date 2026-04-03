import os, pickle, argparse
from tqdm import tqdm
from glob import glob

from utils import process_config

#----------------------------------#
# Main
#----------------------------------#

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir",type = str, default=None)
    parser.add_argument("--name",type = str, default="ONNE")
    parser.add_argument("--output",type = str, default="./ml_data")
    args, _ = parser.parse_known_args()

    directory, name, output_dir = args.s, args.n, args.d
    config_folders = sorted(glob(os.path.join(directory,"*/")))
    print(f"Found {len(config_folders)} frame folders in {directory}")

    ml_data = []
    with tqdm(total=len(config_folders)) as pbar:
        for folder in config_folders:
            feature, label = process_config(folder)
            ml_data.append((feature,label))
            pbar.update(1)

    file_path = os.path.join(output_dir,f"{name}_ml_data.pkl")
    pickle.dump(ml_data,open(file_path,"wb"))

if __name__ == "__main__":
    main()