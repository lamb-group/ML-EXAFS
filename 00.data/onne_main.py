# General Python Libraries 
import os, shutil, subprocess, argparse, logging, numpy as np
from itertools import repeat; from glob import glob; from multiprocessing import Pool

# ONNE Specific libraries
from utils import create_slurm_scripts, add_absorber, write_feff_dir_from_xyz

# This program takes one argument which is the number of cpus
# to be used in parallelization of the generation process
parser = argparse.ArgumentParser()
parser.add_argument("--nprocs", type=int, default = 1)
parser.add_argument("--workdir", type=str, default = "./generated_configs")
parser.add_argument("--scripts", type=int, default = 4)
args = parser.parse_known_args()


# Define and create a working directory to store all macorstates
nprocs, working_dir, feff_scripts = vars(args).values()

os.makedirs(working_dir,exist_ok=True)

# Create a log file for debugging and program tracking
logging.basicConfig(
    filename="data_gen.log",
    filemode="w",        # overwrite log each run, or use "a" to append
    encoding="utf-8",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True
)

logging.info("Program Started")
logging.info(f"{args.nprocs} CPUs detected")
logging.getLogger().handlers[0].flush()

#---------------------------------#
# 3rd party software input templates
#---------------------------------#

# Packmol template: {ID} is the unique structure identifier
# {block} is the text for the atomic positions
# Adjust tolerance for the desired minimum interatomic distance
# of non-absorber atoms

packmol_template = """#
# 
#

tolerance 2.5
filetype xyz 
output structure_{ID}.xyz

{block}
"""

# Block for each atom inside the packmol file
# {rho} is the sampled absorber-neighbor distance

block_template = '''
structure {element}.xyz
   number 1
   inside sphere  0.0 0.0 0.0 {rho}
end structure
'''

# FEFF input file template (tailor to context and needs)
feff_template = """TITLE	{title}
EDGE	{edge}
S02	1.0
CONTROL	1 1	1 1	1 1

SCF 6 1 100 0.2 10
COREHOLE FSR
REAL

EXAFS 14
RPATH 6.0

POTENTIALS
{potentials}
ATOMS
{atoms}
END"""

logging.info("Beginning Sampling for Absorber-Neighbor Distances")

#---------------------------------#
# Absorber-Neighbor Sampling
#---------------------------------#

# Define a random number generator
SEED = 1202; rng = np.random.default_rng(SEED)
n_frames = 800 # Number of microstates per macrostate

# Define a set of means for each Absorber-Neighbor gaussian (Macrostate)
Zr_F = np.linspace(1.9, 2.3, 6, endpoint=True)
Zr_Na = np.linspace(3.3, 4.0, 6, endpoint=True)
Zr_Zr = np.linspace(3.8, 4.8, 6, endpoint=True)

# To increase randomness, shuffle the means
rng.shuffle(); rng.shuffle(Zr_Na); rng.shuffle(Zr_Zr)

# Build macrostates as a tuple of 1st, 2nd and 3rd shell means
macrostates = list(zip(Zr_F, Zr_Na, Zr_Zr))

# Define a list of Zr-F Coordination numbers to cover
# an expanded microstate search space
Zr_F_1st_shell_CN = [4,5,6,7,8,9] 

packmol_inputs = [] # Initialize an empty list to store ALL packmol inputs

# Setting up packmol input files via sampling absorber-neighbor
# distances for each CN
for CN in Zr_F_1st_shell_CN:
    logging.info(f"Sampling for 1st Shell Zr-F CN: {CN}")
    logging.getLogger().handlers[0].flush()
    # 2nd shell species, ratio obtained from literature
    n_Zr = int(0.3 * 10)
    n_Na = int(10 - n_Zr)

    # 3rd shell number, balancing the chemical composition
    # Total F = Na + 4*Zr for NaF-ZrF4
    n_F =  n_Na + 4* n_Zr - CN

    for macrostate, means in enumerate(macrostate):
        mu1, mu2, mu3 = means
        for frame in range(n_frames):

            # 1st shell sampling 
            dist_F = rng.normal(loc = mu1, scale = 0.05, size = CN)
            # Building packmol file
            shell1 = [block_template.format(element='F',rho=rho) for rho in dist_F]
            
        
            # 2nd shell sampling
            dist_Zr = rng.normal(loc = mu2, scale=0.2,size=n_Zr)
            dist_Na = rng.normal(loc = mu3, scale=0.3,size=n_Na)

            shell2_1 = [block_template.format(element='Na',rho=rho) for rho in dist_Na]
            shell2_2 = [block_template.format(element='Zr',rho=rho) for rho in dist_Zr]

            # 3rd shell sampling
            dist_F = rng.normal(loc=4.5,scale=0.67, size=n_F)
            shell3 = [block_template.format(element='F',rho=rho) for rho in dist_F]

            # Concatenating all atom blocks
            all_blocks = shell1 + shell2_1 + shell2_2 + shell3

            # Building packmol input
            final_block = ''
            for block in all_blocks:
                final_block += block
                    
            # Edit packmol input template and append the packmol inputs list
            # with a tuple of (CN, macrostate #, input file)
            file = packmol_template.format(ID = str(frame) ,block=final_block)
            packmol_inputs.append((CN,macrostate,file))

logging.info(f"{len(packmol_inputs)} Files prepared for packmol ")
logging.info("Beginning Structure Construction")
logging.getLogger().handlers[0].flush()

#---------------------------------#
# Microstate Assembly w/ Packmol
#---------------------------------#

# Get the current directory
cwd = os.getcwd() 

# Command to run packmol from CLI
# Must add packmol binaries to PATH for this to work
# alternatively replace packmol with the path to packmol binary
# "~/software/packmol_25_1/packmol < packmol.inp"
command = "packmol < packmol.inp"


# Here we define a worker function for structure generation such that
# packmol could assemble microstates simulatenously using multiprocessing
# This speeds up the structure generation greatly

def generate_structure(task_data):

    '''
        Worker function for multiprocessing tool to use for ONNE
        structure generation

        :param task_data: A Tuple that contains the task ID and packmol input data (task_id, packmol_input)
    '''
    # Unpack Task ID and input file
    task_id, packmol_input = task_data
    CN, macrostate, packmol_file = packmol_input

    # Create working directory for each task
    task_working_dir = os.path.join(working_dir, f"macrostate_{macrostate}",
                                    f"CN_{CN}",
                                    f"frame_{task_id}")
    os.makedirs(task_working_dir, exist_ok=True)

    # Copy atom files needed for packmol in the task directory
    for file in glob(os.path.join(cwd,"*.xyz")):
        try:
            shutil.copy(file,task_working_dir)
        except Exception as e:
            print(f"Error moving {file}: {e}")

    # Go to task directory
    os.chdir(task_working_dir)

    # Write a packmol input file in task directory
    with open('packmol.inp','w') as f:
        f.write(packmol_file)
        f.close()
    
    # Assemble Microstate
    try:
        # Use subprocess to execute the run packmol command
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        # Print the command's output
        logging.info(f"Structure {task_id} built successfully")
        print(result.stdout)

        # Check for errors
        if result.stderr:
            logging.info(f"Structure {task_id} encountered an error: {e}")
            print(result.stderr)

    except Exception as e:
        logging.info(f"Packmol Failed")

    finally:
        # Go back to original directory
        os.chdir(cwd)
    logging.getLogger().handlers[0].flush()


# Main function needed by multiprocessing library
if __name__ == "__main__":

    # Prepare a list of tuples with (ID,Packmol_input) for worker function
    tasks = [(i,packmol_input) for i, packmol_input in enumerate(packmol_inputs)]

    # Use multiprocessing to run through all packmol inputs and post-processing
    with Pool(processes = nprocs) as pool:
        # Microstate assembly
        pool.map(generate_structure, tasks)
        # Go back to current directory (For redundancy)
        os.chdir(cwd)

        logging.info("Finished Construction")


        # Packmol creates the structures without the central absorber
        # Here we add the central absorber using multiprocessing, calling
        # the worker function add_absorber

        logging.info("Post Processing - Adding Central Atoms")
        logging.getLogger().handlers[0].flush()

        # Search for all structure files
        struct_files = glob(os.path.join(working_dir,"frame_*/structure_*.xyz"))
        logging.info(f"Found {len(struct_files)} structure files")

        # Output a sample structure file
        logging.info("Files sample:")
        idx = rng.integers(0,len(struct_files), endpoint=True)
        logging.info(struct_files[idx])
        
        logging.getLogger().handlers[0].flush()

        pool.map(add_absorber, struct_files)

        # For each microstate we create a feff input file based on the
        # structure files

        logging.info("Post Processing - Creating FEFF input files")
        logging.getLogger().handlers[0].flush()
    
        # Creating FEFF input files for each generated structure
        # leveraging multiprocessing for speed with the worker function
        # write_feff_dir_from_xyz

        struct_folders = glob(os.path.join(working_dir,"frame_*"))
        xyz_to_feff_args = zip(struct_files,
                            struct_folders,
                            repeat(0),
                            repeat('K'),
                            repeat('NaF_ZrF4_onne'),
                            repeat(None),
                            repeat(None),
                            repeat(feff_template))

        pool.starmap(write_feff_dir_from_xyz, xyz_to_feff_args)

    # Lastly we create SLURM scripts to run FEFF on HPC (Omit if needed)
    logging.info("Post Processing - Creating FEFF Slurm Scripts")
    create_slurm_scripts(struct_folders, "run", cwd, num_blocks = feff_scripts)
    logging.info("Programm Finished")