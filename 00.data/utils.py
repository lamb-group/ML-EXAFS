import os, shutil,  numpy as np
from ase import Atoms
from ase.io import read, write
from scipy.interpolate import interp1d
from ovito.data import CutoffNeighborFinder,DataCollection, Particles

# Default FEFF Template if not provided as argument
feff_template = """TITLE	{title}
EDGE	{edge}
S02	0.9
CONTROL	1	1	1	1	1	1
PRINT	0	0	0	0	0	0
EXCHANGE	0	1.	0.
SCF	5.5	0	100	0.1	1
COREHOLE	RPA
XANES	5	0.05	0.1
FMS	7	0
POTENTIALS
{potentials}
ATOMS
{atoms}
END"""

# Tailor depending on the HPC you have
slurm_template = """
#!/bin/bash\n
#SBATCH --nodes=4\n
#SBATCH --job-name=feff\n
#SBATCH --ntasks=28\n
#SBATCH --cpus-per-task=1\n
#SBATCH --mem=28G\n
#SBATCH --time=12:00:00\n
#SBATCH --output=feff-%j.out\n
#SBATCH --error=feff-%j.err\n
#SBATCH --partition=cpu\n
\n
export OMP_NUM_THREADS=1\n
\n
"""

#----------------------------------#
# Utility Functions for FEFF files
#----------------------------------#

def make_potential_atoms_from_xyz(xyz, absorber=0):

    config = read(xyz)
    elements = np.array(config.get_chemical_symbols())
    Z = config.get_atomic_numbers()
    coordinates = config.get_positions()

    print("elements: ", elements)
    print("coodinates: ", coordinates)

    # Potentials
    unique_elements, unique_Z = np.unique(elements),np.unique(Z)
    unique_counts = np.array(
        [np.sum(elements == element) for element in unique_elements]
    )
    print("unique_elements: ", unique_elements)
    print("unique_counts: ", unique_counts)

    absorber_Z = Z[absorber]
    print("elements[absorber]: ", elements[absorber])

    # Building POTENTIAL CARD
    potentials = []
    potential_dict = {}

    # Absorber
    potentials.append(f"0 {absorber_Z} {elements[absorber]} -1 -1 0.001") 

    for counter, (element, count, z) in enumerate(zip(unique_elements, unique_counts, unique_Z)):
        # Skip absorber if it there is only one atom in the cluster
        if (element == elements[absorber]) and (count <= 1): 
            continue

        potentials.append(f"{counter+ 1} {z} {element} -1 -1 {count}")
        potential_dict[element] = counter + 1

    print("potentials: ", potentials)
    potentials = "\n".join(potentials)

    # ATOMS Card
    atoms = []
    for i, (element, coordinate) in enumerate(zip(elements, coordinates)):
        if i == absorber:
            print(i)
            atoms.append(f"{coordinate[0]} {coordinate[1]} {coordinate[2]} 0")
            continue
        atoms.append(
            f"{coordinate[0]} {coordinate[1]} {coordinate[2]} {potential_dict[element]}"
        )

    atoms = "\n".join(atoms)
    return potentials, atoms


def write_feff_dir(feff_inp, directory):
    os.makedirs(directory, exist_ok=True)
    feff_file = os.path.join(directory,"feff.inp")
    with open(feff_file, "w") as f:
        f.write(feff_inp)
    
def write_feff_dir_from_xyz(
    xyz, directory, absorber=0, edge="K", title="test", xmu_path=None, feff_inp_path=None, feff_template=feff_template
):
    """
    Creates a FEFF file from a .xyz structure file

    :param xyz: structure file in xyz format
    :param directory: Directory to save the feff.inp
    :param absorber: absorber index wrt xyz file
    :param edge: Absorption Edge (K,M4,L3 etc)
    :param title: File title
    :param xmu_path (Optional): different directory for results
    :param feff_path (Optional): different directory for feff.inp
    :param feff_template: input parameters for FEFF
    """

    potentials, atoms = make_potential_atoms_from_xyz(xyz, absorber=absorber)
    feff_inp = feff_template.format(
        title=title, edge=edge, potentials=potentials, atoms=atoms
    )
    write_feff_dir(feff_inp, directory)

    if xmu_path is not None:
        os.makedirs(os.path.dirname(xmu_path), exist_ok=True)
        shutil.copy(directory + "xmu.dat", xmu_path)
        
    if feff_inp_path is not None:
        os.makedirs(os.path.dirname(feff_inp_path), exist_ok=True)
        shutil.copy(directory + "feff.inp", feff_inp_path)

def create_slurm_scripts(dir_list, script_prefix, file_path, num_blocks = 4):
    """
    Create SLURM batch scripts from a list of directories.

    :param dir_list: List of directories that contain feff.inp files
    :param script_prefix: Prefix for the output SLURM script files
    :param dir_list: path to save the SLURM scripts
    :param num_blocks: How many SLURM scripts
    """
    step_size = len(dir_list)//num_blocks
    for block_num, i in enumerate(range(0, len(dir_list), step_size), start = 1):
        block = dir_list[i:i + step_size]
        script_name = os.path.join(file_path,f"{script_prefix}_feff_{block_num}.sh")

        with open(script_name, 'w') as file:
            file.write(slurm_template)
            for index,dir in enumerate(block):
                if index != 0 and index % 112 == 0:
                    file.write("wait\n")
                dir_modified = dir.replace(file_path, './')
                file.write(f"srun --exclusive -N 1 -n 1 --cpus-per-task=1 --mem=1G --time=00:10:00 --chdir={dir_modified} feff &\n")
            file.write("wait\n")
            
    print(f"Created {block_num} SLURM script(s) with prefix '{script_prefix}' in directory {file_path}.")

#----------------------------------#
# Utility Functions for ML-Preprocessing
#----------------------------------#

rmesh = np.linspace(0.06,6.00,100)
kspace = np.linspace(2,13.0,220)

def read_exafs(file_path):
    '''
    This function reads FEFF output file chi.dat and returns (2,N)
    array that contains k and chi
    file_path: Path to chi.dat file
    '''
    data = np.loadtxt(file_path, skiprows=1)[:,[0,1]]
    return data
    
def read_feff(file_path):
    '''
    This function reads and returns FEFF input file
    file_path: Path to feff.inp file
    '''
    with open(file_path,"r") as f:
        lines = f.readlines(); f.close()
    lines = [line.replace("\n", "").replace("\t", " ") for line in lines]
    return lines

def make_rdf(distances,rmesh):
    '''
    This function constructs a radial distribution function based on the definition
    g(r) = N/(4 x pi x dr x r^2)

    distances: atomic distances from an absorbing atom
    rmesh: Radial distance grid
    '''
    dr = rmesh[1]-rmesh[0]
    digitized = np.digitize(distances, rmesh) - 1
    gr = np.bincount(digitized[digitized>=0],minlength=len(rmesh))
    gr = gr/(4 * np.pi * dr * rmesh**2)
    gr[0] = 0
    return 10*gr

def compute_coordination_number(gr, rmesh, rrange):
    '''
    This function returns the coordination number by integrating
    g(r) = N/(4 x pi x dr x r^2) within a range

    gr: Radial Distribution Function
    rmesh: Radial distance grid
    rrange: Integration range
    '''
    gr = gr/10
    mask = (rmesh >= rrange[0]) & (rmesh <= rrange[1])
    r_selected = rmesh[mask]
    g_r_selected = gr[mask]
    dr = rmesh[1]-rmesh[0]
    # Compute coordination number using numerical integration
    CN = 4 * np.pi * np.trapezoid(g_r_selected * r_selected**2, r_selected,dr)
    return CN

def interpol(exafs, kspace):
    '''
    This function maps any EXAFS data to a new k-space mesh

    exafs: Input EXAFS in the shape (2,N) containing k,chi
    kspace: New mesh to map to
    '''
    x, y = exafs.transpose()
    f1 = interp1d(x, y, kind='cubic')
    test_real = f1(kspace)
    return test_real

#----------------------------------#
# Utility Functions for ONNE structures
#----------------------------------#


def process_config(config_folder):
    '''
    This function processes the feff output for a configuration and 
    returns its ensemble average. The configuration can be an ONNE macrostate
    or a Molecular Dynamics frame

    '''
    config_rdfs, config_exafs = [],[]

    # Collect absorber/microstate folders
    absorber_folders = [config_folder]#sorted(glob(os.path.join(config_folder,"*")))
    
    for folder in absorber_folders:
        
        # EXAFS Collection
        chi_file = os.path.join(folder,"chi.dat")
        try:
            chi_data = read_exafs(chi_file)
        except Exception as e:
            continue
            
        config_exafs.append(chi_data)
        
        # Coordinates Collection
        feff_data = read_feff(chi_file.replace("chi.dat","feff.inp"))
        try:
            coord = feff_data[feff_data.index("ATOMS") + 1:-1]
        except Exception as e:
            print(e, folder)
            continue
        coord = np.array([line.split()[:4] for line in coord], dtype=float)
        coords, ptypes = coord[:,:3], coord[:,3].astype(np.int32())
    
        data, particles = DataCollection(), Particles()
        particles.create_property('Position', data = coords)
        particles.create_property('Type', data = ptypes)
        data.objects.append(particles)

        finder = CutoffNeighborFinder(6, data)
        finder.pbc = True

        # RDF Construction
        F_dist, Zr_dist, Na_dist = [],[],[]
    
        for neigh in finder.find(0):
            idx = neigh.index
            dist = neigh.distance
            if ptypes[idx] == 3:
                Zr_dist.append(dist)
            elif ptypes[idx] == 2:
                Na_dist.append(dist)
            elif ptypes[idx] == 1:
                F_dist.append(dist)
                
        F_dist, Zr_dist, Na_dist = np.array(F_dist), np.array(Zr_dist), np.array(Na_dist)
        config_rdfs.append(np.array([make_rdf(F_dist,rmesh),
                                     make_rdf(Na_dist,rmesh), 
                                     make_rdf(Zr_dist,rmesh)]))
    
    # Return Ensemble average k2 weighted chi (feature) and augmented RDF (label)
    config_rdfs, config_exafs= np.array(config_rdfs), np.array(config_exafs)
    avg_rdfs, avg_exafs = np.mean(config_rdfs, axis = 0),np.mean(config_exafs, axis = 0)
    chik2 = interpol(avg_exafs, kspace)*kspace**2

    return chik2, np.hstack(avg_rdfs)

def add_absorber(file):
    config = read(file, format = "xyz")
    config = Atoms('Zr', positions = [[0, 0, 0]]) + config
    write(file,config)