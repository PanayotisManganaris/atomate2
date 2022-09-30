"""
VASP run parsing script

implemented using atomate and jobflow

Objective: backfill database with legacy experiments
   - collect all calculations
   - fill atomate schema
     - difference between workflow schema and output schema?
   - upload to database
"""

from atomate2.vasp.drones import VaspDrone
from pymatgen.core import Structure, Composition
from jobflow import JobStore, SETTINGS

import os
import monty
from pathlib import Path

from tqdm import tqdm
from itertools import chain

from typing import Any, Union
from collections.abc import Iterable

task_document_kwargs = {
    'additional_fields':{# add fields for submission to database
        
    },
    #'vasp_calculation_kwargs':{
    # control which parts of a calculation are assimilated
    'parse_dos': True,
    'parse_bandstructure': True,
    'average_locpot': True,
    # control minutia of which files are parsed and how
    'vasprun_kwargs': {
        'parse_potcar_file':False,
        'parse_dos':True,
        'parse_eigen':True,
        'parse_projected_eigen':False,
    }
}

drone = VaspDrone(**task_document_kwargs)

#TODO: compute real targets, bandgap and decoE
#TODO: also get total energy
#TODO: build training set directory
#TODO: split necessary functions off to separate script in that directory (using queries)
#TODO: improve key gen
#TODO: use existing decoE computations to derive decoE of intermediates
#TODO: move all environments to depot/apps (quick one)
#TODO: make metadata function to call on directories for use with update_store

store = SETTINGS.JOB_STORE
#look into defining save/load mapping to direct document items to
#additional stores. currently, items that are too big are
#automatically redirected to an alternative by maggma/pymongo

exp_dir = '.' #invoke script from experiment directory

def get_vasp_paths(parent:Union[str,Path]) -> Iterable[str]:
    """
    Use drone to find viable VASP experiment directories
    
    valid paths include those that lead to:
    - unique vasprun.xml (nesting directories ok)
    - relax<###> directory (multi-optimization runs)
    - vasprun.relax<###> (multi-opt)
    """
    vaspaths = chain.from_iterable(
        [drone.get_valid_paths(x) for x in os.walk(parent) if
         drone.get_valid_paths(x)]
    )
    return vaspaths

def filter_vaspaths(vaspaths:Iterable[str],
                    filterlist)->Iterable[str]:
    """
    utility to manually narrow vaspaths to those which don't contain
    strings member to filterlist
    """
    vaspaths = [s for s in vaspaths if not
                any(filterentry in s for filterentry in filterlist)]
    return vaspaths

def assimilate_paths(vaspaths:Iterable[Union[str,Path]],
                     filterlist=[]) -> list:
    """
    Use drone to create list of taskdocuments corresponding to list of
    experiment directories.
    """
    docs = []
    pbar = tqdm(filter_vaspaths(vaspaths, filterlist),
                desc="Processing Path()")
    for vaspath in pbar:
        pbar.set_description(f'Processing Path("{vaspath}")')
        try:
            with monty.os.cd(vaspath):
                docs.append(drone.assimilate())
        except Exception as e:
            print(e)
    return docs

def update_store(store:JobStore, docs:list) -> None:
    """
    input store and documents, connect to store, upload documents
    """
    with store as s:
        s.update(docs, key="output")

### Additional functions to create Graph Network Training Directories

def make_record_name(doc, calc, step)->str:
    """
    return string to uniquely identify a structure file and id_prop
    record from query info.

    unique id made of:
    formula + LoT + step
    """
    formula=doc.dict()['formula_pretty']
    LoT=calc.dict()['run_type']
    record_name = f"{formula}_{LoT}_{step}"
    return record_name

def write_properties_file(record:str, props:list,
                          fdir:Union[str,Path]='.',
                          csv:str='id_prop.csv') -> None:
    """
    write a cgcnn-compliant training target file
    """
    csv_path=os.path.join(fdir, csv)
    props=','.join(map(str,props))
    with open(csv_path, 'a') as f:
        f.write(f"{record},{props}\n")

def structure_to_training_set_entry(struct:Structure,
                                    record:str,
                                    props:list,
                                    fdir:Union[str,Path],
                                    csv:str='id_prop.csv') -> None:
    """
    write a structure to a POSCAR named record in directory fdir
    
    write structure properties to properties file in directory fdir
    """
    filename=os.path.join(fdir, record)
    struct.to(fmt='POSCAR', filename=filename)
    write_properties_file(record, props, fdir, csv)

def main() -> None:
    """silly temporary main function while waiting for Geddes resource"""
    #data_dir = '/depot/amannodi/data/perovskite_structures_training_set'
    data_dir = '/home/panos/MannodiGroup/DFT/parse_test'
    Bel = ['Ca','Sr','Ba','Ge','Sn','Pb']

    write_properties_file(record="id", props=["totE,decoE,bg"],
                          fdir=data_dir, csv="id_prop_master.csv")

    docs = assimilate_paths(get_vasp_paths(exp_dir),
                            filterlist=['LEPSILON','LOPTICS','Phonon_band_structure'])
    # LEPSILON doesn't have bands?  # get_element_spd_dos(el)[band] keyerror
    # LOPTICS doesn't have VASPrun pdos attribute
    # PH disp doesn't have electronic bands # get_element_spd_dos(el)[band] keyerror
    for doc in docs:
        # doc.dict().keys()
        # ['nsites', 'elements', 'nelements', 'composition', 'composition_reduced', 'formula_pretty','formula_anonymous'
        # , 'chemsys', 'volume', 'density', 'density_atomic', 'symmetry', 'dir_name', 'last_updated',
        # 'completed_at', 'input', 'output', 'structure', 'state', 'included_objects', 'vasp_objects', 'entry',
        # 'analysis', 'run_stats', 'orig_inputs', 'task_label', 'tags', 'author', 'icsd_id', 'calcs_reversed',
        # 'transformations', 'custodian', 'additional_json'])
        for calc in doc.calcs_reversed:
            # calc.dict().keys()
            # ['dir_name', 'vasp_version', 'has_vasp_completed', 'input', 'output', 'completed_at', 'task_name',
            # 'output_file_paths', 'bader', 'run_type', 'task_type', 'calc_type']
            # calc.dict()['output'].keys()
            # ['energy', 'energy_per_atom', 'structure', 'efermi', 'is_metal', 'bandgap', 'cbm', 'vbm', 'is_gap_direct',
            # 'direct_gap', 'transition', 'mag_density', 'epsilon_static', 'epsilon_static_wolfe', 'epsilon_ionic',
            # 'frequency_dependent_dielectric', 'ionic_steps', 'locpot', 'outcar', 'force_constants',
            # 'normalmode_frequencies', 'normalmode_eigenvals', 'normalmode_eigenvecs', 'elph_displaced_structures',
            # 'dos_properties', 'run_stats']
            # calc.dict()['input'].keys()
            # ['incar', 'kpoints', 'nkpoints', 'potcar', 'potcar_spec', 'potcar_type', 'parameters', 'lattice_rec',
            # 'structure', 'is_hubbard', 'hubbards']
            struct = calc.dict()['input']['structure'] #POSCAR
            prime_struct = struct.get_primitive_structure(tolerance=0.25)
            formula = Composition(struct.formula)
            prime_formula = Composition(prime_struct.formula)
            
            formula_unit, formula_units_per_super_cell = formula.get_reduced_formula_and_factor()
            _, formula_units_per_unit_cell = prime_formula.get_reduced_formula_and_factor()
            cell_count = formula_units_per_super_cell/formula_units_per_unit_cell

            #cell_count = sum([Bnum for B,Bnum in formula.as_dict().items() if B in Bel])
            toten_pfu = calc.dict()['output']['energy']/cell_count
            # decoE = decomp_energy(formula_dict, toten_pfu) #from cmcl
            decoE = -1
            bg = calc.dict()['output']['bandgap']
            # predictions on POSCARs should predict CONTCAR energies
            record_name = make_record_name(doc, calc, "POSCAR")
            structure_to_training_set_entry(struct,
                                            record_name,
                                            props=[float(toten_pfu), decoE, bg],
                                            fdir=data_dir,
                                            csv='id_prop_master.csv')
            
            for count, step in enumerate(calc.dict()['output']['ionic_steps']):
                # print(step.keys())
                # ['e_fr_energy', 'e_wo_entrp', 'e_0_energy', 'forces', 'stress', 'electronic_steps', 'structure']
                struct = step['structure'] #XDATCAR iteration
                # compute decomposition energy at each interval
                formula_dict = Composition(struct.formula).as_dict()
                count_cells = sum([Bnum for B,Bnum in formula_dict.items() if B in Bel])
                toten_pfu = step['e_fr_energy']/count_cells
                # decoE = decomp_energy(formula_dict, toten_pfu) #from cmcl
                decoE = -1
                bg = "" # bg cannot be saved from intermediates in
                        # vasp runs not configured to return them at
                        # each step

                # compare structure steps to POSCARs before saving?
                # match_kwargs = dict(ltol=0.2,stol=0.3, angle_tol=5,
                #                     primitive_cell=True, scale=True,
                #                     attempt_supercell=False,
                #                     allow_subset=False,
                #                     supercell_size=True)
                # if struct.matches(POSCAR, **match_kwargs):

                record_name = make_record_name(doc, calc, count+1)
                structure_to_training_set_entry(struct,
                                                record_name,
                                                props=[toten_pfu, decoE, bg],
                                                fdir=data_dir,
                                                csv='id_prop_master.csv')

if __name__ == "__main__":
    main()
    # docs = assimilate_paths(get_vasp_paths(exp_dir),
    #                         filterlist=['LEPSILON','LOPTICS','Phonon_band_structure'])
    # update_store(store, docs)
