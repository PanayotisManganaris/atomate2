"""
Backfill a jobstore database with legacy VASP runs

implemented using atomate, jobflow and pymatgen

parallelizes and automates:
1. collect all calculations under current directory
2. fill atomate output record schema
   - workflow specifications are not created by this script
   - backfilling with this utility will not enable dupe checking 
3. upload all experiments to database
"""
import os
import traceback
import time
import monty
from pathlib import Path

from atomate2.vasp.drones import VaspDrone
from jobflow import JobStore, SETTINGS
from jobflow.utils.uuid import suuid
from datetime import datetime

from itertools import chain, zip_longest
import multiprocessing as mp

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
store = SETTINGS.JOB_STORE

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

def filter_paths(paths:Iterable[str],
                 filterlist)->Iterable[str]:
    """
    utility to manually narrow paths to those which don't contain
    strings member to filterlist
    """
    paths = [s for s in paths if not
             any(filterentry in s for filterentry in filterlist)]
    return paths

def gworker(subgroup, eq):
    """
    middleman for multiprocessing

    place diagnostics/status checks here
    """
    for job in subgroup:
        worker(job, eq)

def worker(path, eq):
    """
    use drone to assimilate a simulation path into a task document

    subsequently process document

    gets swallowed up in multiprocessing process
    """
    try:
        with monty.os.cd(path):
            doc = drone.assimilate()
        data = reshape(doc) #insert metadata here if needed
        update_store(store=store, data=data)
        #record_strings = make_training_data(doc, fdir)
        #q.put("\n".join(record_strings))
    except Exception as e:
        eq.put(traceback.format_exc())

def reshape(taskdoc, metadata={})->dict:
    """
    reshape retroactively parsed task documents for uniformity with
    the atomate schema
    """
    data = {
        "uuid": suuid(),
        "index": 'backfilled',
        "output": taskdoc,
        "completed_at": datetime.now().isoformat(),
        "metadata": metadata,
        # "hosts": self.hosts or [], #future jobflow
    }
    return data

def update_store(store:JobStore, data) -> None:
    """
    input store and document, connect to store, upload document
    can be parallelized
    """
    with store as s:
        s.update(data, key="uuid") 

def grouper(it:Iterable, n, fillvalue:Any=()):
    """ extract n-size chunks of iterable """
    args = [iter(it)] * n
    return zip_longest(*args, fillvalue=fillvalue)

def main_parser(paths, l, p):
    s = time.perf_counter()
    manager = mp.Manager()
    eq = manager.Queue()

    for group in grouper(grouper(paths, l), p):
        ps=[]
        for g in group:
            p = mp.Process(target=gworker, args=(g, eq)) 
            ps.append(p)
            p.start()
        for p in ps:
            p.join()

    with open("./err.txt", 'a') as f:
        while not eq.empty():
            f.write(str(eq.get()) + '\n')
        f.flush()

    d = time.perf_counter() - s
    return d

if __name__ == "__main__":
    from tqdm import tqdm
    #exp_dir = '.' #invoke script from experiment directory
    exp_dir = '/depot/amannodi/data/MCHP_Database/MAPbI_3/PBE_relax'
    # exp_dir = '/depot/amannodi/data/MCHP_Database/'
    #exp_dir = "/depot/amannodi/data/Perovs_phases_functionals/Larger_supercell_dataset/PBE_relax"
    #data_dir = '/depot/amannodi/data/perovskite_structures_training_set'
    # data_dir = '/depot/amannodi/data/pbe_perovskite_structures'
    # csv = "id_prop_master.csv"
    # err = "duds.log"
    fl=['LEPSILON','LOPTICS','Phonon_band_structure',
        'V_A', 'V_X', 'Band_structure']
    # LEPSILON doesn't have bands?  # get_element_spd_dos(el)[band] keyerror
    # LOPTICS doesn't have VASPrun pdos attribute
    # PH disp doesn't have electronic bands # get_element_spd_dos(el)[band] keyerror
    paths_gen = get_vasp_paths(exp_dir)
    paths = filter_paths(paths_gen, filterlist=fl)
    pbar = tqdm(paths, desc="Processing")
    d = main_parser(pbar, 1, 1)
    print(d)
