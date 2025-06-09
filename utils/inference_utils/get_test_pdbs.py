'''
script that reads the csv created by the DataCleaner class, grabs the test set clusters,
and parses the dataframe to get the pdbIDs, biounits, and representative chains.
Then the idea is to download the pdbs to specified dir. doing this so can run PMPNN
on this test set and compare to proteusAI. metrics measures will be cel, top1, top3, top5
accuracy. possibly include boltz1 condifence scores later. this cannot be run on the compute nodes,
because need internet access to download pdbs
'''

from concurrent.futures import ThreadPoolExecutor, as_completed
from Bio.PDB import PDBParser, Structure, PDBIO, Select
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import argparse
import requests
import torch
import gzip 
import io

def parse_test_set(data_path, raw_pt_path, max_seq_size, max_resolution, out_path):

    # read the csv
    pdb_info = pd.read_csv(data_path / Path("list.csv"), header=0, engine='python')

    # clusters formatted differently in multi and single chain, so convert them to string for consistency
    pdb_info.CLUSTER = pdb_info.apply(lambda row: str(row.CLUSTER), axis=1)

    # get list of test clusters and filter
    with open(data_path / Path("test_clusters.txt"),  "r") as t:
        test_clusters = [str(i) for i in t.read().split("\n") if i]
    test_pdbs = pdb_info.loc[pdb_info.CLUSTER.isin(test_clusters), :]

    # get indices of each chains biounit sizes that are less than max_seq_size
    test_pdbs["VALID_IDX"] = test_pdbs.loc[:, "BIOUNIT_SIZE"].apply(lambda x: [i for i, size in enumerate(str(x).split(";")) if int(size) <= max_seq_size])

    # remove the indices
    test_pdbs.loc[:,"BIOUNIT"] = test_pdbs.apply(lambda row: [str(row.BIOUNIT).split(";")[idx] for idx in row.VALID_IDX], axis=1)
    test_pdbs.loc[:,"BIOUNIT_SIZE"] = test_pdbs.apply(lambda row: [str(row.BIOUNIT_SIZE).split(";")[idx] for idx in row.VALID_IDX], axis=1)

    # remove any chains who do not have a biounit after the length filter, and remove the VALID IDX column
    test_pdbs = test_pdbs.loc[test_pdbs.BIOUNIT.apply(lambda x: len(x)>0), [col for col in test_pdbs.columns if col != "VALID_IDX"]].reset_index(drop=True)

    # remove pdbs with large resolution
    test_pdbs = test_pdbs.loc[test_pdbs.RESOLUTION <= max_resolution, :]

    # list of unique pdbs that will need to download
    pdbs = test_pdbs.PDB.unique()[:5]

    # biounits is a dict with pdbs benig the keys and corresponding chains the values
    biounits = {}
    for pdb in pdbs:
        # load the raw pt file containing the info about which chains make up the biounits
        pt = raw_pt_path / Path(f"{pdb[1:3]}/{pdb}.pt")
        pt_info = torch.load(pt, map_location="cpu", weights_only=True)
        biounit_chains = pt_info["asmb_chains"]
        biounits[pdb] = biounit_chains

    pdb_path = out_path / Path("pdb")
    pdb_path.mkdir(exist_ok=True, parents=True)

    download_pdb(list(biounits.items())[0], pdb_path)
    # pbar = tqdm(total=len(pdbs), desc="downloading_pdbs", unit="pdbs")
    # with ThreadPoolExecutor(max_workers=1) as executor:
    #     futures = {executor.submit(download_pdb, biounit, pdb_path): biounit for biounit in biounits.items()}                
    #     for future in as_completed(futures):
    #         pbar.update(1)

def download_pdb(pdb_biounits, out_path):

    # pdbid and the list of chains
    pdb, biounits = pdb_biounits

    # send the request to stream the gz of the pdb
    url = f"https://files.wwpdb.org/pub/pdb/data/structures/divided/pdb/{pdb[1:3]}/pdb{pdb}.ent.gz"
    resp = requests.get(url, stream=True, timeout=30)
    resp.raise_for_status()

    # stream, unpack, and store in text-liike wrapper for biopyhton parsing
    gzip_stream = gzip.GzipFile(fileobj=resp.raw)
    text_handle = io.TextIOWrapper(gzip_stream, encoding="utf-8", newline="\n")
    
    # parse the structure
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb, text_handle)

    # for writing the biounits
    pdbio = PDBIO()
    pdbio.set_structure(structure)
 
    # to select the chains for each biounit
    selector = BiounitSelect()
    
    # path for this pdb
    pdb_path = out_path / Path(f"{pdb[1:3]}/{pdb}/")
    pdb_path.mkdir(exist_ok=True, parents=True)
    
    # loop through biounits
    for idx, biounit in enumerate(biounits):

        # get the chains and update the selector
        chains = biounit.split(",")
        selector.update_allowed_chains(chains)

        # save the biounit
        biounit_path = pdb_path / Path(f"{pdb}_{idx}.pdb")
        pdbio.save(str(biounit_path), select=selector)

    # also save a text file containing the chains for each biounit
    with open(pdb_path / Path("biounits.txt"), "w") as f:
        f.write("\n".join(f"{idx}: {chains}" for idx, chains in enumerate(biounits)))

class BiounitSelect(Select):
    def __init__(self):
        self.allowed_chains = []

    def accept_model(self, model):
        return model.id==0

    def accept_chain(self, chain):
        return chain.id in self.allowed_chains

    def update_allowed_chains(self, chains):
        self.allowed_chains = chains

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", default="/scratch/hjc2538/projects/proteusAI/data/multi_chain/processed", type=Path)
    parser.add_argument("--raw_pt_path", default="/scratch/hjc2538/projects/proteusAI/data/multi_chain/raw/pdb", type=Path)
    parser.add_argument("--max_size", default=8192, type=int)
    parser.add_argument("--max_resolution", default=3.5, type=float)
    parser.add_argument("--out_path", default="/scratch/hjc2538/projects/proteusAI/data/multi_chain/evaluation", type=Path)

    args = parser.parse_args()

    parse_test_set(args.data_path, args.raw_pt_path, args.max_size, args.max_resolution, args.out_path)
