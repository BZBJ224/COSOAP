# Cosine and SOAP Deduplication Tool

Supports single .xyz file or folder with multiple .xyz files.

## Install
```bash
pip install .
```
## Useage
```bash
COSOAP [-h] [-i INPUT_PATH] [-p NPROC] [-s SIMLT] [-a ATOMS] [-r RCUT]
```

### optional arguments:

    -h, --help            
                        show this help message and exit
  
    -i INPUT_PATH, --input INPUT_PATH  
                        Input: single .xyz/.extxyz/.traj file or folder containing them
                        
    -p NPROC, --nproc NPROC
                        Number of processes (default: auto)
                        
    -s SIMLT, --simlT SIMLT
                        Similarity threshold (1 - cosine), smaller = stricter (default: 0.005)
                        
    -a ATOMS, --atoms ATOMS
                        Space-separated elements used as SOAP centers (default: 'C H O')
                        
    -r RCUT, --rcut RCUT  
                        SOAP cutoff radius in Å (default: 6.0)
  

### Output files:

    train.xyz       : Unique labeled structures
  
    test.xyz        : Similar labeled structures (for validation)
  
    unlabeled.xyz   : Deduplicated unlabeled structures
  
    uniques.csv     : Statistics per composition
  
    soap_cache/     : Cached descriptors (reused on rerun)
