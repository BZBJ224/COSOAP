# Cosine and SOAP Deduplication Tool

Supports single .xyz file or folder with multiple .xyz files.

## Install
```bash
pip install .
```
## Useage
```bash
COSOAP [-h] [-i INPUT_PATH] [-p NPROC] [-m {fps,threshold}] [-n NUM] [-s SIMLT] [-a ATOMS] [-r RCUT]
```

### optional arguments:

      -h, --help            
                        show this help message and exit
                        
      -i INPUT_PATH, --input INPUT_PATH
                        Input: single .xyz/.extxyz file or folder
                        
      -p NPROC, --nproc NPROC
                        Number of processes
                        
      -m {fps,threshold}, --mode {fps,threshold}
                        Selection mode: 'fps' (select fixed N) or 'threshold' (remove duplicates). Default: fps
                        
      -n NUM, --num NUM     
                        [FPS Mode] Total number of structures to select. Default: 1000
      
      -s SIMLT, --simlT SIMLT
                        [Threshold Mode] Similarity threshold (1 - cosine). Default: 0.005
                        
      -a ATOMS, --atoms ATOMS
                        SOAP centers (e.g. 'C H O')
                        
      -r RCUT, --rcut RCUT  
                        SOAP cutoff radius
  

### Output files:

    train.xyz       : Unique labeled structures
  
    test.xyz        : Similar labeled structures (for validation)


    soap_cache/     : Cached descriptors (reused on rerun)
