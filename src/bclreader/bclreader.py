import numpy as np
import struct
import pathlib
import collections
from collections import abc

base_mapping = np.array([c for c in 'ACGTN'])

def _read_bcl_nr_clusters(filename):
    with open(filename, 'rb') as f:
        return struct.unpack('<I', f.read(4))[0]

_REVCOMP_TRANSLATION_TABLE = str.maketrans('ACTG', 'TGAC')
class Seq(str):
    def reverse_complement(self):
        return self[::-1].translate(_REVCOMP_TRANSLATION_TABLE)

    def __getitem__(self, *args, **kwargs):
        return Seq(super().__getitem__(*args, **kwargs))

class SequenceCollection(abc.Sequence):
    def __init__(self, raw):
        self._raw = raw

    def __len__(self):
        # Number of sequences
        return self._raw.shape[1]
    
    def __getitem__(self, index):
        raw_seq = self._raw[:, index]

        # If the byte is zero, the base call was unclear
        mask = raw_seq == 0

        # The base is encoded as the two least significant bits
        # 0x00 = A
        # 0x01 = C
        # 0x10 = G
        # 0x11 = T
        bases = raw_seq & 0b11
        # The remaining six bits encode the quality score
        quality = (raw_seq & 0b11111100) >> 2

        # We encode an 'N' as a 4.
        bases[mask] = 4
        # ... and set the quality score to 2 (that's Illumina standard)
        quality[mask] = 2

        # Return the sequence as string and the quality score as array of ints
        return {'sequence': Seq(''.join(base_mapping[bases])), 'quality': quality}

def read_cycles(directory):
    # Each Illumina sequencing cycle (i.e. base index) lives in each own subdirectory named Cxx.1
    # Don't know what the .1 stands for though
    cycle_dirs = sorted(directory.glob('C*.1'), key=lambda s: int(s.stem[1:]))
    # Tile info lives in bcl files inside the cycle directories, named consistently across the cycles
    bcls = [filename.stem for filename in cycle_dirs[0].glob('*.bcl')]
    # Get the number of clusters present in each bcl file by reading the header of the bcl files in the first cycle dir
    nr_clusters = [_read_bcl_nr_clusters(cycle_dirs[0] / (bcl + '.bcl')) for bcl in bcls]
    
    # New huge array holding the raw bytes from each cycle and each cluster in each tile
    # The offsets for each tile in our array
    offsets = np.cumsum([0]+nr_clusters)
    raw = np.empty((len(cycle_dirs), sum(nr_clusters)), dtype=np.uint8)

    # Iterate cycle directories
    for c, dir in enumerate(cycle_dirs):
        # Iterate bcl files / tiles
        for b, bcl in enumerate(bcls):
            # Open bcl file as binary read
            with open(dir / (bcl + '.bcl'), 'rb') as f:
                f.seek(4) # Skip header (number of clusters present)
                # Read the whole file into the stride of our raw array
                raw[c, offsets[b]:offsets[b+1]] = np.frombuffer(f.read(), dtype=np.uint8)

    # Make everything nice an accessible via a custom container class
    return SequenceCollection(raw)
