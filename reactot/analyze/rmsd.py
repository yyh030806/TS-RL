from typing import List
import numpy as np

from pymatgen.core import Molecule
from pymatgen.analysis.molecule_matcher import BruteForceOrderMatcher, GeneticOrderMatcher, HungarianOrderMatcher, KabschMatcher
from pymatgen.io.xyz import XYZ

from torch import Tensor


def xh2pmg(xh):
    mol = Molecule(
        species=xh[:, -1].long().cpu().numpy(),
        coords=xh[:, :3].cpu().numpy(),
    )
    return mol


def xyz2pmg(xyzfile):
    xyz_converter = XYZ(mol=None)
    mol = xyz_converter.from_file(xyzfile).molecule
    return mol


def rmsd_core(mol1, mol2, threshold=0.5, same_order=False):
    _, count = np.unique(mol1.atomic_numbers, return_counts=True)
    if same_order:
        bfm = KabschMatcher(mol1)
        _, rmsd = bfm.fit(mol2)
        return rmsd
    total_permutations = 1
    for c in count:
        total_permutations *= np.emath.factorial(c)  # type: ignore
    if total_permutations < 1e4:
        bfm = BruteForceOrderMatcher(mol1)
        _, rmsd = bfm.fit(mol2)
    else:
        bfm = GeneticOrderMatcher(mol1, threshold=threshold)
        pairs = bfm.fit(mol2)
        rmsd = threshold
        for pair in pairs:
            rmsd = min(rmsd, pair[-1])
        if not len(pairs):
            bfm = HungarianOrderMatcher(mol1)
            _, rmsd = bfm.fit(mol2)
    return rmsd


def pymatgen_rmsd(
    mol1,
    mol2,
    ignore_chirality: bool = False,
    threshold: float = 0.5,
    same_order: bool = True,
):
    if isinstance(mol1, str):
        mol1 = xyz2pmg(mol1)
    if isinstance(mol2, str):
        mol2 = xyz2pmg(mol2)
    rmsd = rmsd_core(mol1, mol2, threshold, same_order=same_order)
    if ignore_chirality:
        coords = mol2.cart_coords
        coords[:, -1] = -coords[:, -1]
        mol2_reflect = Molecule(
            species=mol2.species,
            coords=coords,
        )
        rmsd_reflect = rmsd_core(
            mol1, mol2_reflect, threshold, same_order=same_order)
        rmsd = min(rmsd, rmsd_reflect)
    return rmsd

def batch_rmsd(
    fragments_nodes: List[Tensor],
    out_samples: List[Tensor],
    xh: List[Tensor],
    idx: int = 1,
    threshold: float = 0.5,
    same_order: bool = False,
) -> List[float]:
    rmsds = []
    out_samples_use = out_samples[idx]
    xh_use = xh[idx]
    nodes = fragments_nodes[idx].long().cpu().numpy()
    start_ind, end_ind = 0, 0
    for jj, natoms in enumerate(nodes):
        end_ind += natoms
        mol1 = xh2pmg(out_samples_use[start_ind:end_ind])
        mol2 = xh2pmg(xh_use[start_ind:end_ind])
        try:
            rmsd = pymatgen_rmsd(
                mol1,
                mol2,
                ignore_chirality=True,
                threshold=threshold,
                same_order=same_order,
            )
        except:
            rmsd = 1
        rmsds.append(min(rmsd, 1.0))
        start_ind = end_ind
    return rmsds

def batch_rmsd_sb(
    fragments_node: Tensor,
    pred_xh: Tensor,
    target_xh: Tensor,
    threshold: float = 0.5,
    same_order: bool = True,
) -> List[float]:

    rmsds = []

    end_ind = np.cumsum(fragments_node.long().cpu().numpy())
    start_ind = np.concatenate([np.int64(np.zeros(1)), end_ind[:-1]])

    for start, end in zip(start_ind, end_ind):
        mol1 = xh2pmg(pred_xh[start : end])
        mol2 = xh2pmg(target_xh[start : end])
        rmsd = pymatgen_rmsd(
            mol1,
            mol2,
            ignore_chirality=True,
            threshold=threshold,
            same_order=same_order,
        )
        
        rmsds.append(min(rmsd, 1.0))
    return rmsds

def mol_string_to_pymatgen(mol_string):
    """
    将单行字符串格式的分子坐标转换为 pymatgen 的 Molecule 对象。
    """
    species = []
    coords = []
    atoms_data = mol_string.strip().split(';')
    for atom_data in atoms_data:
        if not atom_data.strip():
            continue
        parts = atom_data.strip().split()
        species.append(parts[0])
        coords.append([float(p) for p in parts[1:]])
    return Molecule(species, coords)

def rmsd_str(mol1, mol2):
    mol1 = mol_string_to_pymatgen(mol1)
    mol2 = mol_string_to_pymatgen(mol2)
    
    return pymatgen_rmsd(mol1,mol2)