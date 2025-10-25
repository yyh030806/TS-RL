from morfeus import XTB
from typing import List, Tuple

from reactot.analyze.geomopt import calc_deltaE
import time

AU2EV = 27.2114

def parse_geom_string(geom_string: str) -> Tuple[List[str], List[List[float]]]:
    """
    解析 'Atom X Y Z; Atom X Y Z' 格式的字符串。

    Args:
        geom_string (str): 输入的分子几何结构字符串。

    Returns:
        Tuple[List[str], List[List[float]]]: 
            一个包含元素符号列表和坐标列表的元组。
    """
    elements = []
    coordinates = []
    
    # 使用分号分割每个原子
    atoms = geom_string.strip().split(';')
    
    for atom_data in atoms:
        # 去除每个原子数据前后的空白
        clean_atom_data = atom_data.strip()
        if not clean_atom_data:
            continue
            
        # 分割原子符号和坐标
        parts = clean_atom_data.split()
        
        # 第一个部分是元素符号
        elements.append(parts[0])
        
        # 剩下的部分是坐标，转换为浮点数
        coords = [float(c) for c in parts[1:]]
        coordinates.append(coords)
        
    return elements, coordinates

def calc_deltaE_xtb(geom_string1: str, geom_string2: str) -> float:
    """
    使用 GFN2-xTB 计算两个几何字符串之间的能量差。

    Args:
        geom_string1 (str): 第一个分子的几何字符串。
        geom_string2 (str): 第二个分子的几何字符串。

    Returns:
        float: 能量差 (E2 - E1)，单位为电子伏特 (eV)。
    """
    # --- 计算第一个几何结构的能量 ---
    # 1. 解析字符串得到元素和坐标
    elements1, coordinates1 = parse_geom_string(geom_string1)
    
    # 2. 创建 xTB 计算对象
    xtb1 = XTB(elements1, coordinates1)
    
    # 3. 执行单点能计算 (结果单位为 Hartree)
    energy1_au = xtb1._get_energy()

    # --- 计算第二个几何结构的能量 ---
    # 重复上述步骤
    elements2, coordinates2 = parse_geom_string(geom_string2)
    xtb2 = XTB(elements2, coordinates2)
    energy2_au = xtb2._get_energy()

    # --- 计算能量差并进行单位转换 ---
    delta_e_au = energy2_au - energy1_au
    delta_e_ev = delta_e_au * AU2EV
    
    return delta_e_ev

class EnergyScorer:
    def __init__(self, method: str='xtb'):
        self.method = method
        if method == 'xtb':
            self.score_func = calc_deltaE_xtb
        elif method == 'dft':
            self.score_func = calc_deltaE
        elif method == 'dl':
            raise(NotImplementedError())
    
    def __call__(self, st_tar, st_pred):

        return abs(self.score_func(st_tar, st_pred))
                
if __name__ == '__main__':
    st_tar = 'C 0.021 0.686 0.000; N 0.000 -0.784 0.000; H 0.948 1.082 0.000; H -0.474 1.082 0.821; H -0.474 1.082 -0.821; H 0.435 -1.169 0.803; H -0.435 -1.169 -0.803'
    st_pred = 'C 0.000 0.686 0.000; N 0.000 -0.784 0.000; H 0.948 1.082 0.000; H -0.474 1.082 0.821; H -0.474 1.082 -0.821; H 0.435 -1.169 0.803; H -0.435 -1.169 -0.803'
    
    print("==== xtb ====")
    start = time.time()
    scorer_xtb = EnergyScorer(method='xtb')
    score_xtb = scorer_xtb(st_tar, st_pred)
    end = time.time()
    print(f"score: {score_xtb}")
    print(f"time: {end - start}")
    
        
    print("==== dft ====")
    start = time.time()
    scorer_dft = EnergyScorer(method='dft')
    score_dft = scorer_dft(st_tar, st_pred)
    end = time.time()
    print(f"score: {score_dft}")
    print(f"time: {end - start}")