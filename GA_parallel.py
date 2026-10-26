import os, sys
import shutil
import numpy as np
import random
import uuid
import string
import traceback
import pickle
import time
from datetime import datetime
from multiprocessing import Pool
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt  
from scipy.spatial import cKDTree

def _bind_worker():
    import os
    from multiprocessing import current_process
    identity = current_process()._identity
    if not identity:
        return
    cores = list(range(0, 16))
    worker_id = identity[0]   
    core_id = cores[(worker_id - 1) % len(cores)]
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {core_id})
        print(f"并行评估进程 {worker_id} 已绑定到CPU {core_id}")
    else:
        print(f"并行评估进程 {worker_id} 已启动")

def _worker_timeout_seconds():
    try:
        return max(60.0, float(os.environ.get("GA_WORKER_TIMEOUT_SECONDS", "172800")))
    except ValueError:
        return 1200.0

def _failed_evaluation_result(payload, message):
    return {
        'success': False,
        'positions':payload['positions'],
        'sim_gr': None,
        'id': payload.get('id'),
        'energy': float('inf'),
        'pdf': float('inf'),
        'error': str(message)}

CURRENT_DIR = os.getcwd()
REAXFF_PARAMS = os.path.join(CURRENT_DIR, "coal-HCONSB.ff")
CONTROL_PARAMS = os.path.join(CURRENT_DIR, "lmp_control")
PDF_DATA = os.path.join(CURRENT_DIR, "coal_model.gr")
INITIAL_PDB = os.path.join(CURRENT_DIR, "coal_modelchu.pdb")
TOTAL_HRMC_STEPS = 50000
HRMC_INITIAL_TEMPERATURE = 3000
HRMC_FINAL_TEMPERATURE = 300 

def _load_global_structure():
    coords, elements = [],[]
    box_dims = None
    with open(INITIAL_PDB, 'r') as f:
        for line in f:
            if "ATOM" in line.upper().strip():
                coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                elements.append(line[76:78].strip())
            elif line.startswith('REMARK') and 'Boundary Conditions' in line:
                parts = line.split(':')[1].strip().split()
                box_dims = np.array([float(parts[0]), float(parts[4]), float(parts[8])])
    return np.array(coords), elements, box_dims
GLOBAL_INITIAL_COORDS, GLOBAL_ELEMENTS, GLOBAL_BOX_DIMS = _load_global_structure()
GLOBAL_BOX_VECTORS =[
    np.array([GLOBAL_BOX_DIMS[0], 0, 0]),
    np.array([0, GLOBAL_BOX_DIMS[1], 0]),
    np.array([0, 0, GLOBAL_BOX_DIMS[2]])]

def write_pdb(path, coords, elements, box_dims):
    with open(path, 'w', encoding='utf-8') as f:
        f.write("REMARK   Materials Studio PDB file\n")
        f.write(
            f"REMARK    Boundary Conditions: {box_dims[0]:6.3f}  0.0  0.0  0.0  "
            f"{box_dims[1]:6.3f}  0.0  0.0  0.0  {box_dims[2]:6.3f} \n")
        for i, (c, e) in enumerate(zip(coords, elements), 1):
            atom_name = f"{e}1"
            f.write(
                f"ATOM  {i:5d}  {atom_name:>3} MOL{i:5d}     "
                f"{c[0]:8.3f}{c[1]:8.3f}{c[2]:8.3f}  1.00  0.00          {e:>2}\n")

BOND_MIN_DISTANCES = {
    ('C', 'C'): 1.39,
    ('C', 'H'): 1.10,
    ('C', 'O'): 1.21,
    ('H', 'O'): 0.95,
    ('H', 'N'): 1.00,
    ('C', 'N'): 1.37,
    ('C', 'S'): 1.82,
    ('H', 'S'): 1.32}

def _worker_run_evaluation(payload):
    individual_id = payload['id']
    positions = payload['positions']
    tmp_dir = payload['tmp_dir']
    eval_mode = payload.get('eval_mode', 'static')   
    if eval_mode == 'static':
        import os as _os 
        import numpy as _np
        import uuid as _uuid
        import traceback as _traceback
        from pdbparser.Utilities.BoundaryConditions import PeriodicBoundaries as _PeriodicBoundaries
        from fullrmc.Constraints.PairDistributionConstraints import PairDistributionConstraint as _PairDistributionConstraint
        from fullrmc.Constraints.EnergyConstraints import EnergyConstraint as _EnergyConstraint
        from fullrmc.Constraints.DistanceConstraints import InterMolecularDistanceConstraint as _InterMolecularDistanceConstraint
        from fullrmc.Engine import Engine as _Engine
        from fullrmc.Constraints.PairDistributionConstraints import PairDistributionConstraint as _PairDistributionConstraint
        from MDHRMCEngine import MDHRMCEngine as _MDHRMCEngine
        from GA_parallel import write_pdb
        engine = None
        tmp_pdb = None
        try:
            _os.makedirs(tmp_dir, exist_ok=True)
            tmp_pdb = _os.path.join(tmp_dir, f"{individual_id}_{_uuid.uuid4().hex[:6]}.pdb")
            write_pdb(tmp_pdb, positions, GLOBAL_ELEMENTS, GLOBAL_BOX_DIMS)
            print(f"个体{individual_id}: 结构写入临时PDB文件: {tmp_pdb}")
            engine = _MDHRMCEngine(path=None, freshStart=True)
            engine.set_pdb(tmp_pdb)
            print("[DEBUG]set_pdb 完成")
            print("[DEBUG]原子数:", engine.numberOfAtoms)
            engine.set_boundary_conditions(_PeriodicBoundaries(GLOBAL_BOX_DIMS))   
            pdf_constraint = _PairDistributionConstraint( experimentalData=PDF_DATA, weighting="atomicNumber", scaleFactor=1.0, adjustScaleFactor=[0, 0.8, 1.2])
            energy_constraint = _EnergyConstraint(  reaxff_params=REAXFF_PARAMS, control_params=CONTROL_PARAMS, compute_forces=False)
            inter_constraint = _InterMolecularDistanceConstraint(defaultDistance=0.9, typeDefinition='element', flexible=False, rejectProbability=1.0)
            inter_constraint.set_type_definition('element',
    pairsDistanceDefinition=[('C','C', 1.39), ('C','H', 1.10), ('C','O', 1.21), ('C','S', 1.82), ('C','N', 1.37),
                             ('O','H', 0.95), ('N','H', 1.00), ('S','H', 1.32), ('H','H', 0.75),('O','O', 1.75), 
                             ('N','N', 2.05), ('S','S', 2.15), ('O','N', 1.75), ('N','S', 2.15), ('O','S', 1.15)])
            engine.add_constraints([pdf_constraint, energy_constraint, inter_constraint]) 
            engine.initialize_used_constraints()
            print(f"正在为个体 {individual_id} 执行纯粹的单点能量与PDF评估...")
            plain_pdf_error, energy, sim_gr = engine.evaluate_static_structure(reaxff_params=REAXFF_PARAMS ,control_params=CONTROL_PARAMS)
            print(f"个体 {individual_id} 评估完成: Energy={energy}, plain_pdf_error={plain_pdf_error}")
            if not _np.isfinite(energy) or not _np.isfinite(plain_pdf_error) or energy < -5000000:
                raise ValueError(f"严重异常: 个体 {individual_id} 能量({energy})或PDF误差({plain_pdf_error})发生非物理爆炸 (NaN/Inf 或 能量 < -500万)！")
            return {
                'success': True,
                'id': individual_id,
                'positions' : payload['positions'],
                'energy': energy, 
                'pdf': plain_pdf_error,
                'sim_gr': sim_gr}
        except Exception as e:
            print("Worker评估未完成，已记录错误摘要")
            tb = _traceback.format_exc()
            return _failed_evaluation_result(payload, f"{e}\n{tb}")
        finally:
            if engine is not None:
                try:
                    engine.cleanup()
                except Exception:
                    pass
            if tmp_pdb is not None and _os.path.exists(tmp_pdb):
                try: 
                    _os.remove(tmp_pdb)
                except Exception: 
                    pass
    else:
        import os as _os
        import numpy as _np
        import uuid as _uuid
        import traceback as _traceback
        from pdbparser.Utilities.BoundaryConditions import PeriodicBoundaries as _PeriodicBoundaries
        from fullrmc.Globals import LOGGER as _LOGGER
        from fullrmc.Engine import Engine as _Engine
        from fullrmc.Core.Group import Group as _Group
        from fullrmc.Constraints.PairDistributionConstraints import PairDistributionConstraint as _PairDistributionConstraint
        from fullrmc.Constraints.EnergyConstraints import EnergyConstraint as _EnergyConstraint
        from fullrmc.Constraints.DistanceConstraints import InterMolecularDistanceConstraint as _InterMolecularDistanceConstraint  
        from MDHRMCEngine import MDHRMCEngine as _MDHRMCEngine
        from fullrmc.Generators.Translations import TranslationGenerator as _TranslationGenerator
        from fullrmc.Generators.Rotations import RotationGenerator as _RotationGenerator
        from fullrmc.Core.MoveGenerator import MoveGeneratorCollector as _MoveGeneratorCollector
        from fullrmc.Core.GroupSelector import RecursiveGroupSelector as _RecursiveGroupSelector
        from fullrmc.Selectors.RandomSelectors import RandomSelector as _RandomSelector
        from GA_parallel import write_pdb
        engine = None
        workdir = None
        try:
            _os.makedirs(tmp_dir, exist_ok=True)
            workdir = _os.path.join(tmp_dir, f"individual_{individual_id}")
            _os.makedirs(workdir, exist_ok=True)
            print(f"个体{individual_id}:MD-HRMC模拟输出目录创建于:{workdir}")
            engine = _MDHRMCEngine(path=None, freshStart=True, log_output_dir=workdir)
            engine.setup_output_files(workdir)            
            tmp_pdb = _os.path.join(workdir, f"{individual_id}_{_uuid.uuid4().hex[:6]}.pdb")
            write_pdb(tmp_pdb, positions, GLOBAL_ELEMENTS, GLOBAL_BOX_DIMS)
            print(f"个体{individual_id}: 结构写入临时PDB文件: {tmp_pdb}")
            engine.set_pdb(tmp_pdb)
            print("[DEBUG]set_pdb 完成")
            print("[DEBUG]原子数:", engine.numberOfAtoms)
            engine.set_boundary_conditions(_PeriodicBoundaries(GLOBAL_BOX_DIMS))
            elements_constraint = ['C', 'H', 'O', 'N', 'S']   
            pdf_constraint = _PairDistributionConstraint( experimentalData=PDF_DATA, weighting="atomicNumber", scaleFactor=1.0, adjustScaleFactor=[0, 0.8, 1.2])
            energy_constraint = _EnergyConstraint(  reaxff_params=REAXFF_PARAMS, control_params=CONTROL_PARAMS, compute_forces=False)
            inter_constraint = _InterMolecularDistanceConstraint(defaultDistance=0.9, typeDefinition='element', flexible=False, rejectProbability=1.0)
            inter_constraint.set_type_definition('element',
    pairsDistanceDefinition=[('C','C', 1.39), ('C','H', 1.10), ('C','O', 1.21), ('C','S', 1.82), ('C','N', 1.37),
                             ('O','H', 0.95), ('N','H', 1.00), ('S','H', 1.32), ('H','H', 0.75),('O','O', 1.75), 
                             ('N','N', 2.05), ('S','S', 2.15), ('O','N', 1.75), ('N','S', 2.15), ('O','S', 1.15)])

            engine.add_constraints([pdf_constraint, energy_constraint, inter_constraint]) 
            energy_constraint.initialize() 
            groups = [[idx] for idx, el in enumerate(engine.allElements) if (el in elements_constraint)]
            engine.set_groups(groups) 
            trans_gen = _TranslationGenerator(amplitude=0.1)
            for group in engine.groups:
                group.set_move_generator(trans_gen)
            selector = _RandomSelector(engine)
            group_selector = _RecursiveGroupSelector(selector=selector, recur=10, refine=False, explore=True)
            engine.set_group_selector(group_selector)
            engine.initialize_used_constraints()
            engine.setup_md(reaxff_params=REAXFF_PARAMS, control_params=CONTROL_PARAMS)
            engine.setup_hrmc(initial_temp=HRMC_INITIAL_TEMPERATURE, final_temp=HRMC_FINAL_TEMPERATURE, steps=TOTAL_HRMC_STEPS)
            print(f"\n------ 正在为个体{individual_id}运行MD-HRMC模拟... ------")
            engine._run_md_step()
            engine.run_hrmc()
            print(f"个体{individual_id}的MD-HRMC已模拟完成")
            energy = engine.energy_constraint.compute_energy()
            print(f"个体{individual_id}的能量计算完成: {energy}")
            sim_gr = engine.pdf_constraint.get_constraint_value()["total"]
            plain_pdf_error = engine.compute_chi_square(sim_gr)
            print(f"个体{individual_id}的PDF误差计算完成: plain_pdf_error={plain_pdf_error}")
            final_positions = engine.realCoordinates       
            return {
                'success': True,
                'id': individual_id, 
                'energy': energy, 
                'pdf': plain_pdf_error,
                'positions': final_positions, 
                'sim_gr':sim_gr,
                'error': None}
        except Exception as e:
            print("Worker评估未完成，已记录错误摘要")
            tb = _traceback.format_exc()
            return _failed_evaluation_result(payload, f"{e}\n{tb}")    
        finally:
            if engine is not None:
                try:
                    engine.cleanup()
                except Exception:
                    pass

class GA_MDHRMC_Optimizer:
    def __init__(self, population_size=50, generations=100, md_hrmc_interval=500,
                 checkpoint_path=None):
        self.population_size = population_size
        self.generations = generations
        self.md_hrmc_interval = md_hrmc_interval
        self.checkpoint_path = checkpoint_path       
        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            print(f"发现断点文件: {self.checkpoint_path}，正在恢复环境...")
            with open(self.checkpoint_path, 'rb') as f:
                ckpt = pickle.load(f)   
            self.base_output_dir = ckpt['base_output_dir']
            self.population = ckpt['population']
            self.start_gen = ckpt['last_completed_gen']            
            self.history_generation = ckpt['history_generation']
            self.global_best = ckpt['global_best']
            self.history_best_pdf = ckpt['history_best_pdf']
            self.history_best_energy = ckpt['history_best_energy']
            self.history_avg_pdf = ckpt['history_avg_pdf']
            self.history_std_pdf = ckpt['history_std_pdf']
            self.history_all_pdfs = ckpt['history_all_pdfs']
            self.history_all_energies = ckpt['history_all_energies']
            print(f"断点加载成功！已完成前 {self.start_gen} 代的模拟")
        else:
            self.start_gen = 0
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.base_output_dir = os.path.join(CURRENT_DIR, f"GA_{timestamp}")
            os.makedirs(self.base_output_dir, exist_ok=True)
            print(f"全新启动GA优化，结果将保存到目录: {self.base_output_dir}")
            self.population =[]            
            self.history_generation = []
            self.global_best = {
                'id': None, 'positions': None, 'fitness': -1.0, 'energy': float('inf'), 'pdf': float('inf'), 'sim_gr': None}
            self.history_best_pdf = []
            self.history_best_energy =[]
            self.history_avg_pdf = []
            self.history_std_pdf =[]
            self.history_all_pdfs = []
            self.history_all_energies =[]
        print(f"population_size:{self.population_size},generations:{self.generations}")    
    
    def _write_pdb(self, path, coords, elements, box_dims):
        write_pdb(path, coords, elements, box_dims)   
    
    def _copy_individual_snapshot(self, ind):
        snap = {
            'id': ind['id'],
            'positions': ind['positions'].copy() if hasattr(ind.get('positions'), 'copy') else ind.get('positions'),
            'energy': ind['energy'],
            'pdf': ind['pdf'],
            'fitness': ind.get('fitness', -1.0),
            'sim_gr': ind.get('sim_gr')}
        return snap
    
    def _save_checkpoint(self, generation):
        ckpt_data = {
            'base_output_dir': self.base_output_dir,
            'population': self.population,
            'last_completed_gen': generation,           
            'history_generation': self.history_generation,
            'global_best': self.global_best,
            'history_best_pdf': self.history_best_pdf,
            'history_best_energy': self.history_best_energy,
            'history_avg_pdf': self.history_avg_pdf,
            'history_std_pdf': self.history_std_pdf,
            'history_all_pdfs': self.history_all_pdfs,
            'history_all_energies': self.history_all_energies}
        tmp_ckpt = os.path.join(self.base_output_dir, "checkpoint_tmp.pkl")
        final_ckpt = os.path.join(self.base_output_dir, "checkpoint.pkl")
        with open(tmp_ckpt, 'wb') as f:
            pickle.dump(ckpt_data, f)
        shutil.move(tmp_ckpt, final_ckpt)
        print(f"第 {generation} 代进度已安全保存至断点文件")
        checkpoint_interval = 1000
        if generation % checkpoint_interval == 0:
            archive_ckpt = os.path.join(self.base_output_dir,f"checkpoint_gen{generation:04d}.pkl")
            with open(archive_ckpt, 'wb') as f:
                pickle.dump(ckpt_data, f)
            print(f"第{generation}代永久断点已保存："f"{archive_ckpt}")
    
    def _generate_random_suffix(self, length=4):
        alphabet = string.ascii_uppercase + string.digits
        return ''.join(random.choices(alphabet, k=length))
    
    def _get_parent_info_from_id(self, parent_id):
        parts = parent_id.split('_')
        if len(parts) < 2:
            raise ValueError(f"无法解析父代编号: {parent_id}")
        parent_seq = ''.join(ch for ch in parts[1] if ch.isdigit())
        if not parent_seq:
            fallback = ''.join(ch for ch in parts[1] if ch.isalnum())
            return fallback[-3:].rjust(3, '0')
        return parent_seq[-3:].zfill(3)    
    
    def _create_chaotic_structure(self, initial_coords, box_dims, individual_index, total_target, scale_mode='initial'):
        rand_val = random.random()
        if scale_mode == 'plateau':
            if rand_val < 0.55:
                scale = random.uniform(0.15, 0.45)
            elif rand_val < 0.90:
                scale = random.uniform(0.45, 0.90)
            else:
                scale = random.uniform(0.90, 1.35)
        else:
            if rand_val < 0.15:
                scale = random.uniform(0.5, 1.0)
            elif rand_val < 0.5:
                scale = random.uniform(2.0, 4.0)
            else:
                scale = random.uniform(5.0, 8.0)
        seed = individual_index / (total_target + 1.0)
        N_atoms = len(initial_coords)
        chaos_seq = self._generate_chaotic_sequence(N_atoms * 3, x0=seed)
        chaos_matrix = chaos_seq.reshape((N_atoms, 3))
        perturbation = (chaos_matrix - 0.5) * 2.0 * scale
        perturbed_coords = initial_coords + perturbation
        for dim in range(3):
            perturbed_coords[:, dim] = np.mod(perturbed_coords[:, dim], box_dims[dim])
        return perturbed_coords   
    
    def _generate_chaotic_sequence(self, length, x0=None):
        if x0 is None:
            x0 = random.random()
        try:
            x0 = float(x0)
        except (TypeError, ValueError):
            x0 = random.random()
        if not np.isfinite(x0):
            x0 = random.random()
        x0 = abs(x0) % 1.0
        bad_points = (0.0, 0.25, 0.5, 0.75, 1.0)
        if x0 < 1e-6 or any(abs(x0 - val) < 1e-6 for val in bad_points):
            x0 = (x0 + 0.137931) % 1.0
        x0 = min(max(x0, 1e-6), 1.0 - 1e-6)
        seq = np.zeros(length)
        x = x0
        for i in range(length):
            x = 4 * x * (1 - x) 
            if not np.isfinite(x) or x <= 0.0 or x >= 1.0:
                x = ((i + 1) * 0.61803398875 + x0) % 1.0
                x = min(max(x, 1e-6), 1.0 - 1e-6)
            seq[i] = x
        return seq
    
    def _evaluate_population_parallel(self, payloads, n_workers):
        worker_count = max(1, int(n_workers or 1))
        timeout_seconds = _worker_timeout_seconds()
        results = []
        for start in range(0, len(payloads), worker_count):
            wave = payloads[start:start + worker_count]
            wave_results = [None] * len(wave)
            pool = Pool(processes=len(wave), initializer=_bind_worker, maxtasksperchild=1)
            jobs = [(idx, payload, pool.apply_async(_worker_run_evaluation, (payload,))) for idx, payload in enumerate(wave)]
            deadline = time.monotonic() + timeout_seconds
            pending = set(range(len(jobs)))
            timed_out = False
            while pending:
                for idx in list(pending):
                    _, payload, async_result = jobs[idx]
                    if not async_result.ready():
                        continue
                    try:
                        wave_results[idx] = async_result.get(timeout=0.01)
                    except Exception as exc:
                        wave_results[idx] = _failed_evaluation_result(payload, exc)
                    pending.remove(idx)
                if not pending:
                    break
                if time.monotonic() >= deadline:
                    timed_out = True
                    for idx in pending:
                        _, payload, _ = jobs[idx]
                        wave_results[idx] = _failed_evaluation_result(
                            payload,
                            f"评估任务超过限定时间: {timeout_seconds:.0f}s")
                    break
                time.sleep(0.5)
            if timed_out:
                print(
                    f"警告: 本批 {len(pending)} 个评估超过 {timeout_seconds:.0f}s，"
                    "已终止该批 worker 并继续下一批")
                pool.terminate()
            else:
                pool.close()
            pool.join()
            for idx, payload in enumerate(wave):
                if wave_results[idx] is None:
                    wave_results[idx] = _failed_evaluation_result(payload, "worker returned no result")
            results.extend(wave_results)
        return results
    
    def select_parents(self, parents_pool):
        fitnesses = np.array([max(ind['fitness'], 1e-12) for ind in parents_pool], dtype=float)
        probs = fitnesses / fitnesses.sum()
        try:
            idxs = np.random.choice(len(parents_pool), size=2, replace=False, p=probs)
        except ValueError:
            idxs = np.random.choice(len(parents_pool), size=2, replace=True, p=probs)
        return parents_pool[idxs[0]], parents_pool[idxs[1]]
    
    def crossover(self, parent1, parent2, current_generation):
        print(f"----- 正在进行交叉 (父代:{parent1['id']} 和 {parent2['id']}) -----")
        coords1 = parent1['positions'].copy()
        coords2 = parent2['positions'].copy()
        elements = np.array(GLOBAL_ELEMENTS)
        box_dims = GLOBAL_BOX_DIMS.copy()
        if len(coords1) != len(coords2):
            raise ValueError("父代原子数不一致，不能进行交叉！")
        N = len(coords1) 
        if current_generation < 5000:
            lambda_C, max_d, b_ON, b_S, gamma_val = 0.20, 0.50, 0.30, 0.15, 2.0
        elif current_generation < 15000:
            lambda_C, max_d, b_ON, b_S, gamma_val = 0.12, 0.30, 0.20, 0.10, 1.0
        else:
            lambda_C, max_d, b_ON, b_S, gamma_val = 0.05, 0.10, 0.10, 0.05, 0.5
        # 生成第一个子代: A -> B 输运驱动
        child1_coords = self._CSG_crossover(
            coords1, coords2, elements, box_dims,
            lambda_C=lambda_C, max_disp=max_d,
            beta_O_N=b_ON, beta_S=b_S, gamma=gamma_val)
        # 生成第二个子代: B -> A 反向输运驱动
        child2_coords = self._CSG_crossover(
            coords2, coords1, elements, box_dims,
            lambda_C=lambda_C, max_disp=max_d,
            beta_O_N=b_ON, beta_S=b_S, gamma=gamma_val)
        child1_coords = np.mod(child1_coords, box_dims)
        child2_coords = np.mod(child2_coords, box_dims)
        print(f"----- 正在执行子代的原子重叠检查与修复 ... -----")
        child1_coords = self._resolve_overlaps(child1_coords, elements, box_dims, overlap_factor=0.75, verbose=False)
        child2_coords = self._resolve_overlaps(child2_coords, elements, box_dims, overlap_factor=0.75, verbose=False)        
        if len(child1_coords) != N or len(child2_coords) != N:
            raise ValueError("子代原子数异常！")
        parent1_seq = self._get_parent_info_from_id(parent1['id'])
        parent2_seq = self._get_parent_info_from_id(parent2['id'])
        random_suffix1 = self._generate_random_suffix()
        random_suffix2 = self._generate_random_suffix()
        child1_id = f"G{current_generation}_{parent1_seq}_Cross_{random_suffix1}"
        child2_id = f"G{current_generation}_{parent2_seq}_Cross_{random_suffix2}"
        child1 = {
            'id': child1_id,
            'positions': child1_coords,   
            'fitness': 0.0,
            'energy': 0.0,
            'pdf': 0.0,
            'parent_ids': [parent1['id'], parent2['id']],
            'generation_method': 'Cross'}
        child2 = {
            'id': child2_id,
            'positions': child2_coords,    
            'fitness': 0.0,
            'energy': 0.0,
            'pdf': 0.0,
            'parent_ids': [parent1['id'], parent2['id']],
            'generation_method': 'Cross'}
        print(f"------ 交叉成功! 生成子代 {child1_id}, {child2_id} ------")
        return child1, child2
    
    def _calculate_heavy_coordination_numbers(self, s_coords, heavy_indices, target_indices, box_dims, r_cut=1.85):
        s_heavy = s_coords[heavy_indices]    
        target_in_heavy_mask = np.isin(heavy_indices, target_indices) 
        diff = s_heavy[None, :, :] - s_heavy[:, None, :]
        diff = diff - np.round(diff) 
        dist = np.linalg.norm(diff * box_dims, axis=-1)
        bond_matrix = (dist > 1e-5) & (dist <= r_cut) 
        heavy_CN = np.sum(bond_matrix, axis=1)  
        return heavy_CN[target_in_heavy_mask].astype(float)
    
    def _kabsch_fit(self, P, Q):
        centroid_P = np.mean(P, axis=0) 
        centroid_Q = np.mean(Q, axis=0)
        p_shift = P - centroid_P
        q_shift = Q - centroid_Q    
        H = np.dot(p_shift.T, q_shift)
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T) 
        if np.linalg.det(R) < 0: 
            Vt[2, :] *= -1
            R = np.dot(Vt.T, U.T)
        t = centroid_Q - np.dot(centroid_P, R.T)
        return R, t
    
    def _CSG_crossover(self, parentA_coords, parentB_coords, elements, box_dims, 
                    lambda_C=0.15, max_disp=0.30, 
                    beta_O_N=0.20, beta_S=0.10, 
                    gamma=1.0, k_anchors=4):
        coordsA = np.asarray(parentA_coords, dtype=float)
        coordsB = np.asarray(parentB_coords, dtype=float)
        elements = np.asarray(elements)
        N = len(coordsA)
        sA = coordsA / box_dims
        sB = coordsB / box_dims  
        c_idx = np.flatnonzero(elements == 'C')
        o_idx = np.flatnonzero(elements == 'O')
        n_idx = np.flatnonzero(elements == 'N')
        s_idx = np.flatnonzero(elements == 'S')
        h_idx = np.flatnonzero(elements == 'H')
        heavy_idx = np.flatnonzero(elements != 'H') 
        s_child = sA.copy()
        
        if len(c_idx) > 0:
            sA_C = sA[c_idx]  
            sB_C = sB[c_idx]
            n_C = len(c_idx)
            diff_s = sB_C[None, :, :] - sA_C[:, None, :] 
            diff_s = diff_s - np.round(diff_s)
            dist_matrix = np.linalg.norm(diff_s * box_dims, axis=-1)         
            R_cutoff = 1.85  
            CN_A = self._calculate_heavy_coordination_numbers(sA, heavy_idx, c_idx, box_dims, r_cut=R_cutoff)
            CN_B = self._calculate_heavy_coordination_numbers(sB, heavy_idx, c_idx, box_dims, r_cut=R_cutoff)
            sigma_x = 1.8         
            sigma_x_sq = sigma_x**2 
            lambda_desc = 1.0    
            Cost_space = (dist_matrix ** 2) / sigma_x_sq
            diff_CN = CN_A[:, None] - CN_B[None, :]  
            Cost_chem = lambda_desc * (diff_CN ** 2)
            Cost = Cost_space + Cost_chem  
            a = np.ones(n_C) / n_C 
            b = np.ones(n_C) / n_C
            Pi = ot.sinkhorn(a, b, Cost, reg=gamma, numItermax=1000) 
            Pi_sum = np.sum(Pi, axis=1, keepdims=True) 
            Pi_sum[Pi_sum == 0] = 1.0 
            u_C = np.sum(Pi[:, :, None] * diff_s, axis=1) / Pi_sum
            u_C_phys = u_C * box_dims
            disp_norms = np.linalg.norm(u_C_phys, axis=-1, keepdims=True)
            disp_norms[disp_norms == 0] = 1.0
            scale_factor = np.minimum(1.0, max_disp / disp_norms)
            u_C = u_C * scale_factor 
            s_child[c_idx] = (sA[c_idx] + lambda_C * u_C) % 1.0
        
        hetero_configs = [   
            (o_idx, beta_O_N),
            (n_idx, beta_O_N),
            (s_idx, beta_S)]
        if len(c_idx) >= 3:
            tree_c_A = cKDTree(sA[c_idx] * box_dims, boxsize=box_dims) 
            coords_c_child = s_child[c_idx] * box_dims
            for X_idx_list, beta_X in hetero_configs:
                n_X = len(X_idx_list)
                if n_X == 0:  
                    continue
                sA_X = sA[X_idx_list]
                sB_X = sB[X_idx_list]
                diff_s_X = sB_X[None, :, :] - sA_X[:, None, :] 
                diff_s_X = diff_s_X - np.round(diff_s_X)
                dist_matrix_X = np.linalg.norm(diff_s_X * box_dims, axis=-1)
                CN_A_X = self._calculate_heavy_coordination_numbers(sA, heavy_idx, X_idx_list, box_dims, r_cut=1.85)
                CN_B_X = self._calculate_heavy_coordination_numbers(sB, heavy_idx, X_idx_list, box_dims, r_cut=1.85)
                Cost_space_X = (dist_matrix_X ** 2) / sigma_x_sq
                diff_CN_X = CN_A_X[:, None] - CN_B_X[None, :]
                Cost_chem_X = lambda_desc * (diff_CN_X ** 2)
                Cost_X = Cost_space_X + Cost_chem_X
                a_X = np.ones(n_X) / n_X
                b_X = np.ones(n_X) / n_X
                Pi_X = ot.sinkhorn(a_X, b_X, Cost_X, reg=gamma, numItermax=1000)          
                Pi_sum_X = np.sum(Pi_X, axis=1, keepdims=True)
                Pi_sum_X[Pi_sum_X == 0] = 1.0
                u_X_OT_all = np.sum(Pi_X[:, :, None] * diff_s_X, axis=1) / Pi_sum_X            
                u_X_OT_phys = u_X_OT_all * box_dims
                disp_norms_X = np.linalg.norm(u_X_OT_phys, axis=-1, keepdims=True)
                disp_norms_X[disp_norms_X == 0] = 1.0
                scale_factor_X = np.minimum(1.0, max_disp / disp_norms_X)
                u_X_OT_all = u_X_OT_all * scale_factor_X
                u_X_OT = u_X_OT_all            
                for loop_i, idx in enumerate(X_idx_list):
                    pos_X_A = sA[idx] * box_dims 
                    k_curr = min(k_anchors, len(c_idx)) 
                    _, anchor_local_indices = tree_c_A.query(pos_X_A, k=k_curr)
                    P_anchors = sA[c_idx[anchor_local_indices]] * box_dims 
                    Q_anchors = coords_c_child[anchor_local_indices] 
                    P_shift = P_anchors - P_anchors[0] 
                    P_shift = P_shift - np.round(P_shift / box_dims) * box_dims
                    P_anchors_cont = P_anchors[0] + P_shift
                    Q_shift = Q_anchors - Q_anchors[0]
                    Q_shift = Q_shift - np.round(Q_shift / box_dims) * box_dims
                    Q_anchors_cont = Q_anchors[0] + Q_shift
                    pos_X_A_shift = pos_X_A - P_anchors[0]
                    pos_X_A_shift = pos_X_A_shift - np.round(pos_X_A_shift / box_dims) * box_dims
                    pos_X_A_cont = P_anchors[0] + pos_X_A_shift
                    R, t = self._kabsch_fit(P_anchors_cont, Q_anchors_cont)    
                    pos_X_rigid = np.dot(pos_X_A_cont, R.T) + t  
                    u_X_anchor = (pos_X_rigid - pos_X_A) / box_dims 
                    u_X_anchor = u_X_anchor - np.round(u_X_anchor)
                    u_X_OT_current = u_X_OT_all[loop_i]
                    u_X_final = (1.0 - beta_X) * u_X_anchor + beta_X * u_X_OT_current
                    s_child[idx] = (sA[idx] + lambda_C * u_X_final) % 1.0
        
        if len(h_idx) > 0 and len(heavy_idx) > 0:
            sA_H = sA[h_idx] 
            sA_heavy = sA[heavy_idx] 
            u_heavy_frac = s_child[heavy_idx] - sA_heavy
            u_heavy_frac = u_heavy_frac - np.round(u_heavy_frac)
            u_heavy_phys = u_heavy_frac * box_dims 
            tree_heavy_A = cKDTree(sA_heavy * box_dims, boxsize=box_dims) 
            K_neighbors = min(4, len(heavy_idx))
            sigma = 1.5 
            for i, h_i in enumerate(h_idx):
                pos_h_phys = sA_H[i] * box_dims  
                dists, near_indices = tree_heavy_A.query(pos_h_phys, k=K_neighbors) 
                weights = np.exp(-(dists ** 2) / (2.0 * (sigma ** 2)))
                w_sum = np.sum(weights)
                if w_sum < 1e-12:
                    weights = np.ones_like(weights) / len(weights)
                else:
                    weights = weights / w_sum
                u_H_phys = np.sum(weights[:, None] * u_heavy_phys[near_indices], axis=0)
                s_child[h_i] = (sA[h_i] + u_H_phys / box_dims) % 1.0
        child_coords = s_child * box_dims
        return child_coords   
    
    def mutate(self, individual, current_generation):
        original_individual = individual
        parent_seq = self._get_parent_info_from_id(original_individual['id'])
        random_suffix = self._generate_random_suffix()
        mutated_id = f"G{current_generation}_{parent_seq}_-Mutation-{random_suffix}"
        print(f"----- 正在尝试变异个体 {original_individual['id']} → {mutated_id} -----")
        coords = original_individual['positions'].copy()
        elements = GLOBAL_ELEMENTS
        box_vectors = GLOBAL_BOX_VECTORS
        box_dims = np.array([np.linalg.norm(v) for v in box_vectors])
        if current_generation < 5000:
            mut_scale, max_d, b_ON, b_S, gamma_val = 0.20, 0.50, 0.30, 0.15, 2.0
        elif current_generation < 15000:
            mut_scale, max_d, b_ON, b_S, gamma_val = 0.12, 0.30, 0.20, 0.10, 1.0
        else:
            mut_scale, max_d, b_ON, b_S, gamma_val = 0.05, 0.10, 0.10, 0.05, 0.5
        mutated_coords = self._CSG_mutation(
            coords, elements, box_dims,
            mutation_scale=mut_scale,
            max_disp=max_d,
            beta_O_N=b_ON,
            beta_S=b_S,
            gamma=gamma_val,
            k_anchors=4)
        mutated_coords = np.mod(mutated_coords, box_dims)
        print(f"----- 正在修复变异后可能产生的原子重叠 ... -----")
        mutated_coords = self._resolve_overlaps(mutated_coords, elements, box_dims, overlap_factor=0.75, verbose=True)
        mutated = {
            'id': mutated_id,
            'positions': mutated_coords,    
            'fitness': 0.0,
            'energy': 0.0,
            'pdf': 0.0,
            'parent_ids': [original_individual['id']],
            'generation_method': 'Mutation'}
        print(f"----- 变异成功! 生成新个体 {mutated_id} -----")
        return mutated   
    
    def _CSG_mutation(self, parent, elements, box_dims, 
                    mutation_scale=0.50, max_disp=0.50, 
                    beta_O_N=0.20, beta_S=0.10, gamma=1.0, k_anchors=4):
        coords = np.asarray(parent, dtype=float)
        elements = np.asarray(elements)
        N = len(coords)
        s_parent = coords / box_dims   
        c_idx = np.flatnonzero(elements == 'C')
        o_idx = np.flatnonzero(elements == 'O')
        n_idx = np.flatnonzero(elements == 'N')
        s_idx = np.flatnonzero(elements == 'S')
        h_idx = np.flatnonzero(elements == 'H')
        heavy_idx = np.flatnonzero(elements != 'H')
        s_mutated = s_parent.copy()
        
        if len(c_idx) > 0:
            s_C = s_parent[c_idx]  
            pos_C_phys = s_C * box_dims 
            n_C = len(c_idx)  
            num_centers = max(1, min(5, n_C // 10)) 
            center_indices = np.random.choice(n_C, size=num_centers, replace=False) 
            centers = pos_C_phys[center_indices] 
            center_disps = (np.random.rand(num_centers, 3) - 0.5) * 2.0 * mutation_scale 
            tree_centers = cKDTree(centers, boxsize=box_dims) 
            dists, near_c_idx = tree_centers.query(pos_C_phys, k=num_centers) 
            r_decay = 8.0 
            field_weights = np.exp(-(dists ** 2) / (2.0 * (r_decay ** 2)))
            field_weights /= np.sum(field_weights, axis=1, keepdims=True)       
            target_disp_phys = np.sum(field_weights[:, :, None] * center_disps[near_c_idx], axis=1)
            s_C_target = s_C + (target_disp_phys / box_dims)
            diff_s = s_C_target[:, None, :] - s_C[None, :, :]
            diff_s = diff_s - np.round(diff_s)
            dist_matrix = np.linalg.norm(diff_s * box_dims, axis=-1)
            sigma_x_sq = 1.8 ** 2
            lambda_desc = 1.0
            s_target_full = s_parent.copy()
            s_target_full[c_idx] = s_C_target
            CN_parent = self._calculate_heavy_coordination_numbers(s_parent, heavy_idx, c_idx, box_dims, r_cut=1.85)
            CN_target = self._calculate_heavy_coordination_numbers(s_target_full, heavy_idx, c_idx, box_dims, r_cut=1.85)
            Cost_space = (dist_matrix ** 2) / sigma_x_sq
            Cost_chem = lambda_desc * ((CN_target[:, None] - CN_parent[None, :]) ** 2)
            Cost = Cost_space + Cost_chem
            a = np.ones(n_C) / n_C
            b = np.ones(n_C) / n_C
            Pi = ot.sinkhorn(a, b, Cost, reg=gamma, numItermax=1000)
            Pi_sum = np.sum(Pi, axis=1, keepdims=True)
            Pi_sum[Pi_sum == 0] = 1.0
            u_C = np.sum(Pi[:, :, None] * diff_s, axis=1) / Pi_sum
            u_C_phys = u_C * box_dims
            disp_norms = np.linalg.norm(u_C_phys, axis=-1, keepdims=True)
            disp_norms[disp_norms == 0] = 1.0
            scale_factor = np.minimum(1.0, max_disp / disp_norms)
            u_C = u_C * scale_factor
            s_mutated[c_idx] = (s_parent[c_idx] + u_C) % 1.0  
        
        hetero_configs = [(o_idx, beta_O_N), (n_idx, beta_O_N), (s_idx, beta_S)]
        if len(c_idx) >= 3:
            tree_c_parent = cKDTree(s_parent[c_idx] * box_dims, boxsize=box_dims) 
            coords_c_mutated = s_mutated[c_idx] * box_dims 
            for X_idx_list, beta_X in hetero_configs:
                if len(X_idx_list) == 0:
                    continue
                for idx in X_idx_list:
                    pos_X_parent = s_parent[idx] * box_dims
                    k_curr = min(k_anchors, len(c_idx))
                    _, anchor_local_indices = tree_c_parent.query(pos_X_parent, k=k_curr)
                    P_anchors = s_parent[c_idx[anchor_local_indices]] * box_dims
                    Q_anchors = coords_c_mutated[anchor_local_indices]
                    # 展平处理
                    P_shift = P_anchors - P_anchors[0]
                    P_shift = P_shift - np.round(P_shift / box_dims) * box_dims
                    P_anchors_cont = P_anchors[0] + P_shift
                    Q_shift = Q_anchors - Q_anchors[0]
                    Q_shift = Q_shift - np.round(Q_shift / box_dims) * box_dims
                    Q_anchors_cont = Q_anchors[0] + Q_shift
                    pos_X_shift = pos_X_parent - P_anchors[0]
                    pos_X_shift = pos_X_shift - np.round(pos_X_shift / box_dims) * box_dims
                    pos_X_cont = P_anchors[0] + pos_X_shift
                    R, t = self._kabsch_fit(P_anchors_cont, Q_anchors_cont)
                    pos_X_rigid = np.dot(pos_X_cont, R.T) + t
                    u_X_anchor = (pos_X_rigid - pos_X_parent) / box_dims
                    u_X_anchor = u_X_anchor - np.round(u_X_anchor)
                    u_X_noise = (np.random.rand(3) - 0.5) * 2.0 * (mutation_scale * 0.2) / box_dims
                    u_X_final = (1.0 - beta_X) * u_X_anchor + beta_X * u_X_noise
                    s_mutated[idx] = (s_parent[idx] + u_X_final) % 1.0   
        
        if len(h_idx) > 0 and len(heavy_idx) > 0:
            s_parent_H = s_parent[h_idx]
            s_parent_heavy = s_parent[heavy_idx]
            u_heavy_frac = s_mutated[heavy_idx] - s_parent_heavy
            u_heavy_frac = u_heavy_frac - np.round(u_heavy_frac)
            u_heavy_phys = u_heavy_frac * box_dims
            tree_heavy_parent = cKDTree(s_parent_heavy * box_dims, boxsize=box_dims)
            K_neighbors = min(4, len(heavy_idx))
            sigma = 1.5
            for i, h_i in enumerate(h_idx):
                pos_h_phys = s_parent_H[i] * box_dims
                dists, near_indices = tree_heavy_parent.query(pos_h_phys, k=K_neighbors)
                weights = np.exp(-(dists ** 2) / (2.0 * (sigma ** 2)))
                w_sum = np.sum(weights)
                if w_sum < 1e-12:
                    weights = np.ones_like(weights) / len(weights)
                else:
                    weights = weights / w_sum
                u_H_phys = np.sum(weights[:, None] * u_heavy_phys[near_indices], axis=0)
                s_mutated[h_i] = (s_parent[h_i] + u_H_phys / box_dims) % 1.0
        mutated_coords = s_mutated * box_dims
        return mutated_coords
    
    def _update_population_fitness(self, target_population):
        if not target_population:
            return
        pdf_errors = np.asarray([ind['pdf'] for ind in target_population],dtype=float)
        energies = np.asarray([ind['energy'] for ind in target_population],dtype=float)
        P_min, P_max = np.min(pdf_errors), np.max(pdf_errors)
        E_min, E_max = np.min(energies), np.max(energies)
        eps = 1e-8
        pdf_norm = (pdf_errors - P_min) / (P_max - P_min + eps)
        energy_norm = (energies - E_min) / (E_max - E_min + eps)
        w_pdf = 0.8
        w_energy = 0.2
        rho = w_pdf * pdf_norm + w_energy * energy_norm
        beta = 2.5
        fitness = 0.5 * (1.0 - np.tanh(beta * (2.0 * rho - 1.0)))
        for ind, fitness_i in zip(target_population, fitness):
            ind['fitness'] = float(fitness_i)
    
    def _wrap_delta(self,vec, box_dims):
        return vec - np.round(vec / box_dims) * box_dims
    
    def _get_pair_overlap_threshold(self, elem_i, elem_j, overlap_factor=0.75, default_bond_length=0.95):
        key = tuple(sorted((elem_i, elem_j)))
        bond_len = BOND_MIN_DISTANCES.get(key, default_bond_length)
        return bond_len * overlap_factor
    
    def _resolve_overlaps(self, coords, elements, box_dims, overlap_factor=0.75, verbose=True):
        coords = np.asarray(coords, dtype=float).copy()
        coords = np.mod(coords, box_dims)
        box_dims = np.asarray(box_dims, dtype=float)       
        if not np.all(np.isfinite(coords)):
            raise ValueError("OverlapFix收到非有限坐标，已跳过该候选结构")
        if not np.all(np.isfinite(box_dims)) or np.any(box_dims <= 0):
            raise ValueError("OverlapFix收到异常盒子尺寸")
        if len(coords) != len(elements):
            raise ValueError(f"坐标数量 ({len(coords)}) 与元素列表数量 ({len(elements)}) 不匹配！")
        all_pairs_thresholds = [self._get_pair_overlap_threshold(e1, e2, overlap_factor)
                                for e1 in set(elements) for e2 in set(elements)]
        max_search_radius = max(all_pairs_thresholds)      
        max_attempts = 50
        for attempt in range(max_attempts):
            tree = cKDTree(coords, boxsize=box_dims)
            candidate_pairs = tree.query_pairs(r=max_search_radius)
            actual_clashes = []
            for i, j in candidate_pairs:
                threshold = self._get_pair_overlap_threshold(elements[i], elements[j], overlap_factor)
                diff = self._wrap_delta(coords[i] - coords[j], box_dims)
                d_sq = np.sum(diff**2)
                if d_sq < (threshold ** 2):
                    actual_clashes.append((i, j, threshold, diff, np.sqrt(d_sq)))
            n_clash = len(actual_clashes)
            if n_clash == 0:
                if verbose:
                    print("所有元素对间距均大于各自的75%阈值，未检测到重叠！")
                break
            if verbose:
                print(f"原子重叠修正轮次{attempt+1}/{max_attempts} | 冲突对数={n_clash}")
            moved_pairs = 0
            for i, j, threshold, diff, d in actual_clashes:
                if d < 1e-6:
                    direction = np.random.rand(3) - 0.5
                    direction /= np.linalg.norm(direction)
                    d = 0.001
                else:
                    direction = diff / d
                push_mag = (threshold - d) / 2.0 + 0.05
                coords[i] += direction * push_mag
                coords[j] -= direction * push_mag
                moved_pairs += 1
            if verbose:
                print(f"本轮共调整{moved_pairs}对原子")
            coords = np.mod(coords, box_dims)
        else:
            if verbose:
                print("已达到最大修正轮次，局部仍可能有残余的轻微重叠")
        return coords
   
    def initialize_population(self, n_workers):
        print("=" * 80)
        print(f"{'':<20}******* 种群初始化阶段(Generation 0) *******{'':>20}")
        print("=" * 80)
        target_size = self.population_size
        population=[]
        total_attempt = 0
        max_total_attempt = target_size * 4
        while len(population) < target_size: 
            remaining = target_size - len(population)  
            payloads = []
            for i in range(remaining):  
                total_attempt += 1
                if total_attempt > max_total_attempt:
                    raise RuntimeError(f"初始化失败：尝试次数已用尽({max_total_attempt}次)")
                current_idx = len(population) + i + 1
                individual_id = f"G0_{current_idx:03d}_initial_{self._generate_random_suffix()}"
                perturbed_coords1 = self._create_chaotic_structure(GLOBAL_INITIAL_COORDS, GLOBAL_BOX_DIMS, current_idx, target_size)
                print(f"个体{individual_id}扰动成功！")
                perturbed_coords2 = self._resolve_overlaps(perturbed_coords1, GLOBAL_ELEMENTS, GLOBAL_BOX_DIMS, overlap_factor=0.75, verbose=True)
                payloads.append({
                    'id': individual_id,
                    'positions': perturbed_coords2,
                    'tmp_dir': os.path.join(self.base_output_dir, "tmp_workers"),
                    })
            results = self._evaluate_population_parallel(payloads, n_workers)
            for payload, res in zip(payloads, results):
                if len(population) >= target_size:
                    break
                if not res['success']:
                    print(f"警告: {res['id']}初始化失败，原因: {res.get('error', 'Unknown')[:30]}...")
                    continue    
                population.append({
                    'id': res['id'],
                    'positions': res['positions'],
                    'fitness': 0.0,
                    'pdf': res['pdf'],
                    'energy': res['energy'],
                    'sim_gr':res['sim_gr'],
                    'generation_method': 'initial',
                    'parent_ids': ['None']
                })
                print(f" 成功初始化个体: {res['id']}")
        self.population = population
        print("=" * 50)
        print(f" 初代种群初始化完成，共{len(self.population)}个个体")
        print("=" * 50)
    
    def run_optimization(self, n_workers=None):
        print(f"启动并行评估 Workers: {n_workers}，单批最长时间设置为: {_worker_timeout_seconds():.0f}s")
        Np = self.population_size   
        Nm = int(Np / 3)        
        n_elite_ratio = 0.1 
        num_elites = max(1, int(Np * n_elite_ratio))  
        n_mutation_ratio = 1 / 3
        Nx = max(1, int(Np * n_mutation_ratio))   
        print(f"参数设置 | Pop(Np):{Np}, Pairs(Nm):{Nm}->Child:{2*Nm}, Mutants(Nx):{Nx}, Elites:{num_elites}")        
        if self.start_gen == 0:
            self.initialize_population(n_workers=n_workers)
            print("\n 正在评估初代种群适应度...")
            self._update_population_fitness(self.population)           
            current_best = max(self.population, key=lambda ind: ind['fitness'])
            self.global_best = self._copy_individual_snapshot(current_best)           
            self.history_generation.append(0) 
            self.history_best_pdf.append(self.global_best['pdf'])    
            self.history_best_energy.append(self.global_best['energy'])
            current_pdfs = np.array([ind['pdf'] for ind in self.population])
            self.history_avg_pdf.append(np.mean(current_pdfs))
            self.history_std_pdf.append(np.std(current_pdfs))
            current_pdfs_list = [ind['pdf'] for ind in self.population] 
            self.history_all_pdfs.append(current_pdfs_list)
            current_energies_list = [ind['energy'] for ind in self.population]
            self.history_all_energies.append(current_energies_list)
            self._save_checkpoint(0)
            print("第0代(初始种群)的评估数据已成功加入统计历史！")        
        
        for generation in range(self.start_gen + 1, self.generations + 1):
            print(f"{'='*80}")
            print(f"{' ':25}***** Generation {generation}/{self.generations} *****{' ':25}")
            print(f"{'='*80}")
            is_mdhrmc_gen = (generation % self.md_hrmc_interval == 0)
            eval_mode = 'md_hrmc' if is_mdhrmc_gen else 'static'
            if is_mdhrmc_gen:
                print(f"【第 {generation} 代】触发 MD-HRMC 弛豫机制，本代所有个体将进行动态演化！")            
            offspring = []
            parents_pool = self.population[:]
            for i in range(Nm):
                parent1, parent2 = self.select_parents(parents_pool)
                if parent1['id'] == parent2['id']:
                    others = [p for p in parents_pool if p['id'] != parent1['id']]
                    if others:
                        parent2 = random.choice(others)
                c1, c2 = self.crossover(parent1, parent2, generation)
                offspring.append(c1)
                offspring.append(c2)
            print("------ 对offspring进行评估... ------")
            payloads = []
            print("为子代构建评估任务...")
            for child in offspring:
                payloads.append({
                    'id': child['id'],
                    'positions': child['positions'],
                    'tmp_dir': os.path.join(self.base_output_dir, "tmp_workers"),
                    'eval_mode': eval_mode
                    })
            print(f"子代payload总数={len(payloads)}，并行提交给worker！")
            results_list = self._evaluate_population_parallel(payloads, n_workers)  
            results_map = {res['id']: res for res in results_list if res is not None}
            valid_offspring = []
            for child in offspring:
                res = results_map.get(child['id'])
                if not res['success']:
                    continue 
                child['positions'] = res['positions']
                child['energy'] = res['energy']
                child['pdf'] = res['pdf']
                child['sim_gr'] = res['sim_gr']
                valid_offspring.append(child)
            print(f"子代处理完毕:成功模拟个体{len(valid_offspring)}/总数{len(payloads)}")   
            intermediate = self.population + valid_offspring
            print(f"中间种群构建完成，总数: {len(intermediate)}")
            self._update_population_fitness(intermediate)
            intermediate_sorted = sorted(intermediate, key=lambda x: x['fitness'], reverse=True)
            elite = intermediate_sorted[:num_elites]
            elite_ids = {ind['id'] for ind in elite}
            print(f"精英筛选完成:{len(elite)}个")           
            core_pool = intermediate_sorted[: int(len(intermediate_sorted) * 0.55)]
            diversity_pool = intermediate_sorted[int(len(intermediate_sorted) * 0.20): int(len(intermediate_sorted) * 0.85)]
            pool_for_mutation = list({ind['id']: ind for ind in (core_pool + diversity_pool)}.values())
            mutate_count = min(Nx, len(pool_for_mutation))
            mutate_candidates = random.sample(pool_for_mutation, mutate_count)
            print(f"从中间种群挑选{len(mutate_candidates)}个个体做变异")
            mutants = []
            for cand in mutate_candidates:
                try:
                    m = self.mutate(cand, generation)
                except Exception as exc:
                    print(f"主变异个体 {cand['id']} 生成失败，已跳过: {exc}")
                    continue
                mutants.append(m)
            print(f"变异完成，共生成{len(mutants)}个突变体")
            print("为突变体构建评估任务...")
            payloads = []
            for mut in mutants:
                payloads.append({
                    'id': mut['id'],
                    'positions': mut['positions'],
                    'tmp_dir': os.path.join(self.base_output_dir, "tmp_workers"),
                    'eval_mode': eval_mode
                    })
            print(f"突变体payload总数={len(payloads)}，提交给worker并行！")
            results_list = self._evaluate_population_parallel(payloads, n_workers)
            results_map = {res['id']: res for res in results_list if res is not None}
            valid_mutants = []
            for mut in mutants:
                res = results_map.get(mut['id'])
                if not res['success']:
                    continue
                mut['positions'] = res['positions']
                mut['energy'] = res['energy']
                mut['pdf'] = res['pdf']
                mut['sim_gr'] = res['sim_gr']
                valid_mutants.append(mut)
            print("-" * 85)
            if is_mdhrmc_gen:
                next_gen_population = valid_mutants
            else:
                next_gen_population = elite + valid_mutants
            remaining_slots = Np - len(next_gen_population)
            if remaining_slots > 0:
                print(f"需要从子代中补充{remaining_slots}个名额(强制优先valid_offspring)......")
                current_ids = set(ind['id'] for ind in next_gen_population)
                extras = []
                valid_offspring_sorted = sorted(valid_offspring, key=lambda x: x['fitness'], reverse=True)
                for ind in valid_offspring_sorted:
                    if len(extras) == remaining_slots:
                        break
                    if ind['id'] in current_ids:
                        continue
                    extras.append(ind)
                    print(f"     接纳新结构: {ind['id']}")
                if len(extras) < remaining_slots:
                    print(" 允许重复使用 valid_offspring 中的个体补足")
                    while len(extras) < remaining_slots:
                        ind = random.choice(valid_offspring)
                        extras.append(ind)
                next_gen_population += extras 
            print("更新下一代种群适应度...")
            self._update_population_fitness(next_gen_population)
            self.population = next_gen_population
            assert len(self.population) == Np, "种群数量异常！"
            current_best_individual = max(self.population, key=lambda ind: ind['fitness'])
            self.global_best = self._copy_individual_snapshot(current_best_individual)           
            current_best_id = current_best_individual['id']
            import glob
            old_best_files = glob.glob(os.path.join(self.base_output_dir, "Current_GlobalBest_Gen*"))
            for old_file in old_best_files:
                try:
                    os.remove(old_file)
                except Exception:
                    pass
            best_base_name = f"Current_GlobalBest_Gen{generation:04d}_{self.global_best['id']}"
            best_pdb_path = os.path.join(self.base_output_dir, best_base_name + ".pdb")
            self._write_pdb(best_pdb_path, self.global_best['positions'], GLOBAL_ELEMENTS, GLOBAL_BOX_DIMS)
            print(f"已动态更新全局最佳结构至: {best_base_name}.pdb")
            if self.global_best.get('sim_gr') is not None:
                best_png_path = os.path.join(self.base_output_dir, best_base_name + "_gr.png")
                try:
                    exp_data = np.loadtxt(PDF_DATA, comments="#")
                    plt.figure(figsize=(8, 5))
                    plt.plot(exp_data[:, 0], exp_data[:, 1], 'k-', label='Experimental g(r)', linewidth=1.8)
                    plt.plot(exp_data[:, 0], self.global_best['sim_gr'], 'r-', label='Simulated g(r)', linewidth=1.5, alpha=0.85)
                    plt.xlabel('r (Å)', fontsize=12)
                    plt.ylabel('g(r)', fontsize=12)
                    plt.title(f"Global Best g(r) Fitting - Generation {generation}\n"
                              f"Energy: {self.global_best['energy']} | $\\chi^2$ (PDF Error): {self.global_best['pdf']}", 
                              fontsize=14)
                    plt.legend(loc="upper right", fontsize=11)
                    plt.grid(True, linestyle='--', alpha=0.6)
                    plt.xlim(0.0, 10.0)  
                    plt.tight_layout()
                    plt.savefig(best_png_path, dpi=250)
                    plt.close()
                    print(f"已动态更新全局最佳g(r)拟合图至: {best_base_name}_gr.png")
                except Exception as e:
                    print(f" 绘制 g(r) 图失败: {e}")
                    plt.close('all')
            current_pdfs = [ind['pdf'] for ind in self.population]
            current_energies = [ind['energy'] for ind in self.population]
            self.history_generation.append(generation)
            self.history_best_pdf.append(self.global_best['pdf'])    
            self.history_best_energy.append(self.global_best['energy'])
            self.history_avg_pdf.append(np.mean(current_pdfs))  
            self.history_std_pdf.append(np.std(current_pdfs)) 
            self.history_all_pdfs.append(current_pdfs)
            self.history_all_energies.append(current_energies)        
            self._save_checkpoint(generation)
        print("遗传算法演化完成！")
        best_individual = self.global_best
        print(f"全局最优个体 ID: {best_individual['id']}")
        print(f"最终 Energy:  {best_individual['energy']}")
        print(f"最终 PDF误差: {best_individual['pdf']}")
        return best_individual

if __name__ == "__main__":
    optimizer = GA_MDHRMC_Optimizer(
        population_size=100,
        generations=20000,
        md_hrmc_interval=1000,
        checkpoint_path=None)
    best_individual= optimizer.run_optimization(n_workers=16)