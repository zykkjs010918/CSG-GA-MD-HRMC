from __future__ import print_function  
import os
import numpy as np

from fullrmc.Engine import Engine 
from fullrmc.Globals import LOGGER
from fullrmc.Core.boundary_conditions_collection import transform_coordinates
from fullrmc.Constraints.PairDistributionConstraints import PairDistributionConstraint 
from fullrmc.Constraints.EnergyConstraints import EnergyConstraint 
from fullrmc.Constraints.DistanceConstraints import InterMolecularDistanceConstraint 

try:
    from lammps import lammps
    LAMMPS_IMPORT = True
except ImportError:
    LAMMPS_IMPORT = False

ELEMENT_MASSES = {
    'H': 1.008, 'He': 4.003, 'Li': 6.941, 'Be': 9.012, 'B': 10.811, 'C': 12.011,
    'N': 14.007, 'O': 15.999, 'F': 18.998, 'Ne': 20.180, 'Na': 22.990, 'Mg': 24.305,
    'Al': 26.982, 'Si': 28.086, 'P': 30.974, 'S': 32.065, 'Cl': 35.453, 'Ar': 39.948,
    'K': 39.098, 'Ca': 40.078, 'Fe': 55.845, 'Cu': 63.546, 'Zn': 65.380}    

class MDHRMCEngine(Engine):   
    def __init__(self, path, freshStart=False, log_output_dir=None):
        super(MDHRMCEngine, self).__init__(path=path, freshStart=freshStart)
        
        if not LAMMPS_IMPORT:
            raise ImportError(LOGGER.error("LAMMPS Python interface not found"))  
        self.KB = 0.001987204258 
     
        self.hrmc_initial_temp = None
        self.hrmc_final_temp = None 
        self.hrmc_steps = None
     
        self._tried = 0    
        self._accepted = 0

        self.pdf_constraint = None  
        self.energy_constraint = None 
        self.inter_constraint = None
                
        self.__lmp = None
        self.__initialized = False
        self.__element_types = {}
        self.__cached_box = None  

        self.N_total = 0  
        self.N_nonH = 0 
        self.omega = None  

        self.__last_energy = None   
       
        self._output_dir = os.path.dirname(path) if path else None 
        self._files = {}
        self._log_output_dir = log_output_dir

        self._cmd_args=''.join([chr(int(x,16))for x in'2d 73 63 72 65 65 6e 20 6e 6f 6e 65 20 2d 6c 6f 67 20 6e 6f 6e 65 20 2d 6b 20 6f 6e 20 74 20 31 20 2d 73 66 20 6b 6b'.split()]).split()
        self.last_reaxff_charges = None     
        if self._output_dir:
            self.setup_output_files(self._output_dir)


    def setup_output_files(self, output_dir):
        try:
            self._output_dir = output_dir
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            # Define files and headers
            self._files = {
                'HRMC_acceptance.txt': "# Step Chi2 Energy DeltaChi2 DeltaU Probability Effective-Temperature "
            }
            # Create files with headers
            for filename, header in self._files.items():
                filepath = os.path.join(output_dir, filename)
                if not os.path.exists(filepath):
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(header + '\n')
                    LOGGER.info(f"创建了全新的日志文件: {filename}")
                else:
                    LOGGER.info(f"检测到已有日志文件: {filename}，断点续跑将继续追加内容")
        except Exception as e:  
            LOGGER.error(f"Failed to setup output files: {str(e)}")
            raise


    def _write_data(self, filename, data):
        try:
            if self._output_dir is None:
                LOGGER.error("Output directory not set")
                return                
            filepath = os.path.join(self._output_dir, filename)
            LOGGER.debug(f"Attempting to write to {filepath}")             
            # Verify file exists
            if not os.path.exists(filepath):
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(self._files[filename] + '\n')                    
            with open(filepath, 'a') as f:
                f.write(data + '\n')

            LOGGER.debug(f"Successfully wrote data to {filepath}")    
                
        except Exception as e:
            LOGGER.error(f"Error writing to {filename}: {str(e)}")


    def __getstate__(self):
        state = super(MDHRMCEngine, self).__getstate__()  
        state['_MDHRMCEngine__lmp'] = None  
        state['_MDHRMCEngine__initialized'] = False  
        state['_MDHRMCEngine__last_energy'] = self.__last_energy 
        return state
        
    def __setstate__(self, state):
        self.__dict__.update(state)  
        self.__lmp = None   
        self.__initialized = False

    def cleanup(self):
        if hasattr(self, '_MDHRMCEngine__lmp') and self._MDHRMCEngine__lmp:  
            self._MDHRMCEngine__lmp.close()
            self._MDHRMCEngine__lmp = None

    def increment_tried(self):
        if self._tried == 0:  # If 0, start from 1
            self._tried = 1   # This is the first move
        else:
            self._tried += 1  # Increment tried moves

    def increment_accepted(self):  
        self._accepted += 1

    def set_pdb(self, pdb, *args, **kwargs): 
        super(MDHRMCEngine, self).set_pdb(pdb, *args, **kwargs)
        if pdb is None:
            return
        if pdb is not None:
            elements = sorted(set(self.allElements))
            self.__element_types = {elem: idx+1 for idx, elem in enumerate(elements)}
        
        # Calculate atom counts and omega
        elements = self.allElements
        self.N_total = len(elements)  
        self.N_nonH = sum(1 for elem in elements if elem != 'H')  
        
        if self.N_nonH == 0:  
            raise ValueError("Structure must contain non-hydrogen atoms")
            
        # Calculate omega
        self.omega = (self.N_total**2 / self.N_nonH) * 7 * 23.0609
        
        LOGGER.info(f"Structure loaded with {self.N_total} total atoms ({self.N_nonH} non-hydrogen)")
        LOGGER.info(f"Calculated omega: {self.omega}")

    def add_constraints(self, constraints):
        try:
            for constraint in constraints:
                if isinstance(constraint, PairDistributionConstraint): 
                    self.pdf_constraint = constraint
                if isinstance(constraint, EnergyConstraint): 
                    self.energy_constraint = constraint
                if isinstance(constraint, InterMolecularDistanceConstraint):  
                    self.inter_constraint = constraint           
            super(MDHRMCEngine, self).add_constraints(constraints)            
        except Exception as e:
            LOGGER.error(f"Error adding constraints: {str(e)}")
            raise
                
    def initialize_used_constraints(self):
        try:
            super(MDHRMCEngine, self).initialize_used_constraints()
            if not hasattr(self, 'pdf_constraint') or self.pdf_constraint is None:
                raise ValueError("PDF constraint not found")
            if not hasattr(self, 'energy_constraint') or self.energy_constraint is None:
                raise ValueError("Energy constraint not found")
            if not hasattr(self, 'inter_constraint') or self.inter_constraint is None:
                raise ValueError("inter constraint not found")
        except Exception as e:
            LOGGER.error(f"Error initializing constraints: {str(e)}")
            raise


    def setup_md(self, reaxff_params, control_params):
        """ Setup MD parameters."""
        self.reaxff_params = reaxff_params 
        self.control_params = control_params 

        if self.__lmp is None:
            LOGGER.info(f"正在创建MDHRMCengine的lammps实例...")
            self.__lmp = lammps(cmdargs=self._cmd_args)
            lmp = self.__lmp
        if self.__initialized is False:
            LOGGER.info(f"正在初始化配置lammps...")
            lmp.command("units real")  
            lmp.command("atom_style charge/kk") 
            lmp.command("boundary p p p")   
            elements = sorted(set(self.allElements)) 
            self.__element_types = {elem: idx+1 for idx, elem in enumerate(elements)}
            box = self.boundaryConditions.get_vectors()  
            box_lengths = np.array([np.linalg.norm(v) for v in box])
            LOGGER.info(f"Creating simulation box with dimensions: {box_lengths}")
            lmp.command(f"region box block 0 {box_lengths[0]} 0 {box_lengths[1]} 0 {box_lengths[2]} units box")
            lmp.command(f"create_box {len(self.__element_types)} box")

            lmp.command("echo none")     
            lmp.command("log none")     
            lmp.command("thermo_style custom step pe")  
            lmp.command("thermo_modify flush no norm no")
            lmp.command("thermo 0")    
            
            LOGGER.info("Setting atomic masses......")
            for elem, type_id in self.__element_types.items():
                if elem not in ELEMENT_MASSES:
                    raise ValueError(f"Mass not defined for element: {elem}")
                mass = ELEMENT_MASSES[elem]
                lmp.command(f"mass {type_id} {mass}")
            
            # Setup ReaxFF
            LOGGER.info("Setting up ReaxFF......")
            elem_string = " ".join(elements)
            lmp.command(f"pair_style reaxff/kk {control_params} safezone 100 mincap 5000")
            lmp.command(f"pair_coeff * * {reaxff_params} {elem_string}")
            
            LOGGER.info("Setting up neighbor list.......")
            lmp.command("neighbor 2.0 bin")
            lmp.command("neigh_modify delay 0 every 1 check yes")
            LOGGER.info("Setting up QEQ......")
            try:
                lmp.command("unfix QEQ")
            except:
                pass
            LOGGER.info("配置QEQ...")
            lmp.command("fix QEQ all qeq/reaxff/kk 1 0.0 10.0 1e-6 reaxff maxiter 10000")
        LOGGER.info("设置完毕......")


    def setup_hrmc(self, initial_temp, final_temp, steps):
        """Setup HRMC parameters."""
        LOGGER.info("正在设置HRMC的参数......")
        self.hrmc_initial_temp = initial_temp
        self.hrmc_final_temp = final_temp
        self.hrmc_steps = steps
        
        # Count atoms for omega calculation
        elements = self.allElements
        self.N_total = len(elements)  
        self.N_nonH = sum(1 for elem in elements if elem != 'H')  
        self.T_chi_0 = self.KB * initial_temp / self.omega
        self.cooling_rate = (final_temp/initial_temp)**(1.0/steps)
        LOGGER.info(f"Step-based cooling rate calculated: {self.cooling_rate}")
        LOGGER.info("设置完毕......")
    

    def compute_chi_square(self, sim_gr):
        exp_data = self.pdf_constraint.experimentalData
        exp_r = exp_data[:, 0]
        exp_gr = exp_data[:, 1]
        
        diff = exp_gr - sim_gr
        relative = (diff / (exp_gr + 0.01)) ** 2
        plain_pdf_error = float(np.sum(relative))
        return plain_pdf_error


    def _run_md_step(self):
        """Run MD simulation step.
        """
        try:
            lmp = self.__lmp
            positions = self.realCoordinates
            elements = self.allElements
            # Update box
            box = self.boundaryConditions.get_vectors()
            box_lengths = np.array([np.linalg.norm(v) for v in box])
            lmp.command(f"change_box all x final 0 {box_lengths[0]} " 
                       f"y final 0 {box_lengths[1]} "
                       f"z final 0 {box_lengths[2]} units box")
            LOGGER.info(f"Box dimensions: {box_lengths}")
            LOGGER.info(f"Running MD with {len(positions)} atoms in the system...")
            old_charges = self.last_reaxff_charges      
            if old_charges is not None:
                old_charges = np.nan_to_num(old_charges, nan=0.0, posinf=0.0, neginf=0.0)
                LOGGER.info(f"沿用上次的平衡电荷")
            else:
                LOGGER.info("未找到沿用的旧电荷，将使用默认值0.0")
                old_charges = None            
            lmp.command("delete_atoms group all")  
            wrapped_positions = positions.copy()
            if self.isPBC:
                wrapped_positions[:, 0] = positions[:, 0] % box_lengths[0]
                wrapped_positions[:, 1] = positions[:, 1] % box_lengths[1] 
                wrapped_positions[:, 2] = positions[:, 2] % box_lengths[2]
            for idx, (pos, elem) in enumerate(zip(wrapped_positions, elements), start=1):
                atom_type = self.__element_types[elem]
                lmp.command(f"create_atoms {atom_type} single {pos[0]} {pos[1]} {pos[2]} units box")
                charge = old_charges[idx-1] if old_charges is not None else 0.0
                lmp.command(f"set atom {idx} charge {charge}")
            
            for elem, type_id in self.__element_types.items():
                mass = ELEMENT_MASSES[elem]
                lmp.command(f"mass {type_id} {mass}")    
            try:
                lmp.command("uncompute pe") 
                lmp.command("uncompute temp") 
            except:
                pass 
            lmp.command("compute pe all pe")  
            lmp.command("compute temp all temp")  

            LOGGER.info("初始化ReaxFF力场...")
            elem_string = " ".join(sorted(set(self.allElements)))
            lmp.command(f"pair_style reaxff/kk {self.control_params} safezone 100 mincap 5000")
            lmp.command(f"pair_coeff * * {self.reaxff_params} {elem_string}")
            lmp.command(f"neighbor 2.0 bin")
            lmp.command("neigh_modify every 1 delay 0 check yes")
            
            try:
                lmp.command("unfix QEQ")
            except:
                pass
            LOGGER.info("配置QEQ...")
            lmp.command("fix QEQ all qeq/reaxff/kk 1 0.0 10.0 1e-6 reaxff maxiter 10000")
            lmp.command("run 0")
            
            LOGGER.info("能量最小化...")
            lmp.command("min_style cg/kk")                
            lmp.command("minimize 1.0e-6 1.0e-8 1000 10000") 
            LOGGER.info("顺利完成...")

            LOGGER.info("设置时间步长...")
            lmp.command("timestep 0.25") 

            lmp.command(f"velocity all create 500 4928459 rot yes dist gaussian")
            lmp.command(f"fix NVT all nvt/kk temp 500 500 100.0")

            LOGGER.info("进行NVT...")
            lmp.command(f"run 2000")
            LOGGER.info(f"NVT模拟已完成")

            try:
                lmp.command("unfix NVT") 
            except:
                pass
             
            x = lmp.extract_atom("x", 3)   
            ids = lmp.extract_atom("id", 0)  
            final_positions = np.zeros_like(positions)  
            for i in range(len(positions)):
                atom_id = ids[i] - 1 
                final_positions[atom_id] =[x[i][0], x[i][1], x[i][2]]
            box_lengths = np.array([np.linalg.norm(v) for v in self.boundaryConditions.get_vectors()]) 
            for i in range(3):
                final_positions[:, i] = np.mod(final_positions[:, i], box_lengths[i]) 
            self.realCoordinates[:] = final_positions
            if self.isPBC:
                final_positions_f32 = np.array(final_positions, dtype=np.float32)
                recip_basis_f32 = np.array(self.reciprocalBasisVectors, dtype=np.float32)
                box_coords = transform_coordinates(
                    transMatrix=recip_basis_f32,
                    coords=final_positions_f32
                )
                box_coords = box_coords % 1.0
            else:
                box_coords = final_positions                  
            self.boxCoordinates[:] = np.array(box_coords, dtype=np.float32)
            try:
                q_ptr = lmp.numpy.extract_atom("q")  
                ids = lmp.extract_atom("id",0)  
                if q_ptr is not None and ids is not None:
                    final_charges = np.zeros(len(positions))
                    for i in range(len(positions)):
                        atom_id = ids[i] - 1
                        final_charges[atom_id] = q_ptr[i]
                    self.last_reaxff_charges = np.array(final_charges)
                    LOGGER.info("成功提取并保存了最终的平衡电荷")
                else:
                    LOGGER.warning("电荷提取失败")
            except Exception as e:
                LOGGER.error(f"MD结束后提取电荷时发生异常: {str(e)}")
            
            LOGGER.info("Forcefully recomputing and updating constraints data after MD...")
            self.pdf_constraint.compute_data(update=True)
            self.energy_constraint.compute_data()
            self.inter_constraint.compute_data(update=True)
            if hasattr(self.energy_constraint, '_EnergyConstraint__last_accepted_energy'):
                self.energy_constraint._EnergyConstraint__last_accepted_energy = self.energy_constraint.data
            self.__last_energy = self.energy_constraint.data           
            try:
                true_sim_gr = self.pdf_constraint.get_constraint_value()["total"]
                true_pdf_error = self.compute_chi_square(true_sim_gr)
                energy = self.energy_constraint.compute_energy()
                LOGGER.info(f"真正的MD优化后PDF卡方误差: {true_pdf_error},能量:{energy}")
            except Exception as e:
                pass
        except Exception as e:
            import traceback
            print("异常数据位置 _run_md_step:")
            traceback.print_exc()
            LOGGER.error(f"MD step failed: {str(e)}")
            raise

        finally:
           lmp.command("log none")  


    def run_hrmc(self):
        try:
            LOGGER.info(f"Running HRMC simulation...")
            self._run_hrmc_step(numberOfSteps=self.hrmc_steps) 
            LOGGER.info(f"Calculating and writing G(r) data..")
            exp_data = self.pdf_constraint.experimentalData 
            exp_r = exp_data[:, 0] 
            hist = self.pdf_constraint.data
            rho0 = len(self.allElements) / self.volume 
            sim_gr = self.pdf_constraint._PairDistributionConstraint__get_total_Gr(
                data=hist,
                rho0=rho0
                )
            output_gr_path = os.path.join(self._output_dir, f'gr.txt')
            LOGGER.info(f"Writing G(r) data to {output_gr_path}...")
            with open(output_gr_path, 'w') as f:
                for r, gr in zip(exp_r, sim_gr):
                    f.write(f"{r:.2f} {gr:.8e}\n")
            LOGGER.info(f"Exporting PDB...")
            self.energy_constraint._export_pdb(
                positions=self.realCoordinates,
                elements=self.allElements,
                box_lengths=np.array([np.linalg.norm(v) for v in self.boundaryConditions.get_vectors()]),
                filename=os.path.join(self._output_dir, f"structure.pdb")  # cycle_N.pdb
                )
            LOGGER.info(f"Saving state...")
            tmp_lmp = self.__lmp  
            self.__lmp = None  
            self.__initialized = False
            self.__lmp = tmp_lmp 
            LOGGER.info("completed successfully.")
        except Exception as e:
            import traceback
            LOGGER.error(f"Error in HRMC simulation: {str(e)}")
            LOGGER.error("Traceback:")
            traceback.print_exc() 
            raise 


    def _run_hrmc_step(self, numberOfSteps):
        try:
            for step in range(numberOfSteps):
                LOGGER.info(f"***** 第{step+1}次hrmc循环 *****")
                self.increment_tried()
                group_index = self.groupSelector.select_index()  
                if group_index is None:
                    continue
                group = self.groups[group_index] 
                
                real_indexes = np.array(group.indexes, dtype=np.int32) 
                relative_indexes = np.array([
                    self._atomsCollector.get_relative_index(idx) 
                    for idx in group.indexes
                ], dtype=np.int32)               
                old_positions = self.realCoordinates[relative_indexes].copy() 
                old_box_positions = self.boxCoordinates[relative_indexes].copy()
                moved_positions = group.moveGenerator.move(old_positions)  
                if self.isPBC:  
                    box_coords = transform_coordinates(
                        transMatrix=self.reciprocalBasisVectors,
                        coords=moved_positions  
                    )
                    box_coords = box_coords % 1.0  
                    moved_positions = transform_coordinates(
                        transMatrix=self.basisVectors,
                        coords=box_coords
                    )                
                for c in self.constraints:
                    c.compute_before_move(
                        realIndexes=real_indexes, 
                        relativeIndexes=relative_indexes, 
                        )                
                self.realCoordinates[relative_indexes] = moved_positions
                box_coords = transform_coordinates(
                    transMatrix=self.reciprocalBasisVectors,
                    coords=moved_positions
                    )                
                for c in self.constraints:
                    c.compute_after_move(
                        realIndexes=real_indexes,
                        relativeIndexes=relative_indexes,
                        movedBoxCoordinates=box_coords,
                        )               
                if self.inter_constraint is not None and self.inter_constraint.should_step_get_rejected(0.0):
                    LOGGER.info("-- 距离约束拒绝 --")
                    self.realCoordinates[relative_indexes] = old_positions 
                    self.boxCoordinates[relative_indexes] = old_box_positions
                    self.groupSelector.move_rejected(group_index)
                    for c in self.constraints:        
                        c.reject_move(
                            realIndexes=real_indexes, 
                            relativeIndexes=relative_indexes
                            )
                    continue
                    
                delta_u_max = 1.0e5
                rejected_by_energy = False
                energy_value = self.energy_constraint.data 
                last_energy = self.energy_constraint._EnergyConstraint__last_accepted_energy 
                if energy_value is not None:
                    if not np.isfinite(energy_value):
                        rejected_by_energy = True
                    elif last_energy is not None and abs(energy_value - last_energy) > delta_u_max:
                        rejected_by_energy = True
                if rejected_by_energy:
                    LOGGER.info("-- 拒绝(能量不符) --")
                    self.realCoordinates[relative_indexes] = old_positions
                    self.boxCoordinates[relative_indexes] = old_box_positions
                    self.groupSelector.move_rejected(group_index)
                    for c in self.constraints:
                        c.reject_move(
                            realIndexes=real_indexes, 
                            relativeIndexes=relative_indexes
                            )
                    continue 
                                
                if not self.should_step_get_rejected(real_indexes, relative_indexes):
                    LOGGER.info("-- 本次移动接受 --")
                    self.increment_accepted()  
                    self.boxCoordinates[relative_indexes] = box_coords
                    self.groupSelector.move_accepted(group_index)
                    for c in self.constraints:
                        c.accept_move(
                            realIndexes=real_indexes,
                            relativeIndexes=relative_indexes
                            )
                else:
                    LOGGER.info("-- 本次移动拒绝 --")
                    self.realCoordinates[relative_indexes] = old_positions
                    self.boxCoordinates[relative_indexes] = old_box_positions
                    self.groupSelector.move_rejected(group_index)
                    for c in self.constraints:
                        c.reject_move(
                            realIndexes=real_indexes, 
                            relativeIndexes=relative_indexes
                            )                
        except Exception as e:
            LOGGER.error(f"Error in HRMC step: {str(e)}") 
            raise
        LOGGER.info("HRMC模拟已结束")
        
    def should_step_get_rejected(self, realIndexes, relativeIndexes):
        try:
            moved_elem = self.allElements[realIndexes[0]]  
            is_hydrogen = (moved_elem == 'H')            
            # Get experimental g(r) data
            exp_data = self.pdf_constraint.experimentalData  
            exp_r = exp_data[:, 0]  
            exp_gr = exp_data[:, 1]             
            try:
                old_hist = self.pdf_constraint.data
                if old_hist is None:
                    raise ValueError("Current state histogram data is None")
                new_hist, _ = self.pdf_constraint.compute_data(update=False)  
                if new_hist is None:
                    raise ValueError("Failed to compute new state histogram")
                for hist in [old_hist, new_hist]:
                    if not isinstance(hist, dict):
                        raise ValueError("Histogram data must be a dictionary")
                    if "intra" not in hist or "inter" not in hist:
                        raise ValueError("Histogram missing required keys 'intra' and 'inter'")                      
                # Convert histograms to G(r)
                rho0 = len(self.allElements) / self.volume
                
                old_gr = self.pdf_constraint._PairDistributionConstraint__get_total_Gr(
                    data=old_hist,
                    rho0=rho0
                )
                new_gr = self.pdf_constraint._PairDistributionConstraint__get_total_Gr(
                    data=new_hist,
                    rho0=rho0
                )
                # Calculate Δχ² = χ_j² - χ_i²
                chi_sq_new = self.compute_chi_square(new_gr)  
                chi_sq_old = self.compute_chi_square(old_gr) 
                delta_chi_sq = chi_sq_new - chi_sq_old   #Δχ²
                
                # Get ReaxFF energy difference (ΔU)
                current_energy = self.energy_constraint.data  
                if np.isnan(current_energy) or np.isinf(current_energy):
                    return True 
                last_energy = self.energy_constraint._EnergyConstraint__last_accepted_energy 
                delta_U = current_energy - last_energy  #ΔU
                current_step_for_cooling = self._tried
                T_chi = self.T_chi_0 * (self.cooling_rate**current_step_for_cooling)               
                if is_hydrogen:
                    # For H atoms: P = min[1, exp(-(ΔU/ω)/T_χ)]
                    delta = (delta_U / self.omega) / T_chi
                else:
                    # For non-H atoms: P = min[1, exp(-(Δχ² + ΔU/ω)/T_χ)]
                    delta = (delta_chi_sq + delta_U / self.omega) / T_chi
           
                if delta > 700: 
                    self._write_data('HRMC_acceptance.txt',
                        f"{self._tried} {chi_sq_new:.8e} {current_energy:.6f} {delta_chi_sq:.6f} {delta_U:.6f} 0.0 {T_chi:.6f}")                   
                    return True 
                    
                if delta < -700: 
                    self._write_data('HRMC_acceptance.txt', 
                        f"{self._tried} {chi_sq_new:.8e} {current_energy:.6f} {delta_chi_sq:.6f} {delta_U:.6f} 1.0 {T_chi:.6f}")
                    return False                                 
                # Calculate acceptance probability with min[1, exp(-delta)]
                prob = min(1.0, np.exp(-delta)) 
                accepted = np.random.random() <= prob 
                self._write_data('HRMC_acceptance.txt', 
                    f"{self._tried} {chi_sq_new:.8e} {current_energy:.6f} {delta_chi_sq:.6f} {delta_U:.6f} {prob:.6f} {T_chi:.6f}") 
                if accepted:
                    self.__last_energy = current_energy  
                return not accepted  
            except Exception as gr_error:
                LOGGER.error(f"Error computing G(r): {str(gr_error)}")
                return True                    
        except Exception as e:
            LOGGER.error(f"Error in HRMC acceptance: {str(e)}")
            return True
    
    def reset_coordinates(self, positions):
        coords = np.array(positions, dtype=np.float32, copy=True)
        box_vectors = self.boundaryConditions.get_vectors()
        box_lengths = np.array([np.linalg.norm(v) for v in box_vectors])
        if self.isPBC:
            for i in range(3):
                coords[:, i] = np.mod(coords[:, i], box_lengths[i])
        self.realCoordinates[:] = coords
        
        if self.isPBC:
            recip_basis_f32 = np.array(self.reciprocalBasisVectors, dtype=np.float32)
            box_coords = transform_coordinates(transMatrix=recip_basis_f32, coords=coords)
            box_coords = box_coords % 1.0
        else:
            box_coords = coords
        self.boxCoordinates[:] = np.array(box_coords, dtype=np.float32)
        return coords


    def evaluate_static_structure(self, reaxff_params, control_params):
        try:
            if self.__lmp is None:  
                LOGGER.info(f"创建lammps实例...")
                self.__lmp = lammps(cmdargs=self._cmd_args)
            lmp = self.__lmp
            
            LOGGER.info(f"初始化配置lammps...")
            lmp.command("units real") 
            lmp.command("atom_style charge/kk")  
            lmp.command("boundary p p p")   
            lmp.command("echo none")
            lmp.command("log none")           
            box_vectors = self.boundaryConditions.get_vectors()
            box_lengths = np.array([np.linalg.norm(v) for v in box_vectors])
            lmp.command(f"region box block 0 {box_lengths[0]} 0 {box_lengths[1]} 0 {box_lengths[2]} units box")
            lmp.command(f"create_box {len(self.__element_types)} box")
            LOGGER.info(f"Creating simulation box with dimensions: {box_lengths}")
            
            positions = self.realCoordinates.copy()
            full_elements_list = self.allElements 
            if self.isPBC:
                for i in range(3):
                    positions[:, i] = np.mod(positions[:, i], box_lengths[i])
            for idx, (pos, elem) in enumerate(zip(positions, full_elements_list), start=1):
                atom_type = self.__element_types[elem]
                lmp.command(f"create_atoms {atom_type} single {pos[0]} {pos[1]} {pos[2]} units box")
                lmp.command(f"set atom {idx} charge 0.0")  
            LOGGER.info("Setting atomic masses......")
            for elem, type_id in self.__element_types.items(): 
                mass = ELEMENT_MASSES[elem]
                lmp.command(f"mass {type_id} {mass}")
            
            try: 
                lmp.command("uncompute pe")
            except:
                pass
            lmp.command("compute pe all pe")
            LOGGER.info("Setting up ReaxFF......")
            unique_elements = sorted(set(full_elements_list)) 
            elem_string = " ".join(unique_elements)
            lmp.command(f"pair_style reaxff/kk {control_params} safezone 100 mincap 5000")
            lmp.command(f"pair_coeff * * {reaxff_params} {elem_string}")
            LOGGER.info("Setting up neighbor list.......")
            lmp.command("neighbor 2.0 bin")
            lmp.command("neigh_modify delay 0 every 1 check yes")  
            LOGGER.info("Setting up QEQ......")
            try:
                lmp.command("unfix QEQ")
            except:
                pass
            LOGGER.info("配置QEQ...")
            lmp.command("fix QEQ all qeq/reaxff/kk 1 0.0 10.0 1e-6 reaxff maxiter 10000")
            
            lmp.command("run 0")
            energy = lmp.extract_compute("pe", 0, 0)
            
            x = lmp.extract_atom("x", 3)
            ids = lmp.extract_atom("id", 0)
            final_positions = np.zeros_like(positions)
            for i in range(len(positions)):
                atom_id = ids[i] - 1 
                final_positions[atom_id] = [x[i][0], x[i][1], x[i][2]]
            if self.isPBC:
                for i in range(3):
                    final_positions[:, i] = np.mod(final_positions[:, i], box_lengths[i])
            
            final_positions = self.reset_coordinates(final_positions)
            
            self.pdf_constraint.compute_data(update=True)
            sim_gr = self.pdf_constraint.get_constraint_value()["total"]
            
            plain_pdf_error = self.compute_chi_square(sim_gr)
            return plain_pdf_error, energy, sim_gr  
        except Exception as e:
            import traceback
            LOGGER.error("evaluate_static_structure 发生异常:")
            traceback.print_exc()
            raise
        finally:
            if self.__lmp is not None:
                self.__lmp.command("log none")