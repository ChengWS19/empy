import os
import numpy as np
import empymod
import time
from multiprocessing import Pool, cpu_count
from walktem import TEMCalculator

class AppResCalculator:
    """
    Class to calculate apparent resistivity using empymod.

    Attributes:
        inputfile (str): Path to the input file.
        src (list): Source coordinates [x0, y0, z0, x1, y1, z1].
        rec (list): Receiver coordinates [x0, y0, z0, x1, y1, z1].
        depth (list): Depth interval for computation [zmin, zmax].
        res_list (list): List of resistivities.
        time_channel (ndarray): Time array.
        search_value (ndarray): Search values.
        RHO_min (float): Minimum resistivity value.
        RHO_max (float): Maximum resistivity value.
        parameters (dict): Dictionary of empymod parameters.
        flag (int): Flag for extreme resistivity search.
        RHO_A (ndarray): Array to store calculated apparent resistivity values.
        RHO_A1 (ndarray): Array to store calculated RHO_A1 values.
        RHO_A2 (ndarray): Array to store calculated RHO_A2 values.
    """

    def __init__(
        self,
        inputfile,
        src,
        rec,
        depth,        
        res_list,
        mrec,
        srcpts,
        strength,
        signal,
        caltype,
        waveflag,
        verb,
        time_channel,
        search_value,
        RHO_min,
        RHO_max,
    ):
        """
        Initialize the MyAppResCalculator instance.

        Args:
            inputfile (str): Path to the input file.
            src (list): Source coordinates [x0, y0, z0, x1, y1, z1].
            rec (list): Receiver coordinates [x0, y0, z0, x1, y1, z1].
            depth (list): Depth interval for computation [zmin, zmax].
            res_list (list): List of resistivities.
            time_channel (ndarray): Time array.
            search_value (ndarray): Search values.
            RHO_min (float, optional): Minimum resistivity value. Defaults to 1.
            RHO_max (float, optional): Maximum resistivity value. Defaults to 1000.
        """
        self.inputfile = inputfile
        self.src = src
        self.rec = rec
        self.depth = depth        
        self.res_list = res_list
        self.mrec = mrec
        self.srcpts = srcpts
        self.strength = strength
        self.signal = signal
        self.caltype = caltype
        self.waveflag = waveflag
        self.verb = verb  
        
        self.time_channel = time_channel    
        self.search_value = search_value     
        self.RHO_min = RHO_min
        self.RHO_max = RHO_max  
        self.TEM = TEMCalculator()

        self.flag = 0   # 是否单调标识
        self.RHO_A = np.zeros_like(self.time_channel)
        self.RHO_A1 = np.zeros_like(self.time_channel)
        self.RHO_A2 = np.zeros_like(self.time_channel)
        self.output_filename = os.path.basename(inputfile)
        self.output_filename = os.path.splitext(self.output_filename)[0]

    def linearSearch(self, rho, time_point, field_value):
        """
        Perform linear search for resistivity value.

        Args:
            rho (float): Resistivity value.
            time_point (float): Time value.
            field_value (float): Field value.

        Returns:
            float: Calculated resistivity value.
        """
        
        eigenValue = 1.701411e38
        self.res_list[-1] = rho         
        Bp_t1 = np.abs(self.TEM.walktem(time_point, self.waveflag, self.signal, self.src, self.rec, self.depth, self.res_list, self.mrec, self.strength, self.srcpts, self.caltype, self.verb))

        g = np.abs((field_value - Bp_t1) / field_value)
        i = 1

        while g > 0.001 and i < 50:
   
            self.res_list[-1] = 1.01 * rho
            Bp_t1 = np.abs(self.TEM.walktem(time_point, self.waveflag, self.signal, self.src, self.rec, self.depth, self.res_list, self.mrec, self.strength, self.srcpts, self.caltype, self.verb))

            gp = np.abs((field_value - Bp_t1) / field_value)
            gp = 100.0 * (gp - g) / rho
            dP = -g / gp
            if np.abs(dP) > 0.5 * rho:
                dP = 0.5 * np.abs(dP) * rho / dP
            rho = rho + dP


            self.res_list[-1] = rho
            Bp_t1 = np.abs(self.TEM.walktem(time_point, self.waveflag, self.signal, self.src, self.rec, self.depth, self.res_list, self.mrec, self.strength, self.srcpts, self.caltype, self.verb))


            g = np.abs((field_value - Bp_t1) / field_value)
            i += 1

        if  rho >= self.RHO_min and rho <= self.RHO_max:
            RHO_A = rho
            return RHO_A,i,g
        else:
            RHO_A = eigenValue
            return RHO_A,i,g

    def getExtremeRes(self, flag, time_point, RHO_min, RHO_max):
        """
        Get extreme resistivity value.

        Args:
            flag (int): Flag for extreme resistivity search.
            time_point (float): Time value.
            RHO_min (float): Minimum resistivity value.
            RHO_max (float): Maximum resistivity value.

        Returns:
            tuple: Flag and extreme resistivity value.
        """
        RHO_extreme = 0.0
        x1 = RHO_min
        x2 = RHO_max
        dx = x2 - x1
        
        grad1 = self.GradB(x1, time_point)
        grad2 = self.GradB(x2, time_point)
        
        if grad2 > 0:
            flag = 1
        else:
            flag = -1

        if grad1 * grad2 >= 0:
            return flag, x1 + dx / 2.0

        else:
            dx = dx / 2
            eps = 1e-2

            while dx > eps:
                x2 = x1 + dx
                grad2 = self.GradB(x2, time_point)

                if grad2 * flag < 0:
                    x1 = x2
                dx = dx / 2

            RHO_extreme = x1
            flag = 0
            return flag, RHO_extreme

    def GradB(self, res, time_point):
        """
        Calculate gradient of B.

        Args:
            res (float): Resistivity value.
            time_point (float): Time value.

        Returns:
            float: Gradient of B.
        """

        self.res_list[-1] = res
        Bi1 = np.abs(self.TEM.walktem(time_point, self.waveflag, self.signal, self.src, self.rec, self.depth, self.res_list, self.mrec, self.strength, self.srcpts, self.caltype, self.verb))


        self.res_list[-1] = res * 1.1
        Bi2 = np.abs(self.TEM.walktem(time_point, self.waveflag, self.signal, self.src, self.rec, self.depth, self.res_list, self.mrec, self.strength, self.srcpts, self.caltype, self.verb))


        GradB = Bi2 - Bi1
        return GradB

    def appRes_point(self, time_point, search_value, index):
        """
        Apparent resistivity calculation.

        Returns:
            float: Calculated time point.
            numpy.ndarray: Array of calculated resistivity values.
        """
        
        eigenValue = 1.701411e38
        isSucceed = 0
        RHO_A = 0.0
        RHO_A1 = 0.0
        RHO_A2 = 0.0
        
        
        self.flag, RHO_extreme = self.getExtremeRes(
            self.flag, time_point, self.RHO_min, self.RHO_max
        )

        if self.flag != 0:            
            RHO_A,i_A,g_A = self.linearSearch(RHO_extreme, time_point, search_value)
            self.RHO_A = RHO_A
            self.RHO_A1 = RHO_A
            self.RHO_A2 = RHO_A
            i_A1 = i_A
            i_A2 = i_A
            g_A1 = g_A
            g_A2 = g_A                        
            isSucceed = 1
            RHO_extreme = eigenValue
        else:
            RHO_A1,i_A1,g_A1 = self.linearSearch(RHO_extreme*0.1, time_point, search_value)
            RHO_A2,i_A2,g_A2 = self.linearSearch(RHO_extreme*10, time_point, search_value)
            if abs(RHO_A2 - eigenValue) < 1e-6 and abs(RHO_A1 - eigenValue) > 1e-6:
                self.RHO_A = RHO_A1
                isSucceed = 1
            elif abs(RHO_A2 - eigenValue) > 1e-6 and abs(RHO_A1 - eigenValue) < 1e-6:
                self.RHO_A = RHO_A2
                isSucceed = 1
            else:
                self.RHO_A = 0
                isSucceed = 0

            self.RHO_A1 = RHO_A1
            self.RHO_A2 = RHO_A2        
        print(
            f"   {self.flag:1}       {index+1:2d}     {i_A1:2d}    {g_A1:6.6f}  {self.RHO_A1:6.2e}  {RHO_extreme:6.2e}    {i_A2:2d}    {g_A2:6.6f}  {self.RHO_A2:6.2e}    {self.RHO_A:6.2e}      {isSucceed:1} "
        )

        return time_point, self.RHO_A, self.RHO_A1, self.RHO_A2

def calculate_resistance(app_res_calculator, time_point, search_value, index):
    """
    计算apparent resistivity。

    Args:
        time_point (float): 时间点。
        search_value (float): 对应的搜索值。
        index (int): 时间点的索引。

    Returns:
        tuple: 包含计算结果的元组 (时间点, 对应的apparent resistivity, RHO_A1, RHO_A2)。
    """
    time_point, rho_a, rho_a1, rho_a2 = app_res_calculator.appRes_point(time_point, search_value, index)
    return time_point, rho_a, rho_a1, rho_a2

def main():
    # =============================================================================
    inputfile = "Ex_on.dat"
    src = [-500, 500, 0, 0, 0, 0]
    rec = [0, 500, 0, 0, 0]
    srcpts = 20
    strength = 1
    signal = 1
    RHO_min = 1
    RHO_max = 1000
    caltype = 'E'
    waveflag = 0
    verb = 0
    parallelflag = 0
    # =============================================================================
    if caltype == 'E' or caltype == 'dE':
        mrec = False
    elif caltype == 'B' or caltype == 'dB':
        mrec = True
    else:
        print("caltype error")
        return
    depth = [0, np.inf]
    res_list = [1e7, 0]                                                 # Example resistivity list
    time_channel = np.loadtxt(inputfile, delimiter=',')[:, 0]           # Example time array
    search_value = np.abs(np.loadtxt(inputfile, delimiter=',')[:, 1])
    print("是否单调 时间道号 迭代L   残差L    RHO左支   RHO特征值   迭代R   残差R    RHO右支   RHO优选值  是否成功")
    # 创建AppResCalculator实例
    app_res_calculator = AppResCalculator(inputfile, src, rec, depth, res_list, mrec, srcpts, strength, signal, caltype, waveflag, verb, time_channel, search_value, RHO_min, RHO_max)
    if parallelflag:
        # 获取CPU核心数
        num_cores = cpu_count()
    else:
        # 不并行
        num_cores = 1
    # 创建进程池
    with Pool(num_cores) as pool:
        # 使用map函数在所有核心上并行计算
        results = pool.starmap(calculate_resistance, [(app_res_calculator, time_point, search_value[i], i) for i, time_point in enumerate(time_channel)])
    # 对结果排序，按时间点升序排列
    results.sort(key=lambda x: x[0])
    # 输出结果
    print("Time    Apparent Resistivity        RHO_A1        RHO_A2")
    for time_point, rho_a, rho_a1, rho_a2 in results:
        print(f"{time_point:.4e}     {rho_a:.4e}        {rho_a1:.4e}       {rho_a2:.4e}")
    time_points = [result[0] for result in results]
    rho_a_values = [result[1] for result in results]
    rho_a1_values = [result[2] for result in results]
    rho_a2_values = [result[3] for result in results]
    data0 = np.column_stack((time_points, rho_a_values))
    data1 = np.column_stack((time_points, rho_a1_values))
    data2 = np.column_stack((time_points, rho_a2_values))
    np.savetxt(app_res_calculator.output_filename + "_appres0.dat", data0, fmt="%f")
    np.savetxt(app_res_calculator.output_filename + "_appres1.dat", data1, fmt="%f")
    np.savetxt(app_res_calculator.output_filename + "_appres2.dat", data2, fmt="%f")
    
if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total execution time: {total_time:5.2f} seconds.")
