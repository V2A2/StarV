
"""
Test util load module
Dung Tran, 9/12/2022
"""

# from StarV.util import load
from StarV.util import load_all_benchmark
# import StarV.util.load1 as load1
import polytope as pc

class Test(object):
    """
    Testing module net class and methods
    """

    def __init__(self):

        self.n_fails = 0
        self.n_tests = 0

    def test_load_2017_IEEE_TNNLS(self):

        self.n_tests = self.n_tests + 1
        try:
            net = load_all_benchmark.load_2017_IEEE_TNNLS()
            net.info()
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')

    def test_load_ACASXU(self, x, y, spec_id):

        self.n_tests = self.n_tests + 1
            
        try:
            net, lb, ub, unsafe_mat, unsafe_vec = load_all_benchmark.load_ACASXU(x,y,spec_id)
            net.info()
            print('input lower bound: {}'.format(lb))
            print('output lower bound: {}'.format(ub))
            print('unsafe region of spec_id = {}: {}'.format(spec_id, pc.Polytope(unsafe_mat, unsafe_vec)))
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')

    def test_load_DRL(self, net_id, spec_id):

        self.n_tests = self.n_tests + 1
        
        try:
            net, lb, ub, unsafe_mat, unsafe_vec = load_all_benchmark.load_DRL(net_id, spec_id)
            net.info()
            print('input lower bound: {}'.format(lb))
            print('output lower bound: {}'.format(ub))
            print('unsafe region of spec_id = {}: {}'.format(spec_id, pc.Polytope(unsafe_mat, unsafe_vec)))
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')

    def test_load_harmonic_oscillator_model(self):

        self.n_tests = self.n_tests + 1

        try:

            plant, lb, ub, input_lb, input_ub = load_all_benchmark.load_harmonic_oscillator_model()
            print('plant info: ')
            plant.info()
            print('initial conditions: ')
            print('lower bound: {}'.format(lb))
            print('upper bound: {}'.format(ub))
            print('input conditions: ')
            print('input lower bound: {}'.format(input_lb))
            print('input upper bound: {}'.format(input_ub))
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')

    def test_load_building_model(self):
      
        self.n_tests = self.n_tests + 1
     
        try:
            print("begin print A plant,, B, C")
            plant = load_all_benchmark.load_building_model()
            print('plant: ',plant.info())
            
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')

    
    def test_load_iss_model(self):
     
        self.n_tests = self.n_tests + 1
     
        try:
            print("begin print A plant,, B, C")
            plant = load_all_benchmark.load_iss_model()
            print('plant: ',plant.info())
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')


    def test_load_helicopter_model(self):
        
        self.n_tests = self.n_tests + 1
     
        try:
            print("begin print plant,A, B, C")
            plant= load_all_benchmark.load_helicopter_model()
            print('plant: ',plant.info())
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')


    def test_load_MNA5_model(self):
        
        self.n_tests = self.n_tests + 1
     
        try:
            print("begin print plant,A, B, C")
            plant= load_all_benchmark.load_MNA5_model()
            # print('\ngA = {}'.format(plant.gA))
            # print('\ngB = {}'.format(plant.gB))
            print('plant: ',plant.info())
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')

    def test_load_mcs_model(self):
        
        self.n_tests = self.n_tests + 1
     
        try:
            print("begin print plant,A, B, C")
            plant = load_all_benchmark.load_mcs_model()
            # print('\ngA = {}'.format(plant.gA))
            # print('\ngB = {}'.format(plant.gB))
            print('plant: ',plant.info())
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')
    
    def test_load_pde_model(self):
        
        self.n_tests = self.n_tests + 1
     
        try:
            print("begin print plant,A, B, C")
            plant= load_all_benchmark.load_pde_model()
            # print('\ngA = {}'.format(plant.gA))
            # print('\ngB = {}'.format(plant.gB))
            print('plant: ',plant.info())
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')

    def test_load_fom_model(self):
        
        self.n_tests = self.n_tests + 1
     
        try:
            print("begin print plant,A, B, C")
            plant= load_all_benchmark.load_fom_model()
            # print('\ngA = {}'.format(plant.gA))
            # print('\ngB = {}'.format(plant.gB))
            print('plant: ',plant.info())
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')


    def test_load_MNA1_model(self):
        
        self.n_tests = self.n_tests + 1
     
        try:
            print("begin print plant,A, B, C")
            plant= load_all_benchmark.load_MNA1_model()
            # print('\ngA = {}'.format(plant.gA))
            # print('\ngB = {}'.format(plant.gB))
            print('plant: ',plant.info())
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')


    def test_load_heat_model(self):
        
        self.n_tests = self.n_tests + 1
     
        try:
            print("begin print plant,A, B, C")
            plant= load_all_benchmark.load_heat_model()
            # print('\ngA = {}'.format(plant.gA))
            # print('\ngB = {}'.format(plant.gB))
            print('plant: ',plant.info())
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')

    def test_load_beam_model(self):
        
        self.n_tests = self.n_tests + 1
     
        try:
            print("begin print plant,A, B, C")
            plant= load_all_benchmark.load_beam_model()
            # print('\ngA = {}'.format(plant.gA))
            # print('\ngB = {}'.format(plant.gB))
            print('plant: ',plant.info())
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')



if __name__ == "__main__":

    test_load = Test()
    print('\n=======================\
    ================================\
    ================================\
    # ===============================\n')
    # test_load.test_load_2017_IEEE_TNNLS()
    # test_load.test_load_ACASXU(x=1,y=2,spec_id=2)
    # test_load.test_load_DRL(net_id=0, spec_id=1)
    # test_load.test_load_harmonic_oscillator_model()
    test_load.test_load_building_model()
    test_load.test_load_iss_model()
    test_load.test_load_helicopter_model()
    test_load.test_load_MNA5_model()
    test_load.test_load_mcs_model()
    test_load.test_load_pde_model()
    test_load.test_load_fom_model()
    test_load.test_load_MNA1_model()
    test_load.test_load_heat_model()
    test_load.test_load_beam_model()
    
    print('\n========================\
    =================================\
    =================================\
    =================================\n')
    print('Testing Load module: fails: {}, successfull: {}, \
    total tests: {}'.format(test_load.n_fails,
                            test_load.n_tests - test_load.n_fails,
                            test_load.n_tests))
    


