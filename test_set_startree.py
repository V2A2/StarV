"""
Test star tree module
Dung Tran, 9/14/2023

"""

from StarV.set.startree import StarNode

class Test(object):

    def __init__(self):
        self.n_fails = 0
        self.n_tests = 0


    def test_StarNode_constructor(self):

        self.n_tests = self.n_tests + 1

        a = StarNode(1)
        b = StarNode(2)
        c = StarNode(3)
        b.add_child(StarNode(4))
        c.add_child(StarNode(5))

        a.add_child(b)
        a.add_child(c)

        print('a = {}'.format(a))

        a_child = a.get_child(0)
        print('a_child = {}'.format(a_child))

        a_all_childs = a.get_childs()
        print('a_all_childs = {}'.format(a_all_childs))

        
    
        
if __name__ == "__main__":
    test = Test()
    print('\n=======================\
    ================================\
    ================================\
    ===============================\n')
    test.test_StarNode_constructor()
    print('\n========================\
    =================================\
    =================================\
    =================================\n')
    print('Testing startree module: fails: {}, successfull: {}, \
    total tests: {}'.format(test.n_fails,
                            test.n_tests - test.n_fails,
                            test.n_tests))
