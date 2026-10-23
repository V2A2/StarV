"""
  Star Tree Class, store star reachable set of NNCS over time t1, t2, ..., tN
  
  Dung Tran, 8/14/2023
"""

class StarNode(object):
    """
    Star Tree class

    a tree of star sets

    each star has 1 parent_node and multiple children nodes

    """
    
    
    
    def __init__(self, data=None):

        self.data = data   # is a star/probstar set
        self.childs = []   # childs is a list of stars 
        

    def add_child(self, child):

        # child is another star node
        self.childs.append(child) 


    def get_child(self, id):

        assert id < len(self.childs), 'error: invalid id'

        return self.childs[id]

    def get_data(self):

        return self.data

    def get_childs(self):

        return self.childs   # get all children

    
