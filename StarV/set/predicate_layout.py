class PredLayout:
    def __init__(self, n_base=0, y_blocks=None, a_blocks=None):
        # number of predicates before any ReLU augmentation
        self.n_base = n_base
        # each entry: (start_col, block_size)
        self.y_blocks = [] if y_blocks is None else list(y_blocks)
        self.a_blocks = [] if a_blocks is None else list(a_blocks)

    def n_total(self):
        n = self.n_base
        for s, m in self.y_blocks:
            n = max(n, s + m)
        for s, m in self.a_blocks:
            n = max(n, s + m)
        return n
    
    def add_y_block(self, m):
        sy = self.n_total()
        self.y_blocks.append((sy, m))
        return sy

    def add_relu_bigM_block(self, num_predicate):
        """
        Appends a new (y,a) block at the end of the predicate vector.
        Returns (y_slice, a_slice) in the *new* predicate vector.
        """
        m = num_predicate
        start_y = self.n_total()
        start_a = start_y + m
        self.y_blocks.append((start_y, m))
        self.a_blocks.append((start_a, m))
        return slice(start_y, start_y + m), slice(start_a, start_a + m)
    
    def __str__(self):
        return f"PredLayout(n_base={self.n_base}, y_blocks={self.y_blocks}, a_blocks={self.a_blocks})"
    
    def __repr__(self):
        return self.__str__()


# Backward compatibility for existing imports using PredicateLayout.
PredicateLayout = PredLayout
