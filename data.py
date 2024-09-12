__all__ = ["Data"]

class Data:
    def __init__(self, 
                 domain, 
                 equations, 
                 boundary_conditions=None, 
                 initial_conditions=None, 
                 num_domain=200, 
                 num_boundary=None, 
                 num_initial=None,
                 additional_data=None):
        self.domain = domain
        self.boundary_conditions = boundary_conditions
        self.equations = equations
        self.initial_conditions = initial_conditions
        self.additional_data = additional_data
        self.num_domain = num_domain
        self.num_boundary = num_boundary
        self.num_initial = num_initial

    def get_domain(self):
        return self.domain

    def get_boundary_conditions(self):
        return self.boundary_conditions
    
    """
    [0, 1]x[0, 1]
    def boudary(x):
        return x[:,0:1]
    
    """

    def get_physics_equation(self):
        return self.equations

    def get_initial_conditions(self):
        return self.initial_conditions

    def get_additional_data(self):
        return self.additional_data
    
    def num_batches(self, batch_size):
        return len(self.data) // batch_size

    def get_batch(self, batch_index, batch_size):
        start = batch_index * batch_size
        end = (batch_index + 1) * batch_size
        return self.data[start:end]