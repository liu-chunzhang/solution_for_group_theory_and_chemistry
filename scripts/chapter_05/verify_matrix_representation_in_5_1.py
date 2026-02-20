import numpy as np

def verify_representation(elements, mult_table, matrices):
    """
    param elements: a list of the name of symmetry elements, like ['E', 'C2', 'i', 'sigma']
    param mult_table: a map to define the group multiplication, like {('C2', 'i'): 'sigma'}
    param matrices: a map connecting the name of symmetry elements and their matrices
    """
    is_valid = True
    for g1 in elements:
        for g2 in elements:
            # get the name of theoretical results
            target_element_name = mult_table[(g1, g2)]
            
            # obtain corresponding matrices
            m1 = matrices[g1]
            m2 = matrices[g2]
            m_target = matrices[target_element_name]
            
            # matrix multiplication
            m_result = np.matmul(m1, m2)
            
            # check the equilivance between real result and theoretical result (using allclose)
            if not np.allclose(m_result, m_target):
                print(f"check failure: {g1} * {g2} should be {target_element_name}")
                print(f"matrices' multiplication:\n{m_result}")
                print(f"target: ({target_element_name}):\n{m_target}")
                is_valid = False
                
    if is_valid:
        print("The matrix representation obeys the group multiplication table.")
    return is_valid

# list all symmetry elements
elems = ['E', 'C2', 'sigma_xz', 'sigma_yz']

# list group multiplication table
table = {
    ('E', 'E'): 'E', ('E', 'C2'): 'C2', ('E', 'sigma_xz'): 'sigma_xz', ('E', 'sigma_yz'): 'sigma_yz',
    ('C2', 'E'): 'C2', ('C2', 'C2'): 'E', ('C2', 'sigma_xz'): 'sigma_yz', ('C2', 'sigma_yz'): 'sigma_xz',
    ('sigma_xz', 'E'): 'sigma_xz', ('sigma_xz', 'C2'): 'sigma_yz', ('sigma_xz', 'sigma_xz'): 'E', ('sigma_xz', 'sigma_yz'): 'C2',
    ('sigma_yz', 'E'): 'sigma_yz', ('sigma_yz', 'C2'): 'sigma_xz', ('sigma_yz', 'sigma_xz'): 'C2', ('sigma_yz', 'sigma_yz'): 'E'
}

# list my matrix representation to be checked.
mats = {
    'E':        np.array([[1,0], [0,1]]),
    'C2':       np.array([[-1, 0], [0, -1]]),
    'sigma_xz': np.array([[0, 1], [1, 0]]),
    'sigma_yz': np.array([[0, -1], [-1, 0]])
}

# check
verify_representation(elems, table, mats)
