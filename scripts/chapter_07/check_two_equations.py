import numpy as np

class CharacterTable:
    def __init__(self, matrix: np.ndarray, irreps: list, classes: list):
        if not isinstance(matrix, np.ndarray):
            raise TypeError("Matrix must be numpy.ndarray type.")
        if not isinstance(irreps, list) or not all(isinstance(x, str) for x in irreps):
            raise TypeError("Irreps must be a list of strings.")
        if not isinstance(classes, list) or not all(isinstance(x, str) for x in classes):
            raise TypeError("Classes must be a list of strings, too.")
        if matrix.shape != (len(irreps), len(classes)):
            raise ValueError("The number of irreps and classes should be the same.")

        self.matrix = matrix
        self.irreps = irreps
        self.classes = classes
        
    def get_num_elements(self) -> int:
        """return the number of elements"""
        result = 0
        for i in range(0, self.get_num_irreps()):
            result += self.get_num_elements_in_an_irrep(i)**2
        return result

    def get_num_irreps(self) -> int:
        """return the number of irreps/classes"""
        return self.matrix.shape[0]
    
    def get_num_elements_in_an_irrep(self, i: int):
        """return the number of a given irreducible representation"""
        if i < 0 or i >= self.matrix.shape[0]:
            raise IndexError("The index of irreps out of range.")
        irrep_name_first_letter = self.get_irrep_name(i)[0]
        if irrep_name_first_letter == 'A' or irrep_name_first_letter == 'B':
            return 1
        elif irrep_name_first_letter == 'E':
            return 2
        elif irrep_name_first_letter == 'T' or irrep_name_first_letter == 'F':
            return 3
        else:
            raise IndexError("Undefined irreduicble representation symbol.")
        
    def get_num_elements_in_a_class(self, i: int):
        """return the number of a given irreducible representation"""
        if i < 0 or i >= self.matrix.shape[0]:
            raise IndexError("The index of classes out of range.")
        classes_name_first_letter = self.get_class_name(i)[0]
        if classes_name_first_letter.isdigit():
            return int(classes_name_first_letter)
        else:
            return 1
        
    def get_element(self, i: int, j: int):
        """return the character of the ith irreps and the jth class (index start from 1)"""
        if i < 0 or i >= self.matrix.shape[0]:
            raise IndexError("The index of irreps out of range.")
        if j < 0 or j >= self.matrix.shape[1]:
            raise IndexError("The index of classes out of range.")
        return self.matrix[i, j]

    def get_irrep_name(self, i: int) -> str:
        """return the name of the ith irrep"""
        return self.irreps[i]

    def get_class_name(self, j: int) -> str:
        """return the name of the jth class"""
        return self.classes[j]
    
    def show_irreps_names(self):
        print(self.irreps)
    
    def show_classes_names(self):
        print(self.classes)
    
    def show_matrix(self):
        last_position = self.get_num_irreps() - 1
        
        for i in range(0, self.get_num_irreps()):
            string = "["
            for j in range(0, self.get_num_irreps()):
                if j < last_position :
                    string += f"{self.matrix[i][j]}, "
                else:
                    string += f"{self.matrix[i][j]}]"
            print(string)

def test_eqn_735( c: CharacterTable ):
    """check the eqn (7-3.5)"""
    c_irreps_number = c.get_num_irreps()
    k = c.get_num_irreps()
    g = c.get_num_elements()
    
    for mu in range(0, c_irreps_number):
        for nu in range(0, c_irreps_number):
            left_sum, right_sum = 0, ( mu == nu ) * g
            for i in range(0, k):
                g_i = c.get_num_elements_in_a_class(i)
                left_sum += g_i * c.get_element(mu,i) * c.get_element(nu,i)
            if abs( left_sum - right_sum ) > 0.01 :
                raise ValueError("Find unfit position!")
    print("The equation (7-3.5) has been checked and it is true.")
            
def test_a7310( c: CharacterTable ):
    c_irreps_number = c.get_num_irreps()
    c_classes_number = c.get_num_irreps()
    g = c.get_num_elements()
    
    for i in range(0, c_classes_number):
        for j in range(0, c_classes_number):
            g_j = c.get_num_elements_in_a_class(j)
            left_sum, right_sum = 0, g / g_j * ( i == j )
            for nu in range(0, c_irreps_number):
                left_sum += c.get_element(nu,i) * c.get_element(nu,j)
            if abs( left_sum - right_sum ) > 0.01 :
                raise ValueError("Find unfit position!")
    print("The equation (A.7-3.10) has been checked and it is true.")

if __name__ == "__main__":
    # Check for the C_{4v} point group.
    c4v_character_table = np.array([[1, 1, 1, 1, 1],
                                [1, 1, 1, -1, -1],
                                [1, -1, 1, 1, -1],
                                [1, -1, 1, -1, 1],
                                [2, 0, -2, 0, 0]])
    c4v_irreps_name = ['A1', 'A2', 'B1', 'B2', 'E']
    c4v_classes_name = ['E', '2C4', 'C2', '2Mv', '2Md']

    c4v_cp = CharacterTable( c4v_character_table , c4v_irreps_name , c4v_classes_name )

    c4v_cp.show_matrix()
    c4v_cp.show_irreps_names()
    c4v_cp.show_classes_names()
    test_eqn_735(c4v_cp)
    test_a7310(c4v_cp)
    
    # Another instance. Check for the T_d point group.
    td_character_table = np.array([[1, 1, 1, 1, 1],
                                [1, 1, 1, -1, -1],
                                [2, -1, 2, 0, 0],
                                [3, 0, -1, 1, -1],
                                [3, 0, -1, -1, 1]])
    td_irreps_name = ['A1', 'A2', 'E', 'T1', 'T2']
    td_classes_name = ['E', '8C3', '3C2', '6S4', '6Md']

    td_cp = CharacterTable( td_character_table , td_irreps_name , td_classes_name )

    td_cp.show_matrix()
    td_cp.show_irreps_names()
    td_cp.show_classes_names()
    test_eqn_735(td_cp)
    test_a7310(td_cp)