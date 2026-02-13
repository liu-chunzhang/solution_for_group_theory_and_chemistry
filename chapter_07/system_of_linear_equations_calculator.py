import numpy as np

def solve_linear_equation(A_list, b_list):
    """
    求解线性方程组 Ax = b
    :param A_list: 嵌套列表形式的矩阵 A
    :param b_list: 列表形式的向量 b
    """
    # 转换为 numpy 数组，支持复数类型
    A = np.array(A_list, dtype=complex)
    b = np.array(b_list, dtype=complex)
    
    try:
        # 求解方程
        x = np.linalg.solve(A, b)
        
        print("--- 求解结果 ---")
        print(f"矩阵 A:\n{A.real.astype(float) if not np.iscomplex(A).any() else A}")
        print(f"向量 b: {b.real.astype(float) if not np.iscomplex(b).any() else b}")
        print(f"解 x: {x.real.astype(float) if not np.iscomplex(x).any() else x}")
        
        # 验证结果
        check = np.allclose(np.dot(A, x), b)
        print(f"验证 (Ax=b): {'成功' if check else '失败'}")
        return x
        
    except np.linalg.LinAlgError:
        print("错误：矩阵 A 是奇异矩阵，无唯一解。")
        return None

# --- 测试你提供的用例 ---

A_input = [[1, 1, 1, 1], 
           [1, 1, -1, -1], 
           [1, -1, 1, -1], 
           [1, -1, -1, 1]]
b_input = [4, -2, 0, -2]

solve_linear_equation(A_input, b_input)

print("---------------------------")

varepsilon = -0.5 + (3**0.5 / 2) * 1j
varepsilon2 = -0.5 - (3**0.5 / 2) * 1j
A_input = [[1, 1, 1, 1, 1, 1], 
           [1, varepsilon, varepsilon2, 1, varepsilon, varepsilon2], 
           [1, varepsilon2, varepsilon, 1, varepsilon2, varepsilon], 
           [1, 1, 1, -1, -1, -1],
           [1, varepsilon, varepsilon2, -1, -varepsilon, -varepsilon2], 
           [1, varepsilon2, varepsilon, -1, -varepsilon2, -varepsilon] ]
b_input = [4, 1, 1, 2, -1, -1]

solve_linear_equation(A_input, b_input)

print("---------------------------")

A_input = [[1, 1, 1, 1, 2], 
           [1, 1, -1, -1, 0], 
           [1, 1, 1, 1, -2], 
           [1, -1, 1, -1, 0],
           [1, -1, -1, 1, 0]]
b_input = [4, 0, 0, 0, -2]

solve_linear_equation(A_input, b_input)

print("---------------------------")

A_input = [[1, 1, 2, 3, 3, 1, 1, 2, 3, 3], 
           [1, 1, -1, 0, 0, 1, 1, -1, 0, 0],
           [1, 1, 2, -1, -1, 1, 1, 2, -1, -1],
           [1, -1, 0, 1, -1, 1, -1, 0, 1, -1],
           [1, -1, 0, -1, 1, 1, -1, 0, -1, 1],
           [1, 1, 2, 3, 3, -1, -1, -2, -3, -3],
           [1, 1, -1, 0, 0, -1, -1, 1, 0, 0],
           [1, 1, 2, -1, -1, -1, -1, -2, 1, 1],
           [1, -1, 0, 1, -1, -1, 1, 0, -1, 1],
           [1, -1, 0, -1, 1, -1, 1, 0, 1, -1]]
b_input = [15, 0, -1, 1, 1, -3, 0, 5, -1, 3]

solve_linear_equation(A_input, b_input)