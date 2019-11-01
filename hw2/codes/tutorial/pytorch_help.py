# at some point you might want to multiply the i-th column of a matrix A with
# the i-th row of a matrix B and then add the resulting matrices
import torch

A = torch.FloatTensor([[1, 1], [2, 1]])  # shape (2, 2)
B = torch.FloatTensor([[0, -1], [-1, 2]])  # shape (2, 2)

# we add an extra dimension to each matrix A and B
# as follows
A = A.t().view(2, 2, 1)
B = B.view(2, 1, 2)

# Now the first 'row' of A (A[0]) is a matrix of dimensions (2, 1)
# corresponding to the first column of A, and the second 'row' of A (A[1]) is
# the second column of A. For B we have that B[0] is the first row of B now
# written as a matrix of dimension (1, 2), and B[1] is the second row

# We multiply the 3-dimensional tensors A and B. The resulting matrix C will
# contain the desired multiplications of the i-th column of A with the i-th row
# of B, and you can access them as C[0] and C[1]
C = torch.matmul(A, B)

# Finally you can sum C along its first axis (index=0) to sum all the resulting
# matrices

result = torch.sum(C, dim=0)  # should be equal to C[0] + C[1]
print(result)
