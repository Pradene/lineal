use crate::vector::Vector;

use std::fmt;
use num::{One, Signed, Zero};
use std::ops::{
    Add,
    Sub,
    Mul,
    Index, IndexMut
};

#[derive(Debug, Clone, Copy)]
pub struct Matrix<T, const M: usize, const N: usize> {
    pub data:  [[T; N]; M],
}

impl<T, const M: usize, const N: usize> fmt::Display for Matrix<T, M, N>
where T:
    fmt::Display
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        
        for i in 0..M {
            write!(f, "\n ")?;

            for j in 0..N {
                if j == 0 {
                    write!(f, "[")?;
                } else {
                    write!(f, ", ")?;
                }

                write!(f, "{}", self[i][j])?;
            }
            
            write!(f, "],")?;
        }
        
        write!(f, "\n]")
    }
}

impl<T, const M: usize, const N: usize> From<[[T; N]; M]> for Matrix<T, M, N> {
    fn from(data: [[T; N]; M]) -> Self {
        Self {data}
    }
}

impl<T, const M: usize, const N: usize> Index<usize> for Matrix<T, M, N> {
    type Output = [T; N];

    fn index(&self, i: usize) -> &Self::Output {
        if i < M {
            &self.data[i] // Return a reference to the element at index `i`
        } else {
            panic!("Index out of bounds");
        }
    }
}

// Implement IndexMut trait for mutable access to elements (self[i] = value)
impl<T, const M: usize, const N: usize> IndexMut<usize> for Matrix<T, M, N> {
    fn index_mut(&mut self, i: usize) -> &mut Self::Output {
        if i < M {
            &mut self.data[i] // Return a mutable reference to the element at index `i`
        } else {
            panic!("Index out of bounds");
        }
    }
}

impl<T, const M: usize, const N: usize> Add for Matrix<T, M, N>
where T:
    Add<Output = T> +
    Copy
{
    type Output = Self;

    fn add(self, m: Self) -> Self::Output {
        let mut result = self.clone();
        for i in 0..M {
            for j in 0..N {
                result[i][j] = result[i][j] + m[i][j];
            }
        }

        result
    }
}

impl<T, const M: usize, const N: usize> Sub for Matrix<T, M, N>
where T:
    Sub<Output = T> +
    Copy
{
    type Output = Self;

    fn sub(self, m: Self) -> Self::Output {
        let mut result = self.clone();
        for i in 0..M {
            for j in 0..N {
                result[i][j] = result[i][j] - m[i][j];
            }
        }

        result
    }
}

impl<T, const M: usize, const N: usize, const P: usize> Mul<Matrix<T, N, P>> for Matrix<T, M, N>
where T:
    Copy +
    Mul<Output = T> +
    Add<Output = T> +
    Default
{
    type Output = Matrix<T, M, P>;

    // Matrix multiplication: self (M x N) * matrix (N x P) -> result (M x P)
    fn mul(self, matrix: Matrix<T, N, P>) -> Self::Output {
        let mut result = Matrix {
            data: [[T::default(); P]; M],
        };

        for i in 0..M {
            for j in 0..P {
                for k in 0..N {
                    result[i][j] = result[i][j] + self[i][k] * matrix[k][j];
                }
            }
        }

        result
    }
}

impl<T, const M: usize, const N: usize> Mul<Vector<T, N>> for Matrix<T, M, N>
where T:
    Copy +
    Default +
    Mul<Output = T> +
    Add<Output = T>
{
    type Output = Vector<T, M>;

    fn mul(self, vector: Vector<T, N>) -> Self::Output {
        let mut result = Vector {
            data: [T::default(); M]
        };

        for i in 0..M {
            for j in 0..N {
                result[i] = result[i] + self[i][j] * vector[j];
            }
        }

        result
    }
}

impl<T, const M: usize, const N: usize> Mul<T> for Matrix<T, M, N>
where T:
    Mul<Output = T> +
    Copy
{
    type Output = Self;

    fn mul(self, scalar: T) -> Self {
        let mut result = self.clone();
        for i in 0..M {
            for j in 0..N {
                result[i][j] = result[i][j] * scalar;
            }
        }

        result
    }
}

impl<T, const M: usize, const N: usize> PartialEq for Matrix<T, M, N>
where T:
    PartialEq
{
    fn eq(&self, matrix: &Self) -> bool {
        self.data == matrix.data
    }
}

impl<T, const M: usize, const N: usize> Matrix<T, M, N> {
    pub fn new(data: [[T; N]; M]) -> Self {
        Self {data}
    }
}

impl<T, const M: usize, const N: usize> Matrix<T, M, N>
where T:
    Copy +
    Default +
{
    pub fn transpose(&self) -> Matrix<T, N, M> {
        let mut result = Matrix {
            data: [[T::default(); M]; N]
        };

        for i in 0..N {
            for j in 0..M {
                result[i][j] = self[j][i];
            }
        }

        result
    }
}

impl<T, const N: usize> Matrix<T, N, N>
where T:
    Copy +
    Default +
    Add<Output = T>
{
    pub fn trace(&self) -> T {
        let mut result = T::default();
        for i in 0..N {
            result = result + self[i][i]
        }

        result
    }
}

impl<T, const M: usize, const N: usize> Matrix<T, M, N>
where T:
    Copy +
    Signed +
    PartialOrd
{
    pub fn row_echelon(&self) -> Matrix<T, M, N> {
        let mut result = self.clone();
        let mut pivot_row = 0;

        for col in 0..N {
            if pivot_row >= M {
                break;
            }

            let mut pivot = None;
            for row in pivot_row..M {
                if result[row][col].abs() > T::zero() {
                    pivot = Some(row);
                    break;
                }
            }

            if pivot.is_none() {
                continue;
            }

            let pivot = pivot.unwrap();
            if pivot != pivot_row {
                result.data.swap(pivot, pivot_row);
            }

            let pivot_value = result[pivot_row][col];
            if pivot_value == T::zero() {
                continue;
            }

            for j in col..N {
                result[pivot_row][j] = result[pivot_row][j] / pivot_value;
            }

            for row in 0..M {
                if row == pivot_row {
                    continue
                }

                let factor = result[row][col];
                for j in col..N {
                    result[row][j] = result[row][j] - factor * result[pivot_row][j];
                }
            }

            pivot_row += 1
        }
        
        result
    }

    pub fn rank(&self) -> usize {
        let mut rank = 0;
        let rref = self.row_echelon();

        for row in 0..M {
            for col in 0..N {
                if rref[row][col] != T::zero() {
                    rank = rank + 1;
                    break;
                }
            }
        }
        
        rank
    }
}

impl<T, const N: usize> Matrix<T, N, N>
where T:
    Copy +
    Signed +
    PartialOrd +
    Into<f32>
{
    fn lu_decomposition(&self) -> (Matrix<T, N, N>, Matrix<T, N, N>, usize) {
        let mut l = Matrix::from([[T::zero(); N]; N]);
        let mut u = self.clone();
        let mut permutation_count = 0;
    
        for i in 0..N {
            // Pivoting (partial pivoting)
            let mut max_row = i;
            for k in i + 1..N {
                if u[k][i].abs() > u[max_row][i].abs() {
                    max_row = k;
                }
            }

            if max_row != i {
                u.data.swap(i, max_row);
                permutation_count += 1;
            }
    
            // Compute L and U
            for j in i..N {
                l[j][i] = u[j][i] / u[i][i];
            }

            for j in i + 1..N {
                for k in i..N {
                    u[j][k] = u[j][k] - l[j][i] * u[i][k];
                }
            }
        }
    
        // Set diagonal of L to 1
        for i in 0..N {
            l[i][i] = T::one();
        }
    
        (l, u, permutation_count)
    }
    
    // Function to compute the determinant using LU Decomposition
    pub fn determinant(&self) -> T {
        let (_, u, permutation_count) = self.lu_decomposition();
        let mut determinant = T::one();
    
        // Product of diagonal elements of U
        for i in 0..N {
            determinant = determinant * u[i][i];
        }
    
        // Adjust for row swaps
        if permutation_count % 2 != 0 {
            determinant = -determinant;
        }
    
        determinant
    }
    
    // Function to compute the inverse using LU Decomposition
    pub fn inverse(&self) -> Option<Matrix<T, N, N>> {
        let det = self.determinant();
        if det == T::zero() {
            return None;
        }

        let (l, u, _) = self.lu_decomposition();
        let mut inverse = [[T::zero(); N]; N];
    
        // Solve for each column of the identity matrix
        for i in 0..N {
            let mut b = [T::zero(); N];
            b[i] = T::one();
    
            // Solve L * y = b using forward substitution
            let mut y = [T::zero(); N];
            for j in 0..N {
                y[j] = b[j];
                for k in 0..j {
                    y[j] = y[j] - l[j][k] * y[k];
                }
            }
    
            // Solve U * x = y using backward substitution
            let mut x = [T::zero(); N];
            for j in (0..N).rev() {
                x[j] = y[j] / u[j][j];
                for k in (j + 1..N).rev() {
                    x[j] = x[j] - u[j][k] * x[k] / u[j][j];
                }
            }
    
            // Assign the solution to the inverse matrix
            for j in 0..N {
                inverse[j][i] = x[j];
            }
        }
    
        Some(Matrix {
            data: inverse
        })
    }
}

impl<T> Matrix<T, 4, 4>
where T:
    Default +
    Copy +
    Add<Output = T> +
    Sub<Output = T> +
    Mul<Output = T> +
    Zero +
    One +
    From<f32>
{
    pub fn projection(fov: f32, ratio: f32, near: f32, far: f32) -> Matrix<f32, 4, 4> {
        let mut projection_matrix = Matrix::from([[0.; 4]; 4]);
    
        let fov_factor = 1. / (fov / 2.).tan();
    
        projection_matrix[0][0] = fov_factor / ratio;
        projection_matrix[1][1] = fov_factor;
        projection_matrix[2][2] = (far + near) / (near - far);
        projection_matrix[2][3] = (2. * far * near) / (near - far);
        projection_matrix[3][2] = -1.;
    
        // Transpose for column major order
        projection_matrix.transpose()
    }

    pub fn rotate(&mut self, angle: f32, axis: Vector<T, 3>) -> Matrix<T, 4, 4> {
        let c = T::from(angle.cos());
        let s = T::from(angle.sin());
        let one_minus_c = T::one() - c;
        let [x, y, z] = axis.data;

        // Rotation matrix components
        let rotation_matrix = Matrix {
            data: [
                [
                    x * x * one_minus_c + c,
                    x * y * one_minus_c - z * s,
                    x * z * one_minus_c + y * s,
                    T::zero(),
                ],
                [
                    y * x * one_minus_c + z * s,
                    y * y * one_minus_c + c,
                    y * z * one_minus_c - x * s,
                    T::zero(),
                ],
                [
                    z * x * one_minus_c - y * s,
                    z * y * one_minus_c + x * s,
                    z * z * one_minus_c + c,
                    T::zero(),
                ],
                [T::zero(), T::zero(), T::zero(), T::one()],
            ],
        };

        // Update the current matrix by multiplying it with the rotation matrix
        let result = *self * rotation_matrix;
        return result;
    }
}

impl<T, const N: usize> Matrix<T, N, N>
where T:
    Copy +
    Zero +
    One
{
    pub fn identity() -> Self {
        let mut data = [[T::zero(); N]; N];

        for i in 0..N {
            data[i][i] = T::one();
        }

        return Matrix { data };
    }
}