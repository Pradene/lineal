use crate::vector::Vector;

use num::Float;
use std::fmt;
use std::ops::{Add, AddAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign};

#[derive(Debug, Clone, Copy)]
pub struct Matrix<T, const R: usize, const C: usize> {
    pub data: [[T; R]; C],
}

impl<T, const R: usize, const C: usize> Matrix<T, R, C>
where
    T: Float,
{
    pub fn new() -> Self {
        return Matrix {
            data: [[T::zero(); R]; C],
        };
    }

    pub fn from_col(cols: [[T; R]; C]) -> Self {
        return Matrix { data: cols };
    }

    pub fn from_row(rows: [[T; C]; R]) -> Self {
        let mut data = [[T::zero(); R]; C];
        for r in 0..R {
            for c in 0..C {
                data[c][r] = rows[r][c];
            }
        }
        return Matrix { data };
    }
}

impl<T, const R: usize, const C: usize> fmt::Display for Matrix<T, R, C>
where
    T: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for r in 0..R {
            write!(f, "\n ")?;
            for c in 0..C {
                if c == 0 {
                    write!(f, "[")?;
                } else {
                    write!(f, ", ")?;
                }
                write!(f, "{}", self[c][r])?;
            }
            write!(f, "],")?;
        }
        write!(f, "\n]")?;

        return Ok(());
    }
}

impl<T, const R: usize, const C: usize> Index<usize> for Matrix<T, R, C> {
    type Output = [T; R];

    fn index(&self, i: usize) -> &Self::Output {
        if i < C {
            &self.data[i] // Return a reference to the element at index `i`
        } else {
            panic!("Index out of bounds");
        }
    }
}

// Implement IndexMut trait for mutable access to elements (self[i] = value)
impl<T, const R: usize, const C: usize> IndexMut<usize> for Matrix<T, R, C> {
    fn index_mut(&mut self, i: usize) -> &mut Self::Output {
        if i < C {
            &mut self.data[i] // Return a mutable reference to the element at index `i`
        } else {
            panic!("Index out of bounds");
        }
    }
}

impl<T, const R: usize, const C: usize> Add for Matrix<T, R, C>
where
    T: Float,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let mut result = self.clone();
        for r in 0..R {
            for c in 0..C {
                result[c][r] = result[c][r] + rhs[c][r];
            }
        }

        return result;
    }
}

impl<T, const R: usize, const C: usize> AddAssign for Matrix<T, R, C>
where
    T: Float,
{
    fn add_assign(&mut self, rhs: Self) {
        for r in 0..R {
            for c in 0..C {
                self[c][r] = self[c][r] + rhs[c][r];
            }
        }
    }
}

impl<T, const R: usize, const C: usize> Sub for Matrix<T, R, C>
where
    T: Float,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut result = self.clone();
        for r in 0..R {
            for c in 0..C {
                result[c][r] = result[c][r] - rhs[c][r];
            }
        }

        return result;
    }
}

impl<T, const R: usize, const C: usize> SubAssign for Matrix<T, R, C>
where
    T: Float,
{
    fn sub_assign(&mut self, rhs: Self) {
        for r in 0..R {
            for c in 0..C {
                self[c][r] = self[c][r] - rhs[c][r];
            }
        }
    }
}

impl<T, const R: usize, const C: usize, const P: usize> Mul<Matrix<T, C, P>> for Matrix<T, R, C>
where
    T: Float,
{
    type Output = Matrix<T, R, P>;

    // Matrix multiplication: self (R x C) * matrix (C x P) -> result (R x P)
    fn mul(self, rhs: Matrix<T, C, P>) -> Self::Output {
        let mut result = Matrix {
            data: [[T::zero(); R]; P],
        };

        for p in 0..P {
            for r in 0..R {
                let mut sum = T::zero();
                for c in 0..C {
                    sum = sum + self[c][r] * rhs[p][c];
                }
                result[p][r] = sum;
            }
        }

        return result;
    }
}

impl<T, const R: usize, const C: usize, const P: usize> MulAssign<Matrix<T, C, P>>
    for Matrix<T, R, C>
where
    T: Float,
{
    fn mul_assign(&mut self, rhs: Matrix<T, C, P>) {
        for p in 0..P {
            for r in 0..R {
                let mut sum = T::zero();
                for c in 0..C {
                    sum = sum + self[c][r] * rhs[p][c];
                }
                self[p][r] = sum;
            }
        }
    }
}

impl<T, const R: usize, const C: usize> Mul<Vector<T, C>> for Matrix<T, R, C>
where
    T: Float,
{
    type Output = Vector<T, R>;

    fn mul(self, rhs: Vector<T, C>) -> Self::Output {
        let mut result = Vector {
            data: [T::zero(); R],
        };

        for r in 0..R {
            for c in 0..C {
                result[r] = result[r] + self[r][c] * rhs[c];
            }
        }

        return result;
    }
}

impl<T, const R: usize, const C: usize> Mul<T> for Matrix<T, R, C>
where
    T: Float,
{
    type Output = Self;

    fn mul(self, rhs: T) -> Self {
        let mut result = self.clone();
        for r in 0..R {
            for c in 0..C {
                result[r][c] = result[r][c] * rhs;
            }
        }

        return result;
    }
}

impl<T, const R: usize, const C: usize> MulAssign<T> for Matrix<T, R, C>
where
    T: Float,
{
    fn mul_assign(&mut self, rhs: T) {
        for r in 0..R {
            for c in 0..C {
                self[r][c] = self[r][c] * rhs;
            }
        }
    }
}

impl<T, const R: usize, const C: usize> PartialEq for Matrix<T, R, C>
where
    T: PartialEq + Float,
{
    fn eq(&self, other: &Self) -> bool {
        for col in 0..C {
            for row in 0..R {
                if (self.data[col][row] - other.data[col][row]).abs() > T::epsilon() {
                    return false;
                }
            }
        }

        true
    }
}

impl<T, const R: usize, const C: usize> Matrix<T, R, C>
where
    T: Float,
{
    pub fn transpose(&self) -> Matrix<T, C, R> {
        let mut data = [[T::zero(); C]; R];

        for c in 0..C {
            for r in 0..R {
                data[r][c] = self[c][r];
            }
        }

        return Matrix::from_col(data);
    }
}

impl<T, const R: usize, const C: usize> Matrix<T, R, C>
where
    T: Float,
{
    pub fn row_echelon(&self) -> Matrix<T, R, C> {
        let mut result = *self;
        let mut pivot_row = 0;

        // Iterate through columns (not exceeding row count)
        for col in 0..C.min(R) {
            if pivot_row >= R {
                break;
            }

            // Find pivot in current column (row-major perspective)
            let mut pivot = None;
            for r in pivot_row..R {
                if result.data[col][r].abs() > T::epsilon() {
                    pivot = Some(r);
                    break;
                }
            }

            let Some(pivot) = pivot else { continue };

            // Swap rows in column-major storage
            if pivot != pivot_row {
                for c in 0..C {
                    result.data[c].swap(pivot_row, pivot);
                }
            }

            // Normalize pivot r
            let pivot_val = result.data[col][pivot_row];
            if pivot_val.abs() <= T::epsilon() {
                continue;
            }

            // Normalize current pivot r
            for c in col..C {
                result.data[c][pivot_row] = result.data[c][pivot_row] / pivot_val;
            }

            // Eliminate other rows
            for r in 0..R {
                if r == pivot_row {
                    continue;
                }

                let factor = result.data[col][r];
                for c in col..C {
                    result.data[c][r] = result.data[c][r] - factor * result.data[c][pivot_row];
                }
            }

            pivot_row += 1;
        }

        return result;
    }

    pub fn rank(&self) -> usize {
        let rref = self.row_echelon();
        let mut rank = 0;

        // Check for non-zero rows (column-major perspective)
        for r in 0..R {
            let mut all_zero = true;
            for c in 0..C {
                if c < r && c >= C {
                    continue;
                }
                if rref.data[c][r].abs() > T::epsilon() {
                    all_zero = false;
                    break;
                }
            }
            if !all_zero {
                rank += 1;
            }
        }

        return rank;
    }
}

impl<T, const S: usize> Matrix<T, S, S>
where
    T: Float,
{
    fn lu_decomposition(&self) -> (Matrix<T, S, S>, Matrix<T, S, S>, Vec<usize>) {
        let mut l = Matrix::new();
        let mut u = self.clone();
        let mut p: Vec<usize> = (0..S).collect(); // Track permutation order [0,1,2,3]
    
        for i in 0..S {
            // Partial pivoting: find max row in column `i`
            let mut max_row = i;
            for row in i..S {
                if u.data[i][row].abs() > u.data[i][max_row].abs() {
                    max_row = row;
                }
            }
    
            if max_row != i {
                // Swap rows in U and L
                for col in 0..S {
                    u.data[col].swap(i, max_row);
                }
                for col in 0..i {
                    l.data[col].swap(i, max_row);
                }
                p.swap(i, max_row); // Update permutation vector
            }
    
            // Compute L and U
            let pivot = u.data[i][i];
            for row in i..S {
                l.data[i][row] = u.data[i][row] / pivot;
            }
    
            for row in (i + 1)..S {
                let factor = l.data[i][row];
                for col in i..S {
                    u.data[col][row] = u.data[col][row] - factor * u.data[col][i];
                }
            }
        }
    
        // Set diagonal of L to 1
        for i in 0..S {
            l.data[i][i] = T::one();
        }
    
        (l, u, p)
    }

    // Function to compute the determinant using LU Decomposition
    pub fn determinant(&self) -> T {
        let (_, u, p) = self.lu_decomposition();
        let mut determinant = T::one();

        // Product of diagonal elements of U
        for i in 0..S {
            determinant = determinant * u[i][i];
        }

        // Adjust for row swaps
        if p.len() % 2 != 0 {
            determinant = -determinant;
        }

        determinant
    }

    // Function to compute the inverse using LU Decomposition
    pub fn inverse(&self) -> Option<Matrix<T, S, S>> {
        let (l, u, p) = self.lu_decomposition();
        let mut inverse: Matrix<T, S, S> = Matrix::new();
    
        // Check determinant (product of U's diagonal)
        let mut det = T::one();
        for i in 0..S {
            det = det * u.data[i][i];
        }
        if det == T::zero() {
            return None;
        }
    
        // Solve for each column of the identity matrix (permuted by p)
        for col in 0..S {
            // Apply permutation p to the identity column
            let mut b = [T::zero(); 4];
            for i in 0..S {
                if p[i] == col {
                    b[i] = T::one();
                    break;
                }
            }
    
            // Forward substitution: solve L * y = b
            let mut y = [T::zero(); 4];
            for row in 0..S {
                y[row] = b[row];
                for k in 0..row {
                    y[row] = y[row] - l.data[k][row] * y[k];
                }
            }
    
            // Backward substitution: solve U * x = y
            let mut x = [T::zero(); 4];
            for row in (0..S).rev() {
                x[row] = y[row];
                for k in (row + 1)..S {
                    x[row] = x[row] - u.data[k][row] * x[k];
                }
                x[row] = x[row] / u.data[row][row];
            }
    
            // Assign to inverse matrix (column-major)
            for row in 0..S {
                inverse.data[col][row] = x[row];
            }
        }
    
        Some(inverse)
    }

    pub fn identity() -> Self {
        let mut data = [[T::zero(); S]; S];

        for i in 0..S {
            data[i][i] = T::one();
        }

        return Matrix::from_row(data);
    }

    pub fn trace(&self) -> T {
        let mut result = T::zero();
        for i in 0..S {
            result = result + self[i][i]
        }

        return result;
    }
}

pub fn look_at(
    position: Vector<f32, 3>,
    target: Vector<f32, 3>,
    up: Vector<f32, 3>,
) -> Matrix<f32, 4, 4> {
    let forward = (position - target).normalize();
    let right = up.cross(&forward).normalize();
    let up = forward.cross(&right);

    return Matrix::from_col([
        // First 3 columns contain basis vectors
        [right[0], up[0], forward[0], 0.],
        [right[1], up[1], forward[1], 0.],
        [right[2], up[2], forward[2], 0.],
        // Fourth column contains translation
        [
            -position.dot(&right),
            -position.dot(&up),
            -position.dot(&forward), // RH system uses positive Z forward
            1.,
        ],
    ]);
}

pub fn projection(fov: f32, ratio: f32, near: f32, far: f32) -> Matrix<f32, 4, 4> {
    let tan_half_fov = (fov / 2.0).tan();
    let fov_factor = 1. / tan_half_fov;
    let range = near - far;

    Matrix::from_col([
        [fov_factor / ratio, 0., 0., 0.],
        [0., fov_factor, 0., 0.], // Negate for Vulkan-style Y-axis
        [0., 0., far / range, -1.],
        [0., 0., (far * near) / range, 0.],
    ])
}

impl<T> Matrix<T, 4, 4>
where
    T: Float,
{
    pub fn translate(&self, position: Vector<T, 3>) -> Matrix<T, 4, 4> {
        let translation = Matrix::from_col([
            [T::one(), T::zero(), T::zero(), T::zero()],
            [T::zero(), T::one(), T::zero(), T::zero()],
            [T::zero(), T::zero(), T::one(), T::zero()],
            [position[0], position[1], position[2], T::one()],
        ]);

        return translation * *self;
    }

    pub fn rotate(&self, angle: T, axis: Vector<T, 3>) -> Matrix<T, 4, 4> {
        let c = angle.cos();
        let s = angle.sin();
        let [x, y, z] = axis.normalize().data;

        let rotation = Matrix::from_col([
            [
                x * x * (T::one() - c) + c,
                y * x * (T::one() - c) + z * s,
                z * x * (T::one() - c) - y * s,
                T::zero(),
            ],
            [
                x * y * (T::one() - c) - z * s,
                y * y * (T::one() - c) + c,
                z * y * (T::one() - c) + x * s,
                T::zero(),
            ],
            [
                x * z * (T::one() - c) + y * s,
                y * z * (T::one() - c) - x * s,
                z * z * (T::one() - c) + c,
                T::zero(),
            ],
            [T::zero(), T::zero(), T::zero(), T::one()],
        ]);

        return rotation * *self;
    }
}
