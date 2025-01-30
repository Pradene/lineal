use crate::vector::Vector;

use num::Float;
use std::fmt;
use std::ops::{Add, Index, IndexMut, Mul, Sub};

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

    fn add(self, m: Self) -> Self::Output {
        let mut result = self.clone();
        for r in 0..R {
            for c in 0..C {
                result[c][r] = result[c][r] + m[c][r];
            }
        }

        return result;
    }
}

impl<T, const R: usize, const C: usize> Sub for Matrix<T, R, C>
where
    T: Float,
{
    type Output = Self;

    fn sub(self, m: Self) -> Self::Output {
        let mut result = self.clone();
        for r in 0..R {
            for c in 0..C {
                result[c][r] = result[c][r] - m[c][r];
            }
        }

        return result;
    }
}

impl<T, const R: usize, const C: usize, const P: usize> Mul<Matrix<T, C, P>> for Matrix<T, R, C>
where
    T: Float,
{
    type Output = Matrix<T, R, P>;

    // Matrix multiplication: self (R x C) * matrix (C x P) -> result (R x P)
    fn mul(self, matrix: Matrix<T, C, P>) -> Self::Output {
        let mut result = Matrix {
            data: [[T::zero(); R]; P],
        };

        for p in 0..P {
            for r in 0..R {
                let mut sum = T::zero();
                for c in 0..C {
                    sum = sum + self[c][r] * matrix[p][c];
                }
                result[p][r] = sum;
            }
        }

        return result;
    }
}

impl<T, const R: usize, const C: usize> Mul<Vector<T, C>> for Matrix<T, R, C>
where
    T: Float,
{
    type Output = Vector<T, R>;

    fn mul(self, vector: Vector<T, C>) -> Self::Output {
        let mut result = Vector {
            data: [T::zero(); R],
        };

        for r in 0..R {
            for c in 0..C {
                result[r] = result[r] + self[r][c] * vector[c];
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

    fn mul(self, scalar: T) -> Self {
        let mut result = self.clone();
        for r in 0..R {
            for c in 0..C {
                result[r][c] = result[r][c] * scalar;
            }
        }

        return result;
    }
}

impl<T, const R: usize, const C: usize> PartialEq for Matrix<T, R, C>
where
    T: PartialEq,
{
    fn eq(&self, matrix: &Self) -> bool {
        return self.data == matrix.data;
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

impl<T, const S: usize> Matrix<T, S, S>
where
    T: Float,
{
    pub fn trace(&self) -> T {
        let mut result = T::zero();
        for i in 0..S {
            result = result + self[i][i]
        }

        return result;
    }
}

impl<T, const R: usize, const C: usize> Matrix<T, R, C>
where
    T: Float + Default + PartialOrd,
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
    fn lu_decomposition(&self) -> (Matrix<T, S, S>, Matrix<T, S, S>, usize) {
        let mut l = Matrix::from_row([[T::zero(); S]; S]);
        let mut u = self.clone();
        let mut permutation_count = 0;

        for i in 0..S {
            // Pivoting (partial pivoting)
            let mut max_row = i;
            for k in i + 1..S {
                if u[k][i].abs() > u[max_row][i].abs() {
                    max_row = k;
                }
            }

            if max_row != i {
                u.data.swap(i, max_row);
                permutation_count += 1;
            }

            // Compute L and U
            for j in i..S {
                l[j][i] = u[j][i] / u[i][i];
            }

            for j in i + 1..S {
                for k in i..S {
                    u[j][k] = u[j][k] - l[j][i] * u[i][k];
                }
            }
        }

        // Set diagonal of L to 1
        for i in 0..S {
            l[i][i] = T::one();
        }

        return (l, u, permutation_count);
    }

    // Function to compute the determinant using LU Decomposition
    pub fn determinant(&self) -> T {
        let (_, u, permutation_count) = self.lu_decomposition();
        let mut determinant = T::one();

        // Product of diagonal elements of U
        for i in 0..S {
            determinant = determinant * u[i][i];
        }

        // Adjust for row swaps
        if permutation_count % 2 != 0 {
            determinant = -determinant;
        }

        return determinant;
    }

    // Function to compute the inverse using LU Decomposition
    pub fn inverse(&self) -> Option<Matrix<T, S, S>> {
        let det = self.determinant();
        if det == T::zero() {
            return None;
        }

        let (l, u, _) = self.lu_decomposition();
        let mut inverse = [[T::zero(); S]; S];

        // Solve for each column of the identity matrix
        for i in 0..S {
            let mut b = [T::zero(); S];
            b[i] = T::one();

            // Solve L * y = b using forward substitution
            let mut y = [T::zero(); S];
            for j in 0..S {
                y[j] = b[j];
                for k in 0..j {
                    y[j] = y[j] - l[j][k] * y[k];
                }
            }

            // Solve U * x = y using backward substitution
            let mut x = [T::zero(); S];
            for j in (0..S).rev() {
                x[j] = y[j] / u[j][j];
                for k in (j + 1..S).rev() {
                    x[j] = x[j] - u[j][k] * x[k] / u[j][j];
                }
            }

            // Assign the solution to the inverse matrix
            for j in 0..S {
                inverse[j][i] = x[j];
            }
        }

        return Some(Matrix { data: inverse });
    }
}

impl<T> Matrix<T, 4, 4>
where
    T: Float,
{
    pub fn look_at(
        position: Vector<T, 3>,
        target: Vector<T, 3>,
        up: Vector<T, 3>,
    ) -> Matrix<T, 4, 4> {
        let forward = (target - position).normalize();
        let right = up.cross(&forward).normalize();
        let up = forward.cross(&right);

        return Matrix::from_col([
            [right[0], right[1], right[2], -position.dot(&right)],
            [up[0], up[1], up[2], -position.dot(&up)],
            [
                -forward[0],
                -forward[1],
                -forward[2],
                -position.dot(&forward),
            ],
            [T::zero(), T::zero(), T::zero(), T::one()],
        ]);
    }

    pub fn projection(fov: T, ratio: T, near: T, far: T) -> Matrix<T, 4, 4> {
        let fov_factor = T::from(1.).unwrap() / (fov / T::from(2.).unwrap()).tan();

        return Matrix::from_col([
            [fov_factor / ratio, T::zero(), T::zero(), T::zero()],
            [T::zero(), fov_factor, T::zero(), T::zero()],
            [
                T::zero(),
                T::zero(),
                (far + near) / (near - far),
                (T::from(2.).unwrap() * far * near) / (near - far),
            ],
            [T::zero(), T::zero(), T::from(-1.).unwrap(), T::zero()],
        ]);
    }

    pub fn rotate(&mut self, angle: f32, axis: Vector<T, 3>) -> Matrix<T, 4, 4> {
        let c = T::from(angle.cos()).unwrap();
        let s = T::from(angle.sin()).unwrap();
        let [x, y, z] = axis.data;

        // Rotation matrix components
        let rotation_matrix = Matrix::from_col([
            [
                x * x * (T::one() - c) + c,
                x * y * (T::one() - c) - z * s,
                x * z * (T::one() - c) + y * s,
                T::zero(),
            ],
            [
                y * x * (T::one() - c) + z * s,
                y * y * (T::one() - c) + c,
                y * z * (T::one() - c) - x * s,
                T::zero(),
            ],
            [
                z * x * (T::one() - c) - y * s,
                z * y * (T::one() - c) + x * s,
                z * z * (T::one() - c) + c,
                T::zero(),
            ],
            [T::zero(), T::zero(), T::zero(), T::one()],
        ]);

        return *self * rotation_matrix;
    }
}

impl<T, const C: usize> Matrix<T, C, C>
where
    T: Float,
{
    pub fn identity() -> Self {
        let mut data = [[T::zero(); C]; C];

        for i in 0..C {
            data[i][i] = T::one();
        }

        return Matrix::from_row(data);
    }
}
