use {
    crate::{number::Number, vector::Vector},
    std::{
        fmt,
        ops::{Add, AddAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign},
    },
};

#[derive(Debug, Clone, Copy)]
pub struct Matrix<T, const R: usize, const C: usize> {
    pub data: [[T; R]; C],
}

impl<T: Number, const R: usize, const C: usize> Matrix<T, R, C> {
    pub fn new() -> Self {
        Matrix {
            data: [[T::ZERO; R]; C],
        }
    }

    pub fn from_col(cols: [[T; R]; C]) -> Self {
        Matrix { data: cols }
    }

    pub fn from_row(rows: [[T; C]; R]) -> Self {
        let mut data = [[T::ZERO; R]; C];
        for r in 0..R {
            for c in 0..C {
                data[c][r] = rows[r][c];
            }
        }

        Matrix { data }
    }
}

impl<T: fmt::Display, const R: usize, const C: usize> fmt::Display for Matrix<T, R, C> {
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

        Ok(())
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

impl<T: Number, const R: usize, const C: usize> Add for Matrix<T, R, C> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let mut result = self.clone();
        for r in 0..R {
            for c in 0..C {
                result[c][r] = result[c][r] + rhs[c][r];
            }
        }

        result
    }
}

impl<T, const R: usize, const C: usize> AddAssign for Matrix<T, R, C>
where
    T: Number,
{
    fn add_assign(&mut self, rhs: Self) {
        *self = self.clone() + rhs;
    }
}

impl<T, const R: usize, const C: usize> Sub for Matrix<T, R, C>
where
    T: Number,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut result = self.clone();
        for r in 0..R {
            for c in 0..C {
                result[c][r] = result[c][r] - rhs[c][r];
            }
        }

        result
    }
}

impl<T, const R: usize, const C: usize> SubAssign for Matrix<T, R, C>
where
    T: Number,
{
    fn sub_assign(&mut self, rhs: Self) {
        *self = self.clone() - rhs;
    }
}

impl<T, const R: usize, const C: usize, const P: usize> Mul<Matrix<T, C, P>> for Matrix<T, R, C>
where
    T: Number,
{
    type Output = Matrix<T, R, P>;

    fn mul(self, rhs: Matrix<T, C, P>) -> Self::Output {
        let mut result = Matrix {
            data: [[T::ZERO; R]; P],
        };

        for p in 0..P {
            for r in 0..R {
                let mut sum = T::ZERO;
                for c in 0..C {
                    sum = sum + self[c][r] * rhs[p][c];
                }
                result[p][r] = sum;
            }
        }

        result
    }
}

impl<T: Number, const R: usize, const C: usize, const P: usize> MulAssign<Matrix<T, C, P>>
    for Matrix<T, R, C>
{
    fn mul_assign(&mut self, rhs: Matrix<T, C, P>) {
        for p in 0..P {
            for r in 0..R {
                let mut sum = T::ZERO;
                for c in 0..C {
                    sum = sum + self[c][r] * rhs[p][c];
                }
                self[p][r] = sum;
            }
        }
    }
}

impl<T: Number, const R: usize, const C: usize> Mul<Vector<T, C>> for Matrix<T, R, C> {
    type Output = Vector<T, R>;

    fn mul(self, rhs: Vector<T, C>) -> Self::Output {
        let mut result = Vector { data: [T::ZERO; R] };

        for r in 0..R {
            for c in 0..C {
                result[r] = result[r] + self[c][r] * rhs[c];
            }
        }

        result
    }
}

impl<T: Number, const R: usize, const C: usize> Mul<T> for Matrix<T, R, C> {
    type Output = Self;

    fn mul(self, rhs: T) -> Self {
        let mut result = self.clone();
        for r in 0..R {
            for c in 0..C {
                result[c][r] = result[c][r] * rhs;
            }
        }

        result
    }
}

impl<T, const R: usize, const C: usize> MulAssign<T> for Matrix<T, R, C>
where
    T: Number,
{
    fn mul_assign(&mut self, rhs: T) {
        for r in 0..R {
            for c in 0..C {
                self[c][r] = self[c][r] * rhs;
            }
        }
    }
}

impl<T, const R: usize, const C: usize> PartialEq for Matrix<T, R, C>
where
    T: PartialEq + Number + Into<f64>,
    f64: From<T>,
{
    fn eq(&self, other: &Self) -> bool {
        for col in 0..C {
            for row in 0..R {
                let diff: f64 = f64::from((self.data[col][row] - other.data[col][row]).abs());
                if diff > std::f32::EPSILON as f64 {
                    return false;
                }
            }
        }

        true
    }
}

impl<T, const R: usize, const C: usize> Matrix<T, R, C>
where
    T: Number,
{
    pub fn transpose(&self) -> Matrix<T, C, R> {
        let mut data = [[T::ZERO; C]; R];

        for c in 0..C {
            for r in 0..R {
                data[r][c] = self[c][r];
            }
        }

        Matrix::from_col(data)
    }
}

impl<T, const R: usize, const C: usize> Matrix<T, R, C>
where
    T: Number,
{
    pub fn row_echelon(&self) -> Self {
        let mut result = self.clone();
        let mut pivot_rows = vec![None; C]; // Track which row contains a pivot for each column

        // Forward phase: Get to row echelon form
        let mut pivot_row = 0;
        for col in 0..C {
            // Stop if we've processed all rows
            if pivot_row >= R {
                break;
            }

            // Find first non-zero element in current column (starting from pivot_row)
            let mut pivot = None;
            for r in pivot_row..R {
                if result.data[col][r].abs() > T::EPSILON {
                    pivot = Some(r);
                    break;
                }
            }

            // If no pivot found, continue to next column
            if let Some(pivot_idx) = pivot {
                // Record which row contains a pivot for this column
                pivot_rows[col] = Some(pivot_row);

                // Swap rows if needed
                if pivot_idx != pivot_row {
                    result.data.swap(pivot_idx, pivot_row);
                }

                // Scale the pivot row to make pivot element 1
                let pivot_val = result.data[col][pivot_row];
                for c in col..C {
                    result.data[c][pivot_row] = result.data[c][pivot_row] / pivot_val;
                }

                // Eliminate in other rows (below)
                for r in (pivot_row + 1)..R {
                    let factor = result.data[col][r];
                    if factor.abs() > T::EPSILON {
                        for c in col..C {
                            result.data[c][r] =
                                result.data[c][r] - factor * result.data[c][pivot_row];
                        }
                    }
                }

                pivot_row += 1;
            }
        }

        // Backward phase: Reduce to reduced row echelon form (eliminate above pivots)
        for col in (0..C).rev() {
            if let Some(pivot_row) = pivot_rows[col] {
                // Eliminate entries above pivot
                for r in 0..pivot_row {
                    let factor = result.data[col][r];
                    if factor.abs() > T::EPSILON {
                        for c in col..C {
                            result.data[c][r] =
                                result.data[c][r] - factor * result.data[c][pivot_row];
                        }
                    }
                }
            }
        }

        result
    }

    pub fn rank(&self) -> usize {
        let rref = self.row_echelon();
        let mut rank = 0;

        for r in 0..R {
            let mut all_zero = true;
            for c in 0..C {
                if rref.data[c][r].abs() > T::EPSILON {
                    all_zero = false;
                    break;
                }
            }
            if !all_zero {
                rank += 1;
            }
        }

        rank
    }
}

impl<T, const S: usize> Matrix<T, S, S>
where
    T: Number,
{
    fn lu_decomposition(&self) -> (Self, Self, Vec<usize>, usize) {
        let mut l = Matrix::identity();
        let mut u = self.clone();
        let mut p: Vec<usize> = (0..S).collect();
        let mut s = 0;

        for i in 0..S {
            let mut max_row = i;
            for row in i..S {
                if u.data[i][row].abs() > u.data[i][max_row].abs() {
                    max_row = row;
                }
            }

            if max_row != i {
                for col in 0..S {
                    u.data[col].swap(i, max_row);
                }
                for col in 0..i {
                    l.data[col].swap(i, max_row);
                }
                p.swap(i, max_row);
                s += 1;
            }

            let pivot = u.data[i][i];
            if pivot == T::ZERO {
                return (l, u, p, s);
            }

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

        for i in 0..S {
            l.data[i][i] = T::ONE;
        }

        (l, u, p, s)
    }

    pub fn determinant(&self) -> T {
        let (_, u, _, s) = self.lu_decomposition();
        let mut determinant = T::ONE;

        for i in 0..S {
            determinant = determinant * u[i][i];
        }

        determinant = determinant * (-T::ONE).powi(s as i32);

        determinant
    }

    pub fn inverse(&self) -> Option<Self> {
        let (l, u, p, _) = self.lu_decomposition();
        let mut inverse = Matrix::new();

        let mut det = T::ONE;
        for i in 0..S {
            det = det * u.data[i][i];
        }
        if det == T::ZERO {
            return None;
        }

        for col in 0..S {
            let mut b = [T::ZERO; S];
            for i in 0..S {
                if p[i] == col {
                    b[i] = T::ONE;
                    break;
                }
            }

            let mut y = [T::ZERO; S];
            for row in 0..S {
                y[row] = b[row];
                for k in 0..row {
                    y[row] = y[row] - l.data[k][row] * y[k];
                }
            }

            let mut x = [T::ZERO; S];
            for row in (0..S).rev() {
                x[row] = y[row];
                for k in (row + 1)..S {
                    x[row] = x[row] - u.data[k][row] * x[k];
                }
                x[row] = x[row] / u.data[row][row];
            }

            for row in 0..S {
                inverse.data[col][row] = x[row];
            }
        }

        Some(inverse)
    }

    pub fn identity() -> Self {
        let mut data = [[T::ZERO; S]; S];

        for i in 0..S {
            data[i][i] = T::ONE;
        }

        Matrix::from_row(data)
    }

    pub fn trace(&self) -> T {
        let mut result = T::ZERO;
        for i in 0..S {
            result = result + self[i][i]
        }

        result
    }
}

impl<T: Number> Matrix<T, 4, 4> {
    pub fn look_at(position: Vector<T, 3>, target: Vector<T, 3>, up: Vector<T, 3>) -> Self {
        let forward = (target - position).normalize();
        let right = up.cross(&forward).normalize();
        let up = forward.cross(&right);

        Matrix::from_col([
            [right[0], right[1], right[2], T::ZERO],
            [up[0], up[1], up[2], T::ZERO],
            [forward[0], forward[1], forward[2], T::ZERO],
            [
                -position.dot(&right),
                -position.dot(&up),
                -position.dot(&forward),
                T::ONE,
            ],
        ])
    }

    pub fn projection(fov: T, ratio: T, near: T, far: T) -> Self {
        let two = T::ONE + T::ONE;
        let scale = T::ONE / (fov / two).tan();
        let range = near - far;
        let two_near_far = two * near * far;

        Matrix::from_col([
            [scale / ratio, T::ZERO, T::ZERO, T::ZERO],
            [T::ZERO, scale, T::ZERO, T::ZERO],
            [T::ZERO, T::ZERO, (far + near) / range, -T::ONE],
            [T::ZERO, T::ZERO, two_near_far / range, T::ZERO],
        ])
    }

    pub fn translate(&self, position: Vector<T, 3>) -> Matrix<T, 4, 4> {
        let translation = Matrix::from_col([
            [T::ONE, T::ZERO, T::ZERO, T::ZERO],
            [T::ZERO, T::ONE, T::ZERO, T::ZERO],
            [T::ZERO, T::ZERO, T::ONE, T::ZERO],
            [position[0], position[1], position[2], T::ONE],
        ]);

        translation * self.clone()
    }

    pub fn rotate(&self, angle: T, axis: Vector<T, 3>) -> Matrix<T, 4, 4> {
        let c = angle.cos();
        let s = angle.sin();
        let [x, y, z] = axis.normalize().data;

        let rotation = Matrix::from_col([
            [
                x * x * (T::ONE - c) + c,
                y * x * (T::ONE - c) + z * s,
                z * x * (T::ONE - c) - y * s,
                T::ZERO,
            ],
            [
                x * y * (T::ONE - c) - z * s,
                y * y * (T::ONE - c) + c,
                z * y * (T::ONE - c) + x * s,
                T::ZERO,
            ],
            [
                x * z * (T::ONE - c) + y * s,
                y * z * (T::ONE - c) - x * s,
                z * z * (T::ONE - c) + c,
                T::ZERO,
            ],
            [T::ZERO, T::ZERO, T::ZERO, T::ONE],
        ]);

        rotation * self.clone()
    }
}
