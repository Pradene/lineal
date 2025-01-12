use std::ops::{Neg, Add, Sub, Mul};
use num::Signed;

#[derive(Debug, Clone, Copy)]
struct Vector<T, const N: usize> {
    data: [T; N],
}

impl<T, const N: usize> Vector<T, N> {
    fn new(data: [T; N ]) -> Self {
        Self {
            data: data
        }
    }
}

impl<T, const N: usize> From<[T; N]> for Vector<T, N> {
    fn from(arr: [T; N]) -> Self {
        Vector { data: arr }
    }
}

impl<T, const N: usize> Add for Vector<T, N>
where T:
    Add<Output = T> + Copy
{
    type Output = Self;

    fn add(self, v: Self) -> Self {
        let mut result = self.data;
        for i in 0..N {
            result[i] = self.data[i] + v.data[i];
        }

        Vector {
            data: result
        }
    }
}

impl<T, const N: usize> Sub for Vector<T, N>
where T:
    Sub<Output = T> + Copy
{
    type Output = Self;

    fn sub(self, v: Self) -> Self {
        let mut result = self.data;
        for i in 0..N {
            result[i] = self.data[i] - v.data[i];
        }

        Vector {
            data: result
        }
    }
}

impl<T, const N: usize> Mul<T> for Vector<T, N>
where T:
    Mul<Output = T> + Copy
{
    type Output = Self;

    fn mul(self, scalar: T) -> Self {
        let mut result = self.data;
        for i in 0..N {
            result[i] = self.data[i] * scalar;
        }

        Vector {
            data: result
        }
    }
}

impl<T, const N: usize> Vector<T, N>
where T:
    Mul<Output = T> + Copy
{
    fn scl(self, scalar: T) -> Self {
        self * scalar
    }
}

impl<T, const N: usize> Vector<T, N>
where T: 
    Mul<Output = T> +
    Add<Output = T> +
    Copy + 
    Into<f32> +
    Default
{
    fn dot(&self, v: Vector<T, N>) -> f32 {
        assert_eq!(self.data.len(), v.data.len(), "Vectors must be of same length");

        self.data
        .iter()
        .zip(v.data.iter())
        .fold(0., |sum, (&x, &y)| sum + x.into() * y.into())
    }
}

fn linear_combination<T, const N: usize>(
    vectors: &[Vector<T, N>],
    scalars: &[T]
) -> Vector<T, N>
where T:
    Add<Output = T> +
    Mul<Output = T> +
    Default +
    Copy
{
    // Check vectors length is not equal to 0
    assert!(!vectors.is_empty(), "Vectors is empty");

    // Check if vectors length and scalars length are equal
    assert_eq!(vectors.len(), scalars.len(), "Vectors length and scalars length must be equal");
    
    let mut result = [T::default(); N];

    for (scalar, vector) in scalars.iter().zip(vectors.iter()) {
        result.iter_mut()
        .zip(vector.data.iter())
        .for_each(|(res, &v)| *res = *res + scalar.clone() * v.clone());
    }

    Vector::new(result)
}

impl<T, const N: usize> Vector<T, N>
where T:
    Neg<Output = T> +
    Copy + 
    Signed +
    Into<f32>
{
    fn norm_1(&self) -> f32 {
        self.data.iter().fold(0., |sum, &x| sum + x.abs().into())
    }
}

impl<T, const N: usize> Vector<T, N>
where T:
    Copy +
    Into<f32> +
    Signed
{
    fn norm(&self) -> f32 {
        self.data
        .iter()
        .fold(0., |sum, &x| sum + x.abs().into().powf(2.))
        .powf(0.5)
    }
}

impl<T, const N: usize> Vector<T, N>
where T:
    Copy +
    Into<f32> +
    Signed +
    PartialOrd
{
    fn norm_inf(&self) -> f32 {
        self.data
        .iter()
        .fold(0., |sum, &x| f32::max(sum, x.abs().into()))
    }
}


fn angle_cos<T, const N: usize>(
    u: &Vector<T, N>,
    v: &Vector<T, N>
) -> f32
where T:
    Copy +
    Default +
    Into<f32> + 
    Signed
{
    let dot_product = u.dot(*v);
    let u_length = u.norm();
    let v_length = v.norm();
    dot_product / (u_length * v_length)
}

fn cross_product<T>(
    u: &Vector<T, 3>,
    v: &Vector<T, 3>
) -> Vector<T, 3>
where T:
    Copy +
    Default +
    Signed +
    Into<f32>
{
    Vector {
        data: [
            u.data[1] * v.data[2] - u.data[2] * v.data[1],
            u.data[2] * v.data[0] - u.data[0] * v.data[2],
            u.data[0] * v.data[1] - u.data[1] * v.data[0],    
        ],
    }
}


#[derive(Debug, Clone)]
struct Matrix<T, const M: usize, const N: usize> {
    data:  [[T; N]; M],
}

impl<T, const M: usize, const N: usize> Matrix<T, M, N> {
    fn new(data: [[T; N]; M]) -> Self {
        Self {
            data: data
        }
    }
}

impl<T, const M: usize, const N: usize> From<[[T; N]; M]> for Matrix<T, M, N> {
    fn from(data: [[T; N]; M]) -> Self {
        Matrix {
            data: data
        }
    }
}

impl<T, const M: usize, const N: usize> Add for Matrix<T, M, N>
where T: Add<Output = T> + Copy {
    type Output = Self;

    fn add(self, m: Self) -> Self {
        let mut result = self.data;
        for i in 0..M {
            for j in 0..N {
                result[i][j] = self.data[i][j] + m.data[i][j];
            }
        }

        Matrix {
            data: result
        }
    }
}

impl<T, const M: usize, const N: usize> Sub for Matrix<T, M, N>
where T: Sub<Output = T> + Copy {
    type Output = Self;

    fn sub(self, m: Self) -> Self {
        let mut result = self.data;
        for i in 0..M {
            for j in 0..N {
                result[i][j] = self.data[i][j] - m.data[i][j];
            }
        }

        Matrix {
            data: result
        }
    }
}

impl<T, const M: usize, const N: usize> Matrix<T, M, N>
where T:
    Copy + Mul<Output = T>
{
    fn scl(&self, scalar: T) -> Self {
        let mut data = self.data;
        for i in 0..M {
            for j in 0..N {
                data[i][j] = self.data[i][j] * scalar;
            }
        }

        Matrix {
            data: data
        }
    }
}

impl<T, const M: usize, const N: usize> Matrix<T, M, N>
where T:
    Copy +
    Mul<Output = T> +
    Add<Output = T> +
    Default
{
    fn mul_mat(&self, matrix: &Matrix<T, M, N>) -> Matrix<T, M, N> {

        let mut data = [[T::default(); N]; M];

        for i in 0..M {
            for j in 0..N {
                for k in 0..N {
                    data[i][j] = data[i][j] + self.data[i][k] * matrix.data[k][j];
                }
            }
        }

        Matrix {
            data: data
        }
    }
}

impl<T, const M: usize, const N: usize> Matrix<T, M, N>
where T:
    Copy +
    Default +
    Mul<Output = T> +
    Add<Output = T>
{
    fn mul_vec(&mut self, vector: &Vector<T, N>) -> Vector<T, M> {
        let mut data = [T::default(); M];

        for i in 0..M {
            for j in 0..N {
                data[i] = data[i] + self.data[i][j] * vector.data[j];
            }
        }

        Vector {
            data: data
        }
    }
}

impl<T, const M: usize, const N: usize> Matrix<T, M, N>
where T:
    Copy +
    Default +
    Mul<Output = T> +
    Add<Output = T>
{
    fn transpose(&self) -> Matrix<T, N, M> {
        let mut data = [[T::default(); M]; N];

        for i in 0..N {
            for j in 0..M {
                data[i][j] = self.data[j][i];
            }
        }

        Matrix {
            data: data
        }
    }
}

impl<T, const N: usize> Matrix<T, N, N>
where T:
    Copy +
    Default +
    Add<Output = T>
{
    fn trace(&self) -> T {
        let mut sum = T::default();
        for i in 0..N {
            sum = sum + self.data[i][i]
        }

        sum
    }
}

fn lerp<T>(x: T, y: T, t: f32) -> T
where T:
    Add<Output = T> +
    Sub<Output = T> +
    Mul<f32, Output = T> +
    Copy
{
    x + (y - x) * t
}

// reduced row echelon form of matrix 
// Find the pivot row (row with value of col with value != 0.)
// if pivot not found skipto next col
// else make pivot equal one by multiply it (and multiply all value of row by this value)
// then substract all rows below by x * row pivot 

impl<T, const M: usize, const N: usize> Matrix<T, M, N>
where T:
    Copy +
    Signed +
    PartialOrd
{
    fn row_echelon(&self) -> Matrix<T, M, N> {
        let mut m = Matrix::from(self.data);
        let mut pivot_row = 0;
        
        for col in 0..N {
            if pivot_row >= M {
                break;
            }

            let mut pivot = None;
            for row in pivot_row..M {
                if m.data[row][col].abs() > T::zero() {
                    pivot = Some(row);
                    break;
                }
            }

            if pivot.is_none() {
                continue;
            }

            let pivot = pivot.unwrap();
            if pivot != pivot_row {
                m.data.swap(pivot, pivot_row);
            }

            let pivot_value = m.data[pivot_row][col];
            if pivot_value == T::zero() {
                continue;
            }

            for j in col..N {
                m.data[pivot_row][j] = m.data[pivot_row][j] / pivot_value;
            }

            for row in 0..M {
                if row == pivot_row {
                    continue
                }

                let factor = m.data[row][col];
                for j in col..N {
                    m.data[row][j] = m.data[row][j] - factor * m.data[pivot_row][j];
                }
            }

            pivot_row += 1
        }
        
        m
    }

    fn rank(&self) -> usize {
        let mut rank = 0;
        let rref = self.row_echelon();

        for row in 0..M {
            for col in 0..N {
                if rref.data[row][col] != T::zero() {
                    rank += 1;
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
        let mut u = Matrix::from(self.data);
        let mut permutation_count = 0;
    
        for i in 0..N {
            // Pivoting (partial pivoting)
            let mut max_row = i;
            for k in i + 1..N {
                if u.data[k][i].abs() > u.data[max_row][i].abs() {
                    max_row = k;
                }
            }

            if max_row != i {
                u.data.swap(i, max_row);
                permutation_count += 1;
            }
    
            // Compute L and U
            for j in i..N {
                l.data[j][i] = u.data[j][i] / u.data[i][i];
            }

            for j in i + 1..N {
                for k in i..N {
                    u.data[j][k] = u.data[j][k] - l.data[j][i] * u.data[i][k];
                }
            }
        }
    
        // Set diagonal of L to 1
        for i in 0..N {
            l.data[i][i] = T::one();
        }
    
        (l, u, permutation_count)
    }
    
    // Function to compute the determinant using LU Decomposition
    fn determinant(&self) -> T {
        let (l, u, permutation_count) = self.lu_decomposition();
        let mut determinant = T::one();
    
        // Product of diagonal elements of U
        for i in 0..N {
            determinant = determinant * u.data[i][i];
        }
    
        // Adjust for row swaps
        if permutation_count % 2 != 0 {
            determinant = -determinant;
        }
    
        determinant
    }
    
    // Function to compute the inverse using LU Decomposition
    fn inverse(&self) -> Option<Matrix<T, N, N>> {
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
                    y[j] = y[k] - l.data[j][k] * y[k];
                }
            }
    
            // Solve U * x = y using backward substitution
            let mut x = [T::zero(); N];
            for j in (0..N).rev() {
                x[j] = y[j] / u.data[j][j];
                for k in (j + 1..N).rev() {
                    x[j] = x[j] - u.data[j][k] * x[k] / u.data[j][j];
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

fn projection(fov: f32, ratio: f32, near: f32, far: f32) -> Matrix<f32, 4, 4> {
    let mut projection_matrix = Matrix::from([[0.; 4]; 4]);

    let fov_factor = 1. / (fov / 2.).tan();

    projection_matrix.data[0][0] = fov_factor / ratio;
    projection_matrix.data[1][1] = fov_factor;
    projection_matrix.data[2][2] = (far + near) / (near - far);
    projection_matrix.data[2][3] = (2. * far * near) / (near - far);
    projection_matrix.data[3][2] = -1.;

    // Transpose for column major order
    projection_matrix.transpose()
}

fn main() {
    let u = Matrix::from([
        [8., 5., -2.],
        [4., 7., 20.],
        [7., 6., 1.],
    ]);
    println!("{:?}", u.determinant());

    let v = Matrix::from([
        [ 8., 5., -2., 4.],
        [ 4., 2.5, 20., 4.],
        [ 8., 5., 1., 4.],
        [28., -4., 17., 1.],
    ]);
    println!("{:?}", v.determinant());

    let t = Matrix::from([
        [1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.],
    ]);
    println!("{}", t.rank());

    let z = Matrix::from([
        [ 1., 2., 0., 0.],
        [ 2., 4., 0., 0.],
        [-1., 2., 1., 1.],
    ]);
    println!("{}", z.rank());
}
