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
    fn row_echelon(&mut self) -> Matrix<T, M, N> {

        let mut pivot_row = 0;
        
        for col in 0..N {
            if pivot_row >= M {
                break;
            }

            let mut pivot = None;
            for row in pivot_row..M {
                if self.data[row][col].abs() > T::zero() {
                    pivot = Some(row);
                    break;
                }
            }

            if pivot.is_none() {
                continue;
            }

            let pivot = pivot.unwrap();
            if pivot != pivot_row {
                self.data.swap(pivot, pivot_row);
            }

            let pivot_value = self.data[pivot_row][col];
            if pivot_value == T::zero() {
                continue;
            }

            for j in col..N {
                self.data[pivot_row][j] = self.data[pivot_row][j] / pivot_value;
            }

            for row in 0..M {
                if row == pivot_row {
                    continue
                }

                let factor = self.data[row][col];
                for j in col..N {
                    self.data[row][j] = self.data[row][j] - factor * self.data[pivot_row][j];
                }
            }

            pivot_row += 1
        }
        
        Matrix {
            data: self.data
        }
    }
}

impl<T, const N: usize> Matrix<T, N, N>
where T:
    Copy +
    Signed +
    PartialOrd +
    Into<f32>
{
    fn determinant(&mut self) -> f32 {
        let mut pivot_row = 0;
        let mut det = 1.;
        
        for col in 0..N {
            if pivot_row >= N {
                break;
            }

            let mut pivot = None;
            for row in pivot_row..N {
                if self.data[row][col].abs() > T::zero() {
                    pivot = Some(row);
                    break;
                }
            }

            if pivot.is_none() {
                return 0.0;
            }

            let pivot = pivot.unwrap();

            if pivot != pivot_row {
                self.data.swap(pivot, pivot_row);
                det = det * -1.;
            }

            let pivot_value = self.data[pivot_row][col];
            if pivot_value == T::zero() {
                continue;
            }

            for row in 0..N {
                if pivot_row == row {
                    continue
                }

                let factor = self.data[row][col] / pivot_value;
                for j in col..N {
                    self.data[row][j] = self.data[row][j] - factor * self.data[pivot_row][j];
                }
            }

            pivot_row += 1;
        }

        // Multiply the diagonal elements to get the determinant
        for i in 0..N {
            det *= self.data[i][i].into();
        }

        det
    }
}


fn main() {
    let mut u = Matrix::from([
        [8., 5., -2.],
        [4., 7., 20.],
        [7., 6., 1.],
    ]);
    println!("{}", u.determinant());   
}

