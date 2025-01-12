use num::Signed;
use std::ops::{
    Neg, 
    Add,
    Sub,
    Mul,
    Index,
    IndexMut
};

#[derive(Debug, Clone, Copy)]
pub struct Vector<T, const N: usize> {
    pub data: [T; N],
}

impl<T, const N: usize> Vector<T, N>
where
T:
    Copy +
    Default
{
    fn new(data: [T; N]) -> Self {
        Self {
            data: data
        }
    }
}

impl<T, const N: usize> From<[T; N]> for Vector<T, N> {
    fn from(data: [T; N]) -> Self {
        Self {
            data: data
        }
    }
}

impl<T, const N: usize> Index<usize> for Vector<T, N> {
    type Output = T;

    fn index(&self, i: usize) -> &Self::Output {
        if i < N {
            &self.data[i] // Return a reference to the element at index `i`
        } else {
            panic!("Index out of bounds");
        }
    }
}

// Implement IndexMut trait for mutable access to elements (self[i] = value)
impl<T, const N: usize> IndexMut<usize> for Vector<T, N> {
    fn index_mut(&mut self, i: usize) -> &mut Self::Output {
        if i < N {
            &mut self.data[i] // Return a mutable reference to the element at index `i`
        } else {
            panic!("Index out of bounds");
        }
    }
}

impl<T, const N: usize> Add for Vector<T, N>
where T:
    Add<Output = T> +
    Copy
{
    type Output = Self;

    fn add(self, v: Self) -> Self {
        let mut result = self.clone();
        for i in 0..N {
            result[i] = result[i] + v[i];
        }

        result
    }
}

impl<T, const N: usize> Sub for Vector<T, N>
where T:
    Sub<Output = T> +
    Copy
{
    type Output = Self;

    fn sub(self, v: Self) -> Self {
        let mut result = self.clone();
        for i in 0..N {
            result[i] = result[i] - v[i];
        }

        result
    }
}

impl<T, const N: usize> Mul<T> for Vector<T, N>
where T:
    Mul<Output = T> +
    Copy
{
    type Output = Self;

    fn mul(self, scalar: T) -> Self {
        let mut result = self.clone();
        for i in 0..N {
            result[i] = result[i] * scalar;
        }

        result
    }
}

impl<T, const N: usize> Vector<T, N>
where
T: 
    Mul<Output = T> +
    Add<Output = T> +
    Copy + 
    Into<f32> +
    Default
{
    pub fn dot(&self, vector: Vector<T, N>) -> f32 {
        self.data
            .iter()
            .zip(vector.data.iter())
            .fold(0., |sum, (&x, &y)| sum + x.into() * y.into())
    }
}

pub fn linear_combination<T, const N: usize>(
    vectors: &[Vector<T, N>],
    scalars: &[T]
) -> Vector<T, N>
where
T:
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

    Vector::from(result)
}

impl<T, const N: usize> Vector<T, N>
where
T:
    Neg<Output = T> +
    Copy + 
    Signed +
    Into<f32>
{
    pub fn norm_1(&self) -> f32 {
        self.data
            .iter()
            .fold(0., |sum, &x| sum + x.abs().into())
    }
}

impl<T, const N: usize> Vector<T, N>
where
T:
    Copy +
    Into<f32> +
    Signed
{
    pub fn norm(&self) -> f32 {
        self.data
            .iter()
            .fold(0., |sum, &x| sum + x.abs().into().powf(2.))
            .powf(0.5)
    }
}

impl<T, const N: usize> Vector<T, N>
where
T:
    Copy +
    Into<f32> +
    Signed +
    PartialOrd
{
    pub fn norm_inf(&self) -> f32 {
        self.data
            .iter()
            .fold(0., |sum, &x| f32::max(sum, x.abs().into()))
    }
}


pub fn angle_cos<T, const N: usize>(
    u: &Vector<T, N>,
    v: &Vector<T, N>
) -> f32
where
T:
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

pub fn cross_product<T>(
    u: &Vector<T, 3>,
    v: &Vector<T, 3>
) -> Vector<T, 3>
where
T:
    Copy +
    Default +
    Signed +
    Into<f32>
{
    Vector {
        data: [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],    
        ],
    }
}

pub fn lerp<T>(x: T, y: T, t: f32) -> T
where
T:
    Add<Output = T> +
    Sub<Output = T> +
    Mul<f32, Output = T> +
    Copy
{
    x + (y - x) * t
}