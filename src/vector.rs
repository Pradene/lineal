use std::fmt;
use num::Float;
use std::ops::{
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

impl<T, const N: usize> fmt::Display for Vector<T, N> 
where T:
    fmt::Display
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        
        for i in 0..N {
            if i != 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", self[i])?;
        }
        
        write!(f, "]")
    }
}

impl<T, const N: usize> From<[T; N]> for Vector<T, N> {
    fn from(data: [T; N]) -> Self {
        Self {data}
    }
}

impl<T, const N: usize> TryFrom<Vec<T>> for Vector<T, N> {
    type Error = String;

    fn try_from(vec: Vec<T>) -> Result<Self, Self::Error> {
        if vec.len() == N {
            let data: [T; N] = vec.try_into().map_err(|_| "Incorrect length")?;
            Ok(Self {
                data: data,
            })

        } else {
            Err("Vector length does not match the expected size".to_string())
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
    Copy +
    Add<Output = T>
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
    Copy +
    Sub<Output = T>
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
    Copy +
    Mul<Output = T>
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

impl<T, const N: usize> PartialEq for Vector<T, N>
where T:
    PartialEq
{
    fn eq(&self, vector: &Self) -> bool {
        self.data == vector.data
    }
}

impl<T, const N: usize> Vector<T, N> {
    pub fn new(data: [T; N]) -> Self {
        Self {data}
    }
}

impl<T, const N: usize> Vector<T, N>
where
T:
    Float
{
    pub fn dot(&self, vector: &Vector<T, N>) -> T {
        self.data
            .iter()
            .zip(vector.data.iter())
            .fold(T::zero(), |sum, (&x, &y)| sum + x * y)
    }

    pub fn norm_1(&self) -> T {
        self.data
            .iter()
            .fold(T::zero(), |sum, &x| sum + x.abs())
    }

    pub fn norm(&self) -> T {
        self.data
            .iter()
            .fold(T::zero(), |sum, &x| {
                sum + x * x
            })
            .powf(T::from(0.5).unwrap())
    }

    pub fn norm_inf(&self) -> T {
        self.data
            .iter()
            .fold(T::zero(), |sum, &x| T::max(sum, x.abs()))
    }

    pub fn cosine(&self, v: &Vector<T, N>) -> T {
        let dot_product = self.dot(v);
        let u_length = self.norm();
        let v_length = v.norm();
        dot_product / (u_length * v_length)
    }
}

impl<T> Vector<T, 3> 
where T:
    Float
{
    pub fn cross(&self, v: &Vector<T, 3>) -> Vector<T, 3>
    {
        Vector {
            data: [
                self[1] * v[2] - self[2] * v[1],
                self[2] * v[0] - self[0] * v[2],
                self[0] * v[1] - self[1] * v[0],    
            ],
        }
    }
}

pub fn linear_combination<T, const N: usize>(vectors: &[Vector<T, N>], scalars: &[T]) -> Vector<T, N>
where T:
    Default +
    Copy +
    Mul<Output = T> +
    Add<Output = T>
{    
    // Check vectors length is not equal to 0
    assert!(!vectors.is_empty(), "Vectors is empty");

    // Check if vectors length and scalars length are equal
    assert_eq!(vectors.len(), scalars.len(), "Vectors length and scalars length must be equal");
    
    let mut result = Vector::from([T::default(); N]);

    for (scalar, vector) in scalars.iter().zip(vectors.iter()) {
        result.data.iter_mut()
            .zip(vector.data.iter())
            .for_each(|(res, &v)| *res = *res + scalar.clone() * v.clone());
    }

    result
}
