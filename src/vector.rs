use {
    crate::number::Number,
    std::{
        convert::{From, TryFrom},
        fmt,
        ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign},
    },
};

#[derive(Debug, Clone, Copy)]
pub struct Vector<T, const N: usize> {
    pub data: [T; N],
}

impl<T, const N: usize> fmt::Display for Vector<T, N>
where
    T: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for i in 0..N {
            if i != 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", self[i])?;
        }
        write!(f, "]")?;

        Ok(())
    }
}

impl<T, const N: usize> Vector<T, N> {
    pub fn new(data: [T; N]) -> Self {
        Self { data }
    }
}

impl<T, const N: usize> From<[T; N]> for Vector<T, N> {
    fn from(data: [T; N]) -> Self {
        Self { data }
    }
}

impl<T, const N: usize> TryFrom<Vec<T>> for Vector<T, N> {
    type Error = String;

    fn try_from(vec: Vec<T>) -> Result<Self, Self::Error> {
        if vec.len() == N {
            let data: [T; N] = vec.try_into().map_err(|_| "Incorrect length")?;
            Ok(Self { data })
        } else {
            Err("Vector length does not match the expected size".to_string())
        }
    }
}

impl<T, const N: usize> Index<usize> for Vector<T, N> {
    type Output = T;

    fn index(&self, i: usize) -> &Self::Output {
        if i < N {
            &self.data[i]
        } else {
            panic!("Index out of bounds");
        }
    }
}

impl<T, const N: usize> IndexMut<usize> for Vector<T, N> {
    fn index_mut(&mut self, i: usize) -> &mut Self::Output {
        if i < N {
            &mut self.data[i]
        } else {
            panic!("Index out of bounds");
        }
    }
}

impl<T, const N: usize> Add for Vector<T, N>
where
    T: Number,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        let mut result = self;
        for i in 0..N {
            result[i] += rhs[i];
        }

        result
    }
}

impl<T, const N: usize> AddAssign for Vector<T, N>
where
    T: Number,
{
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<T, const N: usize> Sub for Vector<T, N>
where
    T: Number,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        let mut result = self;
        for i in 0..N {
            result[i] -= rhs[i];
        }

        result
    }
}

impl<T, const N: usize> SubAssign for Vector<T, N>
where
    T: Number,
{
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<T, const N: usize> Mul<T> for Vector<T, N>
where
    T: Number,
{
    type Output = Self;

    fn mul(self, scalar: T) -> Self {
        let mut result = self;
        for i in 0..N {
            result[i] *= scalar;
        }

        result
    }
}

impl<T, const N: usize> MulAssign<T> for Vector<T, N>
where
    T: Number,
{
    fn mul_assign(&mut self, rhs: T) {
        *self = *self * rhs;
    }
}

impl<T: Number, const N: usize> Div<T> for Vector<T, N> {
    type Output = Self;

    fn div(self, rhs: T) -> Self::Output {
        let mut result = self;
        for i in 0..N {
            result[i] /= rhs;
        }

        result
    }
}

impl<T, const N: usize> DivAssign<T> for Vector<T, N>
where
    T: Number,
{
    fn div_assign(&mut self, rhs: T) {
        *self = *self / rhs;
    }
}

impl<T: Number, const N: usize> PartialEq for Vector<T, N> {
    fn eq(&self, other: &Self) -> bool {
        self.data
            .iter()
            .zip(other.data.iter())
            .all(|(&a, &b)| (a - b).abs() <= T::EPSILON)
    }
}

impl<T, const N: usize> Vector<T, N>
where
    T: Number,
{
    pub fn dot(&self, vector: &Vector<T, N>) -> T {
        self.data
            .iter()
            .zip(vector.data.iter())
            .fold(T::ZERO, |sum, (&x, &y)| sum + x * y)
    }

    pub fn norm_1(&self) -> T {
        self.data.iter().fold(T::ZERO, |sum, &x| sum + x.abs())
    }

    pub fn norm(&self) -> T {
        self.data.iter().fold(T::ZERO, |sum, &x| sum + x * x).sqrt()
    }

    pub fn norm_inf(&self) -> T {
        self.data
            .iter()
            .fold(T::ZERO, |sum, &x| T::max(sum, x.abs()))
    }

    pub fn cosine(&self, v: &Vector<T, N>) -> T {
        let dot_product = self.dot(v);
        let u_length = self.norm();
        let v_length = v.norm();

        if u_length == T::ZERO || v_length == T::ZERO {
            return T::ZERO;
        }

        dot_product / (u_length * v_length)
    }

    fn length(&self) -> T {
        let mut squared_sum = T::ZERO;
        for i in 0..N {
            squared_sum += self[i] * self[i];
        }

        squared_sum.sqrt()
    }

    pub fn normalize(&self) -> Vector<T, N> {
        let len = self.length();
        if len == T::ZERO {
            return *self;
        }

        Vector::new(self.data.map(|v| v / len))
    }
}

impl<T> Vector<T, 3>
where
    T: Number,
{
    pub fn cross(&self, v: &Vector<T, 3>) -> Vector<T, 3> {
        Vector {
            data: [
                self[1] * v[2] - self[2] * v[1],
                self[2] * v[0] - self[0] * v[2],
                self[0] * v[1] - self[1] * v[0],
            ],
        }
    }
}

impl<T, const N: usize> Vector<T, N>
where
    T: Number,
{
    pub fn linear_combination(vectors: &[Vector<T, N>], scalars: &[T]) -> Vector<T, N> {
        assert!(!vectors.is_empty(), "Vectors is empty");

        assert_eq!(
            vectors.len(),
            scalars.len(),
            "Vectors length and scalars length must be equal"
        );

        let mut result = Vector::from([T::ZERO; N]);

        for (scalar, vector) in scalars.iter().zip(vectors.iter()) {
            result
                .data
                .iter_mut()
                .zip(vector.data.iter())
                .for_each(|(res, &v)| *res += *scalar * v);
        }

        result
    }
}
