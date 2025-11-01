use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

pub trait Number:
    Copy
    + PartialEq
    + PartialOrd
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
    + Sized
{
    const ZERO: Self;
    const ONE: Self;
    const EPSILON: Self;
    const PI: Self;

    fn abs(self) -> Self;
    fn sqrt(self) -> Self;
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn tan(self) -> Self;
    fn powi(self, n: i32) -> Self;
    fn powf(self, n: f32) -> Self;
    fn max(self, other: Self) -> Self;
    fn min(self, other: Self) -> Self;
    fn clamp(self, min: Self, max: Self) -> Self;
}

impl Number for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const EPSILON: Self = f32::EPSILON;
    const PI: Self = std::f32::consts::PI;

    #[inline]
    fn abs(self) -> Self {
        f32::abs(self)
    }

    #[inline]
    fn sqrt(self) -> Self {
        f32::sqrt(self)
    }

    #[inline]
    fn sin(self) -> Self {
        f32::sin(self)
    }

    #[inline]
    fn cos(self) -> Self {
        f32::cos(self)
    }

    #[inline]
    fn tan(self) -> Self {
        f32::tan(self)
    }

    #[inline]
    fn powi(self, n: i32) -> Self {
        f32::powi(self, n)
    }

    #[inline]
    fn powf(self, n: f32) -> Self {
        f32::powf(self, n)
    }

    #[inline]
    fn max(self, other: Self) -> Self {
        f32::max(self, other)
    }

    #[inline]
    fn min(self, other: Self) -> Self {
        f32::min(self, other)
    }

    #[inline]
    fn clamp(self, min: Self, max: Self) -> Self {
        if self < min {
            min
        } else if self > max {
            max
        } else {
            self
        }
    }
}

impl Number for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const EPSILON: Self = f64::EPSILON;
    const PI: Self = std::f64::consts::PI;

    #[inline]
    fn abs(self) -> Self {
        f64::abs(self)
    }

    #[inline]
    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }

    #[inline]
    fn sin(self) -> Self {
        f64::sin(self)
    }

    #[inline]
    fn cos(self) -> Self {
        f64::cos(self)
    }

    #[inline]
    fn tan(self) -> Self {
        f64::tan(self)
    }

    #[inline]
    fn powi(self, n: i32) -> Self {
        f64::powi(self, n)
    }

    #[inline]
    fn powf(self, n: f32) -> Self {
        f64::powf(self, n as f64)
    }

    #[inline]
    fn max(self, other: Self) -> Self {
        f64::max(self, other)
    }

    #[inline]
    fn min(self, other: Self) -> Self {
        f64::min(self, other)
    }

    #[inline]
    fn clamp(self, min: Self, max: Self) -> Self {
        if self < min {
            min
        } else if self > max {
            max
        } else {
            self
        }
    }
}
