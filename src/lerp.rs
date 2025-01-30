use std::ops::{Add, Mul, Sub};

pub fn lerp<T>(x: T, y: T, t: f32) -> T
where
    T: Copy + Sub<Output = T> + Add<Output = T> + Mul<f32, Output = T>,
{
    let t = t.clamp(0., 1.);
    x * (1. - t) + y * t
}
