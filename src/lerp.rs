use std::ops::{Add, Mul};

pub fn lerp<T>(x: T, y: T, t: f32) -> T
where
    T: Add<Output = T> + Mul<f32, Output = T>,
{
    let t = t.clamp(0., 1.);
    x * (1. - t) + y * t
}
