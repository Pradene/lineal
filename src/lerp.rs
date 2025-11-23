use std::ops::{Add, Mul};

pub fn lerp<T: Add<Output = T> + Mul<f32, Output = T>>(x: T, y: T, t: f32) -> T {
    let t = t.clamp(0., 1.);
    x * (1. - t) + y * t
}
