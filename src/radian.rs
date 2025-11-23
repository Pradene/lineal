use std::{
    f32::consts::PI,
    ops::{Add, Mul},
};

pub fn radian<T: Add<Output = T> + Mul<f32, Output = T>>(angle: T) -> T {
    angle * (PI / 180.0)
}
