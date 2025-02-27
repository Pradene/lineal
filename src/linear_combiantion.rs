use num::Float;

use crate::Vector;

pub fn linear_combination<T, const N: usize>(
    vectors: &[Vector<T, N>],
    scalars: &[T],
) -> Vector<T, N>
where
    T: Float,
{
    // Check vectors length is not equal to 0
    assert!(!vectors.is_empty(), "Vectors is empty");

    // Check if vectors length and scalars length are equal
    assert_eq!(
        vectors.len(),
        scalars.len(),
        "Vectors length and scalars length must be equal"
    );

    let mut result = Vector::from([T::zero(); N]);

    for (scalar, vector) in scalars.iter().zip(vectors.iter()) {
        result
            .data
            .iter_mut()
            .zip(vector.data.iter())
            .for_each(|(res, &v)| *res = *res + scalar.clone() * v.clone());
    }

    return result;
}
