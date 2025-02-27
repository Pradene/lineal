use lineal::{lerp, Matrix, Vector};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_add() {
        let m1 = Matrix::from_row([[1., 2.], [3., 4.]]);

        let m2 = Matrix::from_row([[7., 4.], [2., 2.]]);

        let result = Matrix::from_row([[8., 6.], [5., 6.]]);

        assert_eq!(result, m1 + m2);
    }

    #[test]
    fn test_matrix_sub() {
        let m1 = Matrix::from_row([[8., 6.], [5., 6.]]);

        let m2 = Matrix::from_row([[7., 4.], [2., 2.]]);

        let result = Matrix::from_row([[1., 2.], [3., 4.]]);

        assert_eq!(result, m1 - m2);
    }

    #[test]
    fn test_matrix_scl_by_2() {
        let m1 = Matrix::from_row([[10., 15.], [20., 25.]]);

        let result = Matrix::from_row([[20., 30.], [40., 50.]]);

        assert_eq!(result, m1 * 2.);
    }

    #[test]
    fn test_matrix_mul_2x2_by_identity() {
        let m1 = Matrix::from_row([[7., 4.], [-2., 2.]]);

        let m2 = Matrix::from_row([[1., 0.], [0., 1.]]);

        let result = m1.clone();

        assert_eq!(result, m1 * m2);
    }

    #[test]
    fn test_matrix_mul_2x2_by_2x2_by_2x2() {
        let m1 = Matrix::from_row([[1., 0.], [0., 1.]]);

        let m2 = Matrix::from_row([[2., 3.], [4., 5.]]);

        let m3 = Matrix::from_row([[1., 2.], [3., 4.]]);

        let result = Matrix::from_row([[11., 16.], [19., 28.]]);

        assert_eq!(result, m1 * m2 * m3);
    }

    #[test]
    fn test_matrix_mul_3x2_by_2x3() {
        let m1 = Matrix::from_row([[1., 0.], [0., 1.], [0., 1.]]);

        let m2 = Matrix::from_row([[2., 3., 5.], [4., 5., 4.]]);

        let result = Matrix::from_row([[2., 3., 5.], [4., 5., 4.], [4., 5., 4.]]);

        assert_eq!(result, m1 * m2);
    }

    #[test]
    fn test_matrix_mul_by_vector() {
        let m = Matrix::from_row([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]);

        let v = Vector::new([9., 8., 7.]);

        let result = v.clone();

        assert_eq!(result, m * v);
    }

    #[test]
    fn test_matrix_transpose() {
        let m = Matrix::from_row([[0., 0., 0., 0.], [1., 1., 1., 1.], [0., 0., 0., 0.]]);

        let result = Matrix::from_row([[0., 1., 0.], [0., 1., 0.], [0., 1., 0.], [0., 1., 0.]]);

        assert_eq!(result, m.transpose());
    }

    #[test]
    fn test_matrix_trace() {
        let m = Matrix::from_row([[1., 4., 5.], [1., 4., 5.], [1., 4., 5.]]);

        assert_eq!(10., m.trace());
    }

    #[test]
    fn test_matrix_lerp() {
        let m1 = Matrix::from_row([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]);

        let m2 = Matrix::from_row([[4., 2., 8.], [2., 6., 0.], [2., 2., 8.]]);

        let result = Matrix::from_row([[2., 1., 4.], [1., 3., 0.], [1., 1., 4.]]);

        assert_eq!(result, lerp(m1, m2, 0.5));
    }

    #[test]
    fn test_matrix_row_echelon_form() {
        let m = Matrix::from_row([[1., 2.], [2., 4.]]);

        let result = Matrix::from_row([[1., 2.], [0., 0.]]);

        assert_eq!(result, m.row_echelon());
    }

    #[test]
    fn test_matrix_determinant() {
        let m = Matrix::from_row([
            [8., 5., -2., 4.],
            [4., 2.5, 20., 4.],
            [8., 5., 1., 4.],
            [28., -4., 17., 1.],
        ]);

        assert_eq!(1032., m.determinant())
    }

    #[test]
    fn test_matrix_inverse_0() {
        let m = Matrix::from_row([[2., 0., 0.], [0., 2., 0.], [0., 0., 2.]]);

        let result = Matrix::from_row([[0.5, 0., 0.], [0., 0.5, 0.], [0., 0., 0.5]]);

        assert_eq!(result, m.inverse().unwrap());
        assert_eq!(m, m.inverse().unwrap().inverse().unwrap());
    }

    #[test]
    fn test_matrix_inverse_1() {
        let m = Matrix::from_col([
            [8., 7., -6., -3.],
            [-5., 5., 0., 0.],
            [9., 6., 9., -9.],
            [2., 1., 6., -4.],
        ]);

        let i = Matrix::from_col([
            [-0.15385, -0.07692, 0.35897, -0.69231],
            [-0.15385, 0.12308, 0.35897, -0.69231],
            [-0.28205, 0.02564, 0.43590, -0.76923],
            [-0.53846, 0.03077, 0.92308, -1.92308],
        ]);

        assert_eq!(i, m.inverse().unwrap());
    }

    #[test]
    fn test_matrix_inverse_2() {
        let m = Matrix::from_col([
            [9., -5., -4., -7.],
            [3., -2., 9., 6.],
            [0., -6., 6., 6.],
            [9., -3., 4., 2.],
        ]);

        let i = Matrix::from_col([
            [-0.04074, -0.07778, -0.02901, 0.17778],
            [-0.07778, 0.03333, -0.14630, 0.06667],
            [0.14444, 0.36667, -0.10926, -0.26667],
            [-0.22222, -0.33333, 0.12963, 0.33333],
        ]);

        assert_eq!(i, m.inverse().unwrap());
    }

    #[test]
    fn test_matrix_rank() {
        let m = Matrix::from_row([[8., 5., -2.], [4., 7., 20.], [7., 6., 1.], [21., 18., 7.]]);

        assert_eq!(3, m.rank())
    }
}
