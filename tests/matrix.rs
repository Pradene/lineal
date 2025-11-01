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
        let m = Matrix::from_row([[8., 5., -2.], [4., 7., 20.], [7., 6., 1.]]);

        let i = Matrix::from_row([
            [0.649425287, 0.097701149, -0.655172414],
            [-0.781609195, -0.126436782, 0.965517241],
            [0.143678161, 0.074712644, -0.206896552],
        ]);

        assert_eq!(i, m.inverse().unwrap());
    }

    #[test]
    fn test_matrix_rank() {
        let m = Matrix::from_row([[8., 5., -2.], [4., 7., 20.], [7., 6., 1.], [21., 18., 7.]]);

        assert_eq!(3, m.rank())
    }
}
