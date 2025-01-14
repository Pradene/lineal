use lineal::{Matrix, Vector};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrix_add() {
        let m1 = Matrix::from([
            [1., 2.],
            [3., 4.],
        ]);

        let m2 = Matrix::from([
            [7., 4.],
            [2., 2.],
        ]);

        let result = Matrix::from([
            [8., 6.],
            [5., 6.],
        ]);

        assert_eq!(result, m1 + m2);
    }

    #[test]
    fn matrix_sub() {
        let m1 = Matrix::from([
            [8., 6.],
            [5., 6.],
        ]);

        let m2 = Matrix::from([
            [7., 4.],
            [2., 2.],
        ]);

        let result = Matrix::from([
            [1., 2.],
            [3., 4.],
        ]);

        assert_eq!(result, m1 - m2);
    }

    #[test]
    fn matrix_scl_by_2() {
        let m1 = Matrix::from([
            [10., 15.],
            [20., 25.],
        ]);

        let result = Matrix::from([
            [20., 30.],
            [40., 50.],
        ]);

        assert_eq!(result, m1 * 2.);
    }

    #[test]
    fn matrix_mul_2x2_by_identity() {
        let m1 = Matrix::from([
            [7., 4.],
            [-2., 2.],
        ]);
        
        let m2 = Matrix::from([
            [1., 0.],
            [0., 1.],
        ]);

        let result = m1.clone();

        assert_eq!(result, m1 * m2);
    }

    #[test]
    fn matrix_mul_2x2_by_2x2_by_2x2() {
        let m1 = Matrix::from([
            [1., 0.],
            [0., 1.],
        ]);
    
        let m2 = Matrix::from([
            [2., 3.],
            [4., 5.],
        ]);
    
        let m3 = Matrix::from([
            [1., 2.],
            [3., 4.],
        ]);
    
        let result = Matrix::from([
            [11., 16.],
            [19., 28.],
        ]);

        assert_eq!(result, m1 * m2 * m3);
    }

    #[test]
    fn matrix_mul_3x2_by_2x3() {
        let m1 = Matrix::from([
            [1., 0.],
            [0., 1.],
            [0., 1.],
        ]);
    
        let m2 = Matrix::from([
            [2., 3., 5.],
            [4., 5., 4.],
        ]);

        let result = Matrix::from([
            [2., 3., 5.],
            [4., 5., 4.],
            [4., 5., 4.],
        ]);

        assert_eq!(result, m1 * m2);
    }

    #[test]
    fn matrix_mul_by_vector() {
        let m = Matrix::from([
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
        ]);

        let v = Vector::from([9., 8., 7.]);

        let result = v.clone();

        assert_eq!(result, m * v);
    }

    #[test]
    fn matrix_transpose() {
        let m = Matrix::from([
            [0., 0., 0., 0.],
            [1., 1., 1., 1.],
            [0., 0., 0., 0.],
        ]);

        let result = Matrix::from([
            [0., 1., 0.],
            [0., 1., 0.],
            [0., 1., 0.],
            [0., 1., 0.],
        ]);

        assert_eq!(result, m.transpose());
    }
}