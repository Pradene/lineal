use matrix::matrix::Matrix;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn
    matrix_add() {
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
    fn matrix_scale() {
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
    fn matrix_mul_identity() {
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
    fn matrix_mul() {
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
}