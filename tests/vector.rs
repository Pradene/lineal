use lineal::{Vector, lerp};

#[cfg(test)]
mod tests {
    use super::*;

    fn epsilon(x: f32, y: f32) -> bool{
        let eps = 0.0000001;
        (x - y).abs() < eps
    }

    #[test]
    fn vector_add() {
        let v1 = Vector::from([1., 1., 1.]);
        let v2 = Vector::from([2., 2., 2.]);
        
        let result = Vector::from([3., 3., 3.]);
    
        assert_eq!(result, v1 + v2);
    }

    #[test]
    fn vector_sub() {
        let v1 = Vector::from([2., 2., 2.]);
        let v2 = Vector::from([1., 1., 1.]);
        
        let result = v2.clone();
    
        assert_eq!(result, v1 - v2);
    }

    #[test]
    fn vector_scl() {
        let v = Vector::from([10., 5., 1.]);

        let result = Vector::from([20., 10., 2.]);

        assert_eq!(result, v * 2.);
    }

    #[test]
    fn vector_dot() {
        let v1 = Vector::from([0., 0.]);
        let v2 = Vector::from([1., 1.]);

        assert_eq!(0., v1.dot(&v2));
    }

    #[test]
    fn vector_norm_1() {
        let v = Vector::from([1., 5., 10.]);

        assert_eq!(16., v.norm_1());
    }

    #[test]
    fn vector_norm() {
        let v = Vector::from([1., 5., 3., 1.]);

        assert_eq!(6., v.norm());
    }

    #[test]
    fn vector_norm_inf() {
        let v = Vector::from([5., 8., -10.]);

        assert_eq!(10., v.norm_inf());
    }

    #[test]
    fn vector_cross() {
        let v1 = Vector::from([1., 2., 3.]);
        let v2 = Vector::from([4., 5., 6.]);

        let result = Vector::from([-3., 6., -3.]);

        assert_eq!(result, v1.cross(&v2));
    }

    #[test]
    fn vector_cosine() {
        let v1 = Vector::from([2., 1.]);
        let v2 = Vector::from([4., 2.]);

        assert!(epsilon(1., v1.cosine(&v2)));
    }

    #[test]
    fn vector_lerp() {
        let v1 = Vector::from([2., 1.]);
        let v2 = Vector::from([4., 2.]);

        assert_eq!(v2, lerp(v1, v2, 1.));
    }
}