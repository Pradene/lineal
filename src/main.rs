use matrix::vector::Vector;
use matrix::matrix::Matrix;

fn main() {
    let v = Vector::from([1, 2, 3]);
    println!("{:?}", v);
    println!("{}", v[0]);
    
    let m = Matrix::from([
        [7., 4.],
        [-2., 2.],
    ]);
    println!("{:?}", m);

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

    let expected = Matrix::from([
        [11., 16.],
        [19., 28.],
    ]);

    println!("{:#?}", m1 * m2 * m3);
    println!("{:#?}", expected);
}
