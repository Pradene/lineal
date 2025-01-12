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
}
