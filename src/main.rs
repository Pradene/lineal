use lineal::Vector;
use lineal::Matrix;
use num::complex::Complex;

fn main() {
    println!("Hello World!");
    
    let v1 = Vector::from([
        Complex::new(3., 4.),
        Complex::new(3., 2.)
    ]);

    let v2 = Vector::from([
        Complex::new(3., 4.),
        Complex::new(3., 2.)
    ]);

    println!("{}", v1);
    println!("{}", v1 + v2);
    println!("{}", v1 * Complex::new(1., 1.));

    let m = Matrix::from([
        [1., 2.],
        [3., 4.],
    ]);
    println!("{}", m);
}
