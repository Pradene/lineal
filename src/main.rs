use matrix::vector::Vector;
use matrix::matrix::Matrix;

fn main() {
    println!("Hello World!");

    
    let v1 = Vector::from([-1., 1.]);
    let v2 = Vector::from([ 1., -1.]);

    println!("{}", v1.cosine(&v2));
    println!("{}", v2.cosine(&v1));
}
