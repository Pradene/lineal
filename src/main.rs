use lineal::Matrix;

fn main() {
    println!("Hello World!");

    let m1 = Matrix::from_row([[1., 0.], [0., 1.], [0., 1.]]);
    let m2 = Matrix::from_row([[2., 3., 5.], [4., 5., 4.]]);

    println!("{}", m1);
    println!("{}", m2);
}
