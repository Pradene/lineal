use lineal::Matrix;

fn main() {
    let m = Matrix::from_row([[2., 0., 0.], [0., 2., 0.], [0., 0., 2.]]);

    println!("{}", m);
    println!("{}", m.inverse().unwrap().inverse().unwrap());
}
