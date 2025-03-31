use lineal::Matrix;

fn main() {
    let fov = 90.0;
    let ratio = 800.0 / 600.0;
    let near = 0.1;
    let far = 100.0;

    let proj = Matrix::projection(fov, ratio, near, far);
    
    for col in proj.data.iter() {
        for value in col {
            print!("{value}, ");
        }
        println!();
    }
}
