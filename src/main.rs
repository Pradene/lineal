use lineal::{lerp, Matrix, Vector};

// Exercice 0
// fn main() {
//     let v0 = Vector::new([0.0, 1.0, 2.0]);
//     let v1 = Vector::new([3.0, 4.0, 5.0]);

//     println!("v0 + v1 = {}", v0 + v1);
//     println!("v1 - v0 = {}", v1 - v0);
//     println!("v0 * 2.0 = {}", v0 * 2.0);
//     println!("v0 * 4.0 = {}", v0 * 4.0);

//     let m0 = Matrix::from_col([[1.0, 2.0], [3.0, 4.0]]);

//     let m1 = Matrix::from_col([[2.0, 3.0], [4.0, 5.0]]);

//     println!("m0 + m1 = {}", m0 + m1);
//     println!("m1 - m0 = {}", m1 - m0);
//     println!("m0 * 2.0 = {}", m0 * 2.0);
//     println!("m1 * 2.0 = {}", m1 * 2.0);
// }

// Exercice 1
// fn main() {
//     let v0 = Vector::new([1.0, 0.0, 0.0]);
//     let v1 = Vector::new([0.0, 1.0, 0.0]);
//     let v2 = Vector::new([0.0, 0.0, 1.0]);

//     println!("v0 * 10.0 + v1 * -2.0 + v2 * 0.5 = {}", Vector::linear_combination(&[v0, v1, v2], &[10.0, -2.0, 0.5]));

//     let v3 = Vector::new([1.0, 2.0, 3.0]);
//     let v4 = Vector::new([0.0, 10.0, -100.0]);

//     println!("v3 * 10.0 + v4 * -2.0 = {}", Vector::linear_combination(&[v3, v4], &[10.0, -2.0]));
// }

// Exercice 2
// fn main() {
//     println!("{}", lerp(0.0, 1.0, 0.0));
//     println!("{}", lerp(0.0, 1.0, 1.0));
//     println!("{}", lerp(0.0, 1.0, 0.5));
//     println!("{}", lerp(21.0, 42.0, 0.3));

//     let v0 = Vector::new([2.0, 1.0]);
//     let v1 = Vector::new([4.0, 2.0]);

//     println!("{}", lerp(v0, v1, 0.3));

//     let m0 = Matrix::from_row([
//         [2.0, 1.0],
//         [3.0, 4.0]
//     ]);

//     let m1 = Matrix::from_row([
//         [20.0, 10.0],
//         [30.0, 40.0]
//     ]);

//     println!("{}", lerp(m0, m1, 0.5));
// }

// Exercice 3
// fn main() {
//     let v0 = Vector::new([0., 0.]);
//     let v1 = Vector::new([1., 1.]);
//     println!("{}", v0.dot(&v1));

//     let v0 = Vector::new([1., 1.]);
//     let v1 = Vector::new([1., 1.]);
//     println!("{}", v0.dot(&v1));

//     let v0 = Vector::new([-1., 6.]);
//     let v1 = Vector::new([3., 2.]);
//     println!("{}", v0.dot(&v1));

//     let v0 = Vector::new([0.0, 1.0, 0.0]);
//     let v1 = Vector::new([0.0, -1.0, 0.0]);
//     println!("{}", v0.dot(&v1));

//     let v0 = Vector::new([0.0, 1.0, 0.0]);
//     let v1 = Vector::new([1.0, 0.0, 0.0]);
//     println!("{}", v0.dot(&v1));
// }

// Exercice 4
// fn main() {
//     let v = Vector::new([0., 0., 0.]);
//     println!("{}, {}, {}", v.norm_1(), v.norm(), v.norm_inf());

//     let v = Vector::new([1., 2., 3.]);
//     println!("{}, {}, {}", v.norm_1(), v.norm(), v.norm_inf());

//     let v = Vector::new([-1., -2.]);
//     println!("{}, {}, {}", v.norm_1(), v.norm(), v.norm_inf());
// }

// Exercice 5
// fn main() {
//     let v0 = Vector::new([1.0, 0.0]);
//     let v1 = Vector::new([1.0, 0.0]);
//     println!("{}", v0.cosine(&v1));

//     let v0 = Vector::new([1.0, 0.0]);
//     let v1 = Vector::new([0.0, 1.0]);
//     println!("{}", v0.cosine(&v1));

//     let v0 = Vector::new([-1., 1.]);
//     let v1 = Vector::new([ 1., -1.]);
//     println!("{}", v0.cosine(&v1));

//     let v0 = Vector::new([2., 1.]);
//     let v1 = Vector::new([4., 2.]);
//     println!("{}", v0.cosine(&v1));

//     let v0 = Vector::new([1., 2., 3.]);
//     let v1 = Vector::new([4., 5., 6.]);
//     println!("{}", v0.cosine(&v1));
// }

// Exercice 6
// fn main() {
//     let v0 = Vector::new([0., 0., 1.]);
//     let v1 = Vector::new([1., 0., 0.]);
//     println!("{}", v0.cross(&v1));

//     let v0 = Vector::new([1., 2., 3.]);
//     let v1 = Vector::new([4., 5., 6.]);
//     println!("{}", v0.cross(&v1));

//     let v0 = Vector::new([4., 2., -3.]);
//     let v1 = Vector::new([-2., -5., 16.]);
//     println!("{}", v0.cross(&v1));
// }

//Exercice 7
// fn main() {
//     let m0 = Matrix::from_row([
//         [1., 0.],
//         [0., 1.],
//     ]);
//     let v0 = Vector::new([4., 2.]);
//     println!("{}", m0 * v0);

//     let m0 = Matrix::from_row([
//         [2., 0.],
//         [0., 2.],
//     ]);
//     let v0 = Vector::new([4., 2.]);
//     println!("{}", m0 * v0);

//     let m0 = Matrix::from_row([
//         [2., -2.],
//         [-2., 2.],
//     ]);
//     let v0 = Vector::new([4., 2.]);
//     println!("{}", m0 * v0);

//     let m0 = Matrix::from_row([
//         [1., 0.],
//         [0., 1.],
//     ]);
//     let m1 = Matrix::from_row([
//         [1., 0.],
//         [0., 1.]
//     ]);
//     println!("{}", m0 * m1);

//     let m0 = Matrix::from_row([
//         [1., 0.],
//         [0., 1.],
//     ]);
//     let m1 = Matrix::from_row([
//         [2., 1.],
//         [4., 2.],
//     ]);
//     println!("{}", m0 * m1);

//     let m0 = Matrix::from_row([
//         [3., -5.],
//         [6., 8.],
//     ]);
//     let m1 = Matrix::from_row([
//         [2., 1.],
//         [4., 2.],
//     ]);
//     println!("{}", m0 * m1);
// }

// Exercice 8
// fn main() {
//     let m = Matrix::from_row([
//         [1., 0.],
//         [0., 1.],
//     ]);
//     println!("{}", m.trace());

//     let m = Matrix::from_row([
//         [2., -5., 0.],
//         [4., 3., 7.],
//         [-2., 3., 4.],
//     ]);
//     println!("{}", m.trace());

//     let m = Matrix::from_row([
//         [-2., -8., 4.],
//         [1., -23., 4.],
//         [0., 6., 4.],
//     ]);
//     println!("{}", m.trace());
// }

//Exercice 9
// fn main() {
//     let m0 = Matrix::from_row([
//         [1., 0.],
//         [0., 1.],
//     ]);

//     println!("{}", m0.transpose());

//     let m0 = Matrix::from_row([
//         [-2., -8., 4.],
//         [1., -23., 4.],
//         [0., 6., 4.],
//     ]);
//     println!("{}", m0.transpose());

//     let m0 = Matrix::from_row([
//         [2., -5., 0.],
//         [4., 3., 7.],
//         [-2., 3., 4.],
//     ]);
//     println!("{}", m0.transpose());
// }

// Exercice 10
// fn main() {
//     let u = Matrix::from_row([
//         [1., 0., 0.],
//         [0., 1., 0.],
//         [0., 0., 1.],
//     ]);
//     println!("{}", u.row_echelon());

//     let u = Matrix::from_row([
//         [1., 2.],
//         [3., 4.],
//     ]);
//     println!("{}", u.row_echelon());

//     let u = Matrix::from_row([
//         [1., 2.],
//         [2., 4.],
//     ]);
//     println!("{}", u.row_echelon());

//     let u = Matrix::from_row([
//         [8., 5., -2., 4., 28.],
//         [4., 2.5, 20., 4., -4.],
//         [8., 5., 1., 4., 17.],
//     ]);
//     println!("{}", u.row_echelon());
// }

// // Exercice 11
// fn main() {
//     let m = Matrix::from_row([
//         [ 1., -1.],
//         [-1., 1.],
//     ]);
//     println!("{}", m.determinant());

//     let m = Matrix::from_row([
//         [2., 0., 0.],
//         [0., 2., 0.],
//         [0., 0., 2.],
//     ]);
//     println!("{}", m.determinant());

//     let m = Matrix::from_row([
//         [8., 5., -2.],
//         [4., 7., 20.],
//         [7., 6., 1.],
//     ]);
//     println!("{}", m.determinant());

//     let m = Matrix::from_row([
//         [ 8., 5., -2., 4.],
//         [ 4., 2.5, 20., 4.],
//         [ 8., 5., 1., 4.],
//         [28., -4., 17., 1.],
//     ]);
//     println!("{}", m.determinant());
// }

// Exercice 12
// fn main() {
//     let m = Matrix::from_row([
//         [1., 0., 0.],
//         [0., 1., 0.],
//         [0., 0., 1.],
//     ]);
//     println!("{}", m.inverse().unwrap());

//     let m = Matrix::from_row([
//         [2., 0., 0.],
//         [0., 2., 0.],
//         [0., 0., 2.],
//     ]);
//     println!("{}", m.inverse().unwrap());

//     let m = Matrix::from_row([
//         [8., 5., -2.],
//         [4., 7., 20.],
//         [7., 6., 1.],
//     ]);
//     println!("{}", m.inverse().unwrap());
// }

// Exercice 13
// fn main() {
//     let m = Matrix::from_row([
//         [1., 0., 0.],
//         [0., 1., 0.],
//         [0., 0., 1.],
//     ]);
//     println!("{}", m.rank());

//     let m = Matrix::from_row([
//         [ 1., 2., 0., 0.],
//         [ 2., 4., 0., 0.],
//         [-1., 2., 1., 1.],
//     ]);
//     println!("{}", m.rank());

//     let m = Matrix::from_row([
//         [ 8., 5., -2.],
//         [ 4., 7., 20.],
//         [ 7., 6., 1.],
//         [21., 18., 7.],
//     ]);
//     println!("{}", m.rank());
// }

// Exercice 14
fn main() {
    let fov = (90.0f32).to_radians();
    let ratio = 600.0 / 600.0;
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
