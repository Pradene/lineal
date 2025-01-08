#[derive(Debug, Clone)]
struct Matrix<K> {
    data: Vec<Vec<K>>,
    rows: usize,
    cols: usize,
}

impl<K> Matrix<K>
where K: 
    std::ops::Add<Output = K> +
    std::ops::Sub<Output = K> + 
    std::ops::Mul<Output = K> + 
    Copy,
{
    fn new(data: Vec<Vec<K>>) -> Self {
        
        let rows = data.len();
        let cols = if rows > 0 { data[0].len() } else { 0 };
        
        Matrix {
            data: data,
            rows: rows,
            cols: cols
        }
    }

    fn add(&mut self, matrix: &Matrix<K>) {
        assert_eq!(self.rows, matrix.rows, "Matrices rows size differ: {} vs {}", self.rows, matrix.rows);
        assert_eq!(self.cols, matrix.cols, "Matrices cols size differ: {} vs {}", self.cols, matrix.cols);

        for row in 0..self.rows {
            for col in 0..self.cols {
                self.data[row][col] = self.data[row][col] + matrix.data[row][col]
            }
        }
    }

    fn substract(&mut self, matrix: &Matrix<K>) {
        assert_eq!(self.rows, matrix.rows, "Matrices rows size differ: {} vs {}", self.rows, matrix.rows);
        assert_eq!(self.cols, matrix.cols, "Matrices cols size differ: {} vs {}", self.cols, matrix.cols);

        for row in 0..self.rows {
            for col in 0..self.cols {
                self.data[row][col] = self.data[row][col] - matrix.data[row][col]
            }
        }
    }

    fn scale(&mut self, scale: K) {
        for row in 0..self.rows {
            for col in 0..self.cols {
                self.data[row][col] = self.data[row][col] * scale
            }
        }
    }

    fn transpose(&mut self) -> Matrix<K> {
        let mut data: Vec<Vec<K>> = Vec::with_capacity(self.cols);

        for col in 0..self.cols {
            let mut new_row: Vec<K> = Vec::with_capacity(self.rows);
            for row in 0..self.rows {
                new_row.push(self.data[row][col]);
            }

            data.push(new_row);
        }

        Matrix {
            data: data,
            rows: self.rows,
            cols: self.cols
        }
    }
}


fn main() {
    let mut matrix = Matrix::new(vec![
        vec![1., 2., 3.],
        vec![4., 5., 6.],
        vec![7., 8., 9.],
    ]);

    let other = matrix.clone();

    matrix.add(&other);
    println!("{:?}", matrix.data);
    matrix.substract(&other);
    println!("{:?}", matrix.data);
    matrix.scale(0.5);
    println!("{:?}", matrix.data);

    let transposed = matrix.transpose();
    println!("{:?}", transposed.data);
}
