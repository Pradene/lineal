#[derive(Debug, Clone)]
struct Vector<K> {
    data: Vec<K>,
}

impl<K> Vector<K>
where K:
    std::ops::Add<Output = K> +
    std::ops::Sub<Output = K> + 
    std::ops::Mul<Output = K> + 
    Copy +
    Default,
{
    fn new(data: Vec<K>) -> Self {
        Vector {
            data: data
        }
    }

    fn add(&mut self, v: Vector<K>) {
        assert_eq!(self.data.len(), v.data.len(), "Vectors must be of the same length");

        for i in 0..self.data.len() {
            self.data[i] = self.data[i] + v.data[i]
        }
    }

    fn sub(&mut self, v: Vector<K>) {
        assert_eq!(self.data.len(), v.data.len(), "Vectors must be of the same length");

        for i in 0..self.data.len() {
            self.data[i] = self.data[i] - v.data[i]
        }
    }

    fn scl(&mut self, scale: K) {
        for i in 0..self.data.len() {
            self.data[i] = self.data[i] * scale
        }
    }
}

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
    Copy +
    Default +
    std::fmt::Debug,
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

        for i in 0..self.rows {
            for j in 0..self.cols {
                self.data[i][j] = self.data[i][j] + matrix.data[i][j]
            }
        }
    }

    fn sub(&mut self, matrix: &Matrix<K>) {
        assert_eq!(self.rows, matrix.rows, "Matrices rows size differ: {} vs {}", self.rows, matrix.rows);
        assert_eq!(self.cols, matrix.cols, "Matrices cols size differ: {} vs {}", self.cols, matrix.cols);

        for i in 0..self.rows {
            for j in 0..self.cols {
                self.data[i][j] = self.data[i][j] - matrix.data[i][j]
            }
        }
    }

    fn scl(&mut self, scale: K) {
        for i in 0..self.rows {
            for j in 0..self.cols {
                self.data[i][j] = self.data[i][j] * scale
            }
        }
    }

    fn mul_mat(&self, matrix: &Matrix<K>) -> Matrix<K> {
        assert_eq!(self.cols, matrix.rows, "Cannot multiply matrices cols and rows differ: {} vs {}", self.cols, matrix.rows);

        let mut result = vec![vec![K::default(); matrix.cols]; self.rows];

        for i in 0..self.rows {
            for j in 0..matrix.cols {
                for k in 0..self.cols {
                    result[i][j] = result[i][j] + self.data[i][k] * matrix.data[k][j];
                }
            }
        }

        Matrix::new(result)
    }

    fn mul_vec(&mut self, vector: &Vector<K>) -> Vector<K> {
        assert_eq!(self.cols, vector.data.len(), "Cannot multiply matirx by vector: {:?} vs {}", self.data, self.rows);

        let mut result = Vec::with_capacity(self.rows);

        for i in 0..self.rows {
            let mut sum = K::default();
            for j in 0..self.cols {
                sum = sum + self.data[i][j] * vector.data[j];
            }

            result.push(sum);
        }

        Vector::new(result)
    }

    fn transpose(&self) -> Matrix<K> {
        let mut data: Vec<Vec<K>> = Vec::with_capacity(self.cols);

        for i in 0..self.cols {
            let mut new_row: Vec<K> = Vec::with_capacity(self.rows);
            for j in 0..self.rows {
                new_row.push(self.data[j][i]);
            }

            data.push(new_row);
        }

        Matrix::new(data)
    }

    fn print(&self) {
        println!("rows: {}\ncols: {}\nvalues: {:?}\n", self.rows, self.cols, self.data);
    }
}

fn linear_combination<K>(vectors: &[Vector<K>], scalars: &[K]) -> Vector<K>
where K:
    std::ops::Add<Output = K> +
    std::ops::Sub<Output = K> +
    std::ops::Mul<Output = K> +
    Clone +
    Copy +
    Default,
{
    // Check vectors length is not equal to 0
    assert!(!vectors.is_empty(), "Vectors is empty");

    // Check if vectors length and scalars length are equal
    assert_eq!(vectors.len(), scalars.len(), "Vectors length and scalars length must be equal");
    
    // Check all vector of vectors have the same length
    assert!(vectors.iter().all(|v| v.data.len() == vectors[0].data.len()), "All vectors must have the same length");

    let mut result = vec![K::default(); vectors[0].data.len()];

    for (scalar, vector) in scalars.iter().zip(vectors.iter()) {
        result.iter_mut()
            .zip(vector.data.iter())
            .for_each(|(res, &v)| *res = *res + scalar.clone() * v.clone());
    }

    Vector::new(result)
}

trait Lerp {
    fn lerp(start: Self, end: Self, t: f32) -> Self;
}

fn lerp<T>(start: T, end: T, t: f32) -> T
where T:
    Lerp,
{
    T::lerp(start, end, t)
}

impl Lerp for f64 {
    fn lerp(start: Self, end: Self, t: f32) -> Self {
        let t = t.clamp(0., 1.);
        start + (t as f64) * (end - start)
    }
}

impl<K> Lerp for Vector<K>
where K:
    std::ops::Add<Output = K> +
    std::ops::Sub<Output = K> +
    std::ops::Mul<f32, Output = K> +
    Copy +
    Lerp,
{
    fn lerp(start: Self, end: Self, t: f32) -> Self {
        assert_eq!(start.data.len(), end.data.len(), "Vectors must be the same length");
        let result = 
            start.data
            .iter().zip(end.data.iter())
            .map(|(&s, &e)| lerp(s, e, t))
            .collect();

        Self {
            data: result
        }
    }
}

impl<K> Lerp for Matrix<K>
where K:
    std::ops::Add<Output = K> +
    std::ops::Sub<Output = K> +
    std::ops::Mul<f32, Output = K> +
    Copy +
    Lerp,
{
    fn lerp(start: Self, end: Self, t: f32) -> Self {
        assert_eq!(start.rows, end.rows, "Matrices must have the same number of rows");
        assert_eq!(start.cols, end.cols, "Matrices must have the same number of cols");
        let result = 
            start.data
            .iter().zip(end.data.iter())
            .map(|(start_row, end_row)| {
                start_row
                .iter()
                .zip(end_row.iter())
                .map(|(&s, &e)| lerp(s, e, t))
                .collect()
            })
            .collect();

        Self {
            data: result,
            rows: start.rows,
            cols: start.cols
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

    matrix.mul_mat(&other);
    matrix.print();
    matrix.add(&other);
    matrix.print();
    matrix.sub(&other);
    matrix.print();
    matrix.scl(0.5);
    matrix.print();

    let transposed = matrix.transpose();
    transposed.print();

    let v1 = Vector::new(vec![1., 2., 3.]);
    let v2 = Vector::new(vec![0., 10., -100.]);

    println!("{:#?}\n", linear_combination(&[v1, v2], &[10., -2.]));

    println!("{}\n", lerp(0., 10., 0.343));
}
