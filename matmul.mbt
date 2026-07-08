// matmul.mbt

///| Matrix multiplication of two 2D Tensors.
pub fn Tensor::matmul(self : Tensor, other : Tensor) -> Tensor {
  let len1 = self.shape.length()
  let len2 = other.shape.length()
  if len1 != 2 || len2 != 2 {
    panic()
  }
  let m = self.shape[0]
  let k1 = self.shape[1]
  let k2 = other.shape[0]
  let n = other.shape[1]
  if k1 != k2 {
    panic()
  }
  
  let data = Array::make(m * n, 0.0)
  let out_shape = [m, n]
  
  let mut r = 0
  for ; r < m; r = r + 1 {
    let mut c = 0
    for ; c < n; c = c + 1 {
      let mut sum = 0.0
      let mut k = 0
      for ; k < k1; k = k + 1 {
        let val1 = self.data[r * self.strides[0] + k * self.strides[1]]
        let val2 = other.data[k * other.strides[0] + c * other.strides[1]]
        sum = sum + val1 * val2
      }
      data[r * n + c] = sum
    }
  }
  
  let req_grad = self.requires_grad || other.requires_grad
  let out = Tensor::new(data, out_shape, requires_grad: req_grad)
  if req_grad {
    out.creator = Some(MatMul(self, other))
  }
  out
}
