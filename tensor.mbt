// tensor.mbt

// Global tensor ID counter to identify tensors in the computation graph

///|
let next_tensor_id : Ref[Int] = { val: 0 }

///|
fn get_next_tensor_id() -> Int {
  let id = next_tensor_id.val
  next_tensor_id.val = id + 1
  id
}

///|
/// Multi-dimensional array representing a Tensor.
pub struct Tensor {
  id : Int
  shape : Array[Int]
  strides : Array[Int]
  data : Array[Double]
  mut grad : Array[Double]?
  requires_grad : Bool
  mut creator : Op?
}

///|
/// Helper to calculate strides for a given shape.
pub fn shape_to_strides(shape : Array[Int]) -> Array[Int] {
  let len = shape.length()
  if len == 0 {
    return []
  }
  let strides = Array::make(len, 1)
  let mut s = 1
  for i in 0..<len {
    let idx = len - 1 - i
    strides[idx] = s
    s = s * shape[idx]
  }
  strides
}

///|
/// Create a new Tensor.
pub fn Tensor::new(
  data : Array[Double],
  shape : Array[Int],
  requires_grad? : Bool = false,
) -> Tensor {
  let strides = shape_to_strides(shape)
  let id = get_next_tensor_id()
  { id, shape, strides, data, grad: None, requires_grad, creator: None }
}

///|
/// Create a Tensor of zeros with the specified shape.
pub fn zeros(shape : Array[Int], requires_grad? : Bool = false) -> Tensor {
  let mut size = 1
  for dim in shape {
    size = size * dim
  }
  if shape.length() == 0 {
    size = 1
  }
  let data = Array::make(size, 0.0)
  Tensor::new(data, shape, requires_grad~)
}

///|
/// Create a Tensor of ones with the specified shape.
pub fn ones(shape : Array[Int], requires_grad? : Bool = false) -> Tensor {
  let mut size = 1
  for dim in shape {
    size = size * dim
  }
  if shape.length() == 0 {
    size = 1
  }
  let data = Array::make(size, 1.0)
  Tensor::new(data, shape, requires_grad~)
}

///|
/// Simple LCG pseudo-random number generator for reproducible initialization.
priv struct Random {
  mut state : Int64
}

///|
fn Random::new(seed : Int) -> Random {
  { state: seed.to_int64() }
}

///|
fn Random::next_double(self : Random) -> Double {
  // LCG parameters (Numerical Recipes)
  let a : Int64 = 1664525
  let c : Int64 = 1013904223
  let m : Int64 = 4294967296
  self.state = (a * self.state + c) % m
  let d = self.state.to_double() / m.to_double()
  d
}

// Approximate normal distribution using Central Limit Theorem

///|
fn Random::next_randn(self : Random) -> Double {
  let mut sum = 0.0
  for _ in 0..<12 {
    sum = sum + self.next_double()
  }
  sum - 6.0
}

///|
/// Create a Tensor of random values from a normal distribution.
pub fn randn(
  shape : Array[Int],
  seed? : Int = 42,
  requires_grad? : Bool = false,
) -> Tensor {
  let mut size = 1
  for dim in shape {
    size = size * dim
  }
  if shape.length() == 0 {
    size = 1
  }
  let rng = Random::new(seed)
  let data = Array::make(size, 0.0)
  for i in 0..<size {
    data[i] = rng.next_randn()
  }
  Tensor::new(data, shape, requires_grad~)
}

///|
/// Create a 1D or 2D Tensor from a flat array.
pub fn from_array(
  data : Array[Double],
  shape : Array[Int],
  requires_grad? : Bool = false,
) -> Tensor {
  Tensor::new(data, shape, requires_grad~)
}

///|
/// Get element at specified indices.
pub fn Tensor::op_get(self : Tensor, indices : Array[Int]) -> Double {
  let mut flat_idx = 0
  let len = indices.length()
  for i in 0..<len {
    flat_idx = flat_idx + indices[i] * self.strides[i]
  }
  self.data[flat_idx]
}

///|
/// Set element at specified indices.
pub fn Tensor::op_set(
  self : Tensor,
  indices : Array[Int],
  val : Double,
) -> Unit {
  let mut flat_idx = 0
  let len = indices.length()
  for i in 0..<len {
    flat_idx = flat_idx + indices[i] * self.strides[i]
  }
  self.data[flat_idx] = val
}

///|
fn format_tensor_data(
  data : Array[Double],
  shape : Array[Int],
  strides : Array[Int],
  dim : Int,
  offset : Int,
) -> String {
  if dim == shape.length() {
    return data[offset].to_string()
  }
  if dim == shape.length() - 1 {
    let mut s = "["
    let size = shape[dim]
    let stride = strides[dim]
    for i in 0..<size {
      if i > 0 {
        s = s + ", "
      }
      s = s + data[offset + i * stride].to_string()
    }
    s = s + "]"
    return s
  }
  let mut s = "["
  let size = shape[dim]
  let stride = strides[dim]
  for i in 0..<size {
    if i > 0 {
      s = s + ",\n"
      let indent_limit = dim + 1
      for _ in 0..<indent_limit {
        s = s + " "
      }
    }
    s = s +
      format_tensor_data(data, shape, strides, dim + 1, offset + i * stride)
  }
  s = s + "]"
  s
}

///|
/// Format the Tensor as a string.
pub fn Tensor::to_string(self : Tensor) -> String {
  let shape_len = self.shape.length()
  let data_str = if shape_len == 0 {
    self.data[0].to_string()
  } else {
    format_tensor_data(self.data, self.shape, self.strides, 0, 0)
  }
  let grad_info = if self.requires_grad { ", requires_grad=true" } else { "" }
  "tensor(" + data_str + grad_info + ")"
}

///|
/// Implementation of Show trait for Tensor.
pub fn Tensor::output(self : Tensor, logger : &Logger) -> Unit {
  logger.write_string(self.to_string())
}
