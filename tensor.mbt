// tensor.mbt

// Global tensor ID counter to identify tensors in the computation graph
let mut next_tensor_id : Int = 0

fn get_next_tensor_id() -> Int {
  let id = next_tensor_id
  next_tensor_id = next_tensor_id + 1
  id
}

///| Multi-dimensional array representing a Tensor.
pub struct Tensor {
  pub id : Int
  pub shape : Array[Int]
  pub strides : Array[Int]
  pub data : Array[Double]
  pub mut grad : Array[Double]?
  pub mut requires_grad : Bool
  pub mut creator : Op?
}

///| Helper to calculate strides for a given shape.
pub fn shape_to_strides(shape : Array[Int]) -> Array[Int] {
  let len = shape.length()
  if len == 0 {
    return []
  }
  let strides = Array::make(len, 1)
  let mut s = 1
  let mut i = len - 1
  for ; i >= 0; i = i - 1 {
    strides[i] = s
    s = s * shape[i]
  }
  strides
}

///| Create a new Tensor.
pub fn Tensor::new(
  data : Array[Double],
  shape : Array[Int],
  requires_grad~ : Bool = false
) -> Tensor {
  let strides = shape_to_strides(shape)
  let id = get_next_tensor_id()
  {
    id,
    shape,
    strides,
    data,
    grad: None,
    requires_grad,
    creator: None,
  }
}

///| Create a Tensor of zeros with the specified shape.
pub fn zeros(shape : Array[Int], requires_grad~ : Bool = false) -> Tensor {
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

///| Create a Tensor of ones with the specified shape.
pub fn ones(shape : Array[Int], requires_grad~ : Bool = false) -> Tensor {
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

///| Simple LCG pseudo-random number generator for reproducible initialization.
struct Random {
  mut state : Int64
}

fn Random::new(seed : Int) -> Random {
  { state: seed.to_int64() }
}

fn next_double(self : Random) -> Double {
  // LCG parameters (Numerical Recipes)
  let a : Int64 = 1664525
  let c : Int64 = 1013904223
  let m : Int64 = 4294967296 // 2^32
  self.state = (a * self.state + c) % m
  let d = self.state.to_double() / m.to_double()
  d
}

// Approximate normal distribution using Central Limit Theorem
fn next_randn(self : Random) -> Double {
  let mut sum = 0.0
  let mut i = 0
  for ; i < 12; i = i + 1 {
    sum = sum + self.next_double()
  }
  sum - 6.0
}

///| Create a Tensor of random values from a normal distribution.
pub fn randn(
  shape : Array[Int],
  seed~ : Int = 42,
  requires_grad~ : Bool = false
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
  let mut i = 0
  for ; i < size; i = i + 1 {
    data[i] = rng.next_randn()
  }
  Tensor::new(data, shape, requires_grad~)
}

///| Create a 1D or 2D Tensor from a flat array.
pub fn from_array(
  data : Array[Double],
  shape : Array[Int],
  requires_grad~ : Bool = false
) -> Tensor {
  Tensor::new(data, shape, requires_grad~)
}
