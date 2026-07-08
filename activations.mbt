// activations.mbt

///| Exponential function approximation using Taylor series.
pub fn exp(x : Double) -> Double {
  if x == 0.0 {
    return 1.0
  }
  let x_int = x.to_int()
  let t = x - x_int.to_double()
  
  let exp_t = 1.0 + t + (t * t) / 2.0 + (t * t * t) / 6.0 + (t * t * t * t) / 24.0 + (t * t * t * t * t) / 120.0 + (t * t * t * t * t * t) / 720.0 + (t * t * t * t * t * t * t) / 5040.0
  
  let e = 2.718281828459045
  let mut exp_int = 1.0
  if x_int > 0 {
    let mut i = 0
    for ; i < x_int; i = i + 1 {
      exp_int = exp_int * e
    }
  } else if x_int < 0 {
    let mut i = 0
    let limit = -x_int
    for ; i < limit; i = i + 1 {
      exp_int = exp_int / e
    }
  }
  exp_int * exp_t
}

///| Natural logarithm approximation using Newton-Raphson method.
pub fn ln(x : Double) -> Double {
  if x <= 0.0 {
    return -1e9
  }
  let mut y = 0.0
  if x > 1.0 {
    let mut temp = x
    for ; temp > 2.0; temp = temp / 2.718281828459045 {
      y = y + 1.0
    }
  } else if x < 1.0 {
    let mut temp = x
    for ; temp < 0.5; temp = temp * 2.718281828459045 {
      y = y - 1.0
    }
  }
  let mut i = 0
  for ; i < 5; i = i + 1 {
    let ey = exp(y)
    y = y - 1.0 + x / ey
  }
  y
}

///| Rectified Linear Unit (ReLU) activation.
pub fn Tensor::relu(self : Tensor) -> Tensor {
  let size = self.data.length()
  let data = Array::make(size, 0.0)
  let mut i = 0
  for ; i < size; i = i + 1 {
    let v = self.data[i]
    data[i] = if v > 0.0 { v } else { 0.0 }
  }
  let req_grad = self.requires_grad
  let out = Tensor::new(data, self.shape, requires_grad: req_grad)
  if req_grad {
    out.creator = Some(ReLU(self))
  }
  out
}

///| Sigmoid activation.
pub fn Tensor::sigmoid(self : Tensor) -> Tensor {
  let size = self.data.length()
  let data = Array::make(size, 0.0)
  let mut i = 0
  for ; i < size; i = i + 1 {
    let x = self.data[i]
    data[i] = 1.0 / (1.0 + exp(-x))
  }
  let req_grad = self.requires_grad
  let out = Tensor::new(data, self.shape, requires_grad: req_grad)
  if req_grad {
    out.creator = Some(Sigmoid(self))
  }
  out
}

///| Hyperbolic Tangent (Tanh) activation.
pub fn Tensor::tanh(self : Tensor) -> Tensor {
  let size = self.data.length()
  let data = Array::make(size, 0.0)
  let mut i = 0
  for ; i < size; i = i + 1 {
    let x = self.data[i]
    let ex = exp(x)
    let enx = exp(-x)
    data[i] = (ex - enx) / (ex + enx)
  }
  let req_grad = self.requires_grad
  let out = Tensor::new(data, self.shape, requires_grad: req_grad)
  if req_grad {
    out.creator = Some(Tanh(self))
  }
  out
}
