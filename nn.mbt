// nn.mbt

///| Linear (fully connected) neural network layer.
pub struct Linear {
  pub weight : Tensor
  pub bias : Tensor?
}

///| Create a new Linear layer with random weights (Xavier initialized) and zero bias.
pub fn Linear::new(in_features : Int, out_features : Int, requires_grad~ : Bool = true) -> Linear {
  // Xavier initialization: limit = sqrt(6 / (in_features + out_features))
  let limit = (6.0 / (in_features + out_features).to_double()).sqrt()
  let weight_data = Array::make(in_features * out_features, 0.0)
  
  // Use a predictable seed for weight init
  let rng = Random::new(42 + in_features * 7 + out_features * 13)
  let mut i = 0
  for ; i < in_features * out_features; i = i + 1 {
    weight_data[i] = (rng.next_double() * 2.0 - 1.0) * limit
  }
  let weight = Tensor::new(weight_data, [in_features, out_features], requires_grad~)
  
  let bias_data = Array::make(out_features, 0.0)
  let bias = Tensor::new(bias_data, [1, out_features], requires_grad~)
  
  { weight, bias: Some(bias) }
}

///| Forward pass of the Linear layer: y = x @ weight + bias.
pub fn Linear::forward(self : Linear, x : Tensor) -> Tensor {
  let out = x.matmul(self.weight)
  match self.bias {
    Some(b) => out + b
    None => out
  }
}

///| Get parameters (weight, bias) of the Linear layer.
pub fn Linear::parameters(self : Linear) -> Array[Tensor] {
  match self.bias {
    Some(b) => [self.weight, b]
    None => [self.weight]
  }
}

///| Mean Squared Error (MSE) loss between prediction and target.
pub fn mse_loss(pred : Tensor, target : Tensor) -> Tensor {
  if pred.shape != target.shape {
    panic()
  }
  let size = pred.data.length()
  let mut sum_val = 0.0
  let mut i = 0
  for ; i < size; i = i + 1 {
    let diff = pred.data[i] - target.data[i]
    sum_val = sum_val + diff * diff
  }
  let loss_val = sum_val / size.to_double()
  
  let req_grad = pred.requires_grad || target.requires_grad
  let out = Tensor::new([loss_val], [], requires_grad: req_grad)
  if req_grad {
    out.creator = Some(MSELoss(pred, target))
  }
  out
}
