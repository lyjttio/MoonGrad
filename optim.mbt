// optim.mbt

///|
/// Stochastic Gradient Descent (SGD) optimizer.
pub struct SGD {
  params : Array[Tensor]
  lr : Double
}

///|
/// Create a new SGD optimizer.
pub fn SGD::new(params : Array[Tensor], lr : Double) -> SGD {
  { params, lr }
}

///|
/// Perform a single optimization step (updating parameter data).
pub fn SGD::step(self : SGD) -> Unit {
  for param in self.params {
    match param.grad {
      None => ()
      Some(g) => {
        let size = param.data.length()
        for i in 0..<size {
          param.data[i] = param.data[i] - self.lr * g[i]
        }
      }
    }
  }
}

///|
/// Reset the gradients of all parameters to zero.
pub fn SGD::zero_grad(self : SGD) -> Unit {
  for param in self.params {
    match param.grad {
      None => ()
      Some(g) => {
        let size = g.length()
        for i in 0..<size {
          g[i] = 0.0
        }
      }
    }
  }
}
