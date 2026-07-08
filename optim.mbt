// optim.mbt

///| Stochastic Gradient Descent (SGD) optimizer.
pub struct SGD {
  pub params : Array[Tensor]
  pub mut lr : Double
}

///| Create a new SGD optimizer.
pub fn SGD::new(params : Array[Tensor], lr : Double) -> SGD {
  { params, lr }
}

///| Perform a single optimization step (updating parameter data).
pub fn SGD::step(self : SGD) -> Unit {
  for param in self.params {
    match param.grad {
      None => ()
      Some(g) => {
        let size = param.data.length()
        let mut i = 0
        for ; i < size; i = i + 1 {
          param.data[i] = param.data[i] - self.lr * g[i]
        }
      }
    }
  }
}

///| Reset the gradients of all parameters to zero.
pub fn SGD::zero_grad(self : SGD) -> Unit {
  for param in self.params {
    match param.grad {
      None => ()
      Some(g) => {
        let size = g.length()
        let mut i = 0
        for ; i < size; i = i + 1 {
          g[i] = 0.0
        }
      }
    }
  }
}
