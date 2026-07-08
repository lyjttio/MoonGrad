// autograd.mbt

///| Enum representing operators in the computation graph.
pub enum Op {
  Add(Tensor, Tensor)
  Sub(Tensor, Tensor)
  Mul(Tensor, Tensor)
  Div(Tensor, Tensor)
  MatMul(Tensor, Tensor)
  Reshape(Tensor, Array[Int])
  Transpose(Tensor, Int, Int)
  ReLU(Tensor)
  Sigmoid(Tensor)
  Tanh(Tensor)
  MSELoss(Tensor, Tensor)
}

///| Helper to get the inputs of an operation.
fn get_op_inputs(op : Op) -> Array[Tensor] {
  match op {
    Add(a, b) => [a, b]
    Sub(a, b) => [a, b]
    Mul(a, b) => [a, b]
    Div(a, b) => [a, b]
    MatMul(a, b) => [a, b]
    Reshape(a, _) => [a]
    Transpose(a, _, _) => [a]
    ReLU(a) => [a]
    Sigmoid(a) => [a]
    Tanh(a) => [a]
    MSELoss(a, b) => [a, b]
  }
}

///| Performs topological sort starting from the root tensor.
fn topological_sort(root : Tensor) -> Array[Tensor] {
  let visited = Array::new()
  let result = Array::new()
  
  fn visit(node : Tensor) {
    let mut found = false
    for v in visited {
      if v.id == node.id {
        found = true
        break
      }
    }
    if not(found) {
      visited.push(node)
      match node.creator {
        Some(op) => {
          let inputs = get_op_inputs(op)
          for input in inputs {
            visit(input)
          }
        }
        None => ()
      }
      result.push(node)
    }
  }
  
  visit(root)
  result
}

///| Sum-reduce gradient delta along broadcasted dimensions.
fn reduce_grad(
  delta : Array[Double],
  out_shape : Array[Int],
  target_shape : Array[Int]
) -> Array[Double] {
  let target_len = target_shape.length()
  let mut target_size = 1
  for dim in target_shape {
    target_size = target_size * dim
  }
  if target_len == 0 {
    target_size = 1
  }
  
  let target_grad = Array::make(target_size, 0.0)
  let out_strides = shape_to_strides(out_shape)
  let target_strides = shape_to_strides(target_shape)
  
  let size = delta.length()
  let mut i = 0
  for ; i < size; i = i + 1 {
    let idx = broadcast_index(i, out_shape, out_strides, target_shape, target_strides)
    target_grad[idx] = target_grad[idx] + delta[i]
  }
  target_grad
}

///| Accumulate gradient delta into target Tensor.
pub fn accumulate_grad(t : Tensor, delta : Array[Double], out_shape : Array[Int]) -> Unit {
  if not(t.requires_grad) {
    return
  }
  let reduced = if t.shape != out_shape {
    reduce_grad(delta, out_shape, t.shape)
  } else {
    delta
  }
  
  match t.grad {
    None => {
      let g = Array::make(t.data.length(), 0.0)
      let mut i = 0
      for ; i < reduced.length(); i = i + 1 {
        g[i] = reduced[i]
      }
      t.grad = Some(g)
    }
    Some(g) => {
      let mut i = 0
      for ; i < reduced.length(); i = i + 1 {
        g[i] = g[i] + reduced[i]
      }
    }
  }
}

///| Propagate the gradient through an operation.
fn propagate_op_grad(op : Op, grad : Array[Double], out_shape : Array[Int]) -> Unit {
  match op {
    Add(a, b) => {
      accumulate_grad(a, grad, out_shape)
      accumulate_grad(b, grad, out_shape)
    }
    Sub(a, b) => {
      accumulate_grad(a, grad, out_shape)
      let neg_grad = Array::make(grad.length(), 0.0)
      let mut i = 0
      for ; i < grad.length(); i = i + 1 {
        neg_grad[i] = -grad[i]
      }
      accumulate_grad(b, neg_grad, out_shape)
    }
    Mul(a, b) => {
      let out_strides = shape_to_strides(out_shape)
      if a.requires_grad {
        let grad_a = Array::make(grad.length(), 0.0)
        let mut i = 0
        for ; i < grad.length(); i = i + 1 {
          let idx_b = broadcast_index(i, out_shape, out_strides, b.shape, b.strides)
          grad_a[i] = grad[i] * b.data[idx_b]
        }
        accumulate_grad(a, grad_a, out_shape)
      }
      if b.requires_grad {
        let grad_b = Array::make(grad.length(), 0.0)
        let mut i = 0
        for ; i < grad.length(); i = i + 1 {
          let idx_a = broadcast_index(i, out_shape, out_strides, a.shape, a.strides)
          grad_b[i] = grad[i] * a.data[idx_a]
        }
        accumulate_grad(b, grad_b, out_shape)
      }
    }
    Div(a, b) => {
      let out_strides = shape_to_strides(out_shape)
      if a.requires_grad {
        let grad_a = Array::make(grad.length(), 0.0)
        let mut i = 0
        for ; i < grad.length(); i = i + 1 {
          let idx_b = broadcast_index(i, out_shape, out_strides, b.shape, b.strides)
          grad_a[i] = grad[i] / b.data[idx_b]
        }
        accumulate_grad(a, grad_a, out_shape)
      }
      if b.requires_grad {
        let grad_b = Array::make(grad.length(), 0.0)
        let mut i = 0
        for ; i < grad.length(); i = i + 1 {
          let idx_a = broadcast_index(i, out_shape, out_strides, a.shape, a.strides)
          let idx_b = broadcast_index(i, out_shape, out_strides, b.shape, b.strides)
          let b_val = b.data[idx_b]
          grad_b[i] = -grad[i] * a.data[idx_a] / (b_val * b_val)
        }
        accumulate_grad(b, grad_b, out_shape)
      }
    }
    MatMul(a, b) => {
      let m = a.shape[0]
      let k = a.shape[1]
      let n = b.shape[1]
      if a.requires_grad {
        let grad_a = Array::make(m * k, 0.0)
        let mut r = 0
        for ; r < m; r = r + 1 {
          let mut c = 0
          for ; c < k; c = c + 1 {
            let mut sum = 0.0
            let mut j = 0
            for ; j < n; j = j + 1 {
              let g_val = grad[r * n + j]
              let b_val = b.data[c * b.strides[0] + j * b.strides[1]]
              sum = sum + g_val * b_val
            }
            grad_a[r * k + c] = sum
          }
        }
        accumulate_grad(a, grad_a, a.shape)
      }
      if b.requires_grad {
        let grad_b = Array::make(k * n, 0.0)
        let mut r = 0
        for ; r < k; r = r + 1 {
          let mut c = 0
          for ; c < n; c = c + 1 {
            let mut sum = 0.0
            let mut j = 0
            for ; j < m; j = j + 1 {
              let a_val = a.data[j * a.strides[0] + r * a.strides[1]]
              let g_val = grad[j * n + c]
              sum = sum + a_val * g_val
            }
            grad_b[r * n + c] = sum
          }
        }
        accumulate_grad(b, grad_b, b.shape)
      }
    }
    Reshape(a, _) => {
      accumulate_grad(a, grad, a.shape)
    }
    Transpose(a, dim0, dim1) => {
      if a.requires_grad {
        let size = grad.length()
        let grad_a = Array::make(size, 0.0)
        let len = a.shape.length()
        let a_strides = a.strides
        let new_strides = shape_to_strides(out_shape)
        
        let mut i = 0
        for ; i < size; i = i + 1 {
          let mut remaining = i
          let coords = Array::make(len, 0)
          let mut d = 0
          for ; d < len; d = d + 1 {
            coords[d] = remaining / a_strides[d]
            remaining = remaining % a_strides[d]
          }
          
          let mut trans_idx = 0
          let mut k = 0
          for ; k < len; k = k + 1 {
            let coord = if k == dim0 { coords[dim1] }
                        else if k == dim1 { coords[dim0] }
                        else { coords[k] }
            trans_idx = trans_idx + coord * new_strides[k]
          }
          grad_a[i] = grad[trans_idx]
        }
        accumulate_grad(a, grad_a, a.shape)
      }
    }
    ReLU(a) => {
      if a.requires_grad {
        let grad_a = Array::make(grad.length(), 0.0)
        let mut i = 0
        for ; i < grad.length(); i = i + 1 {
          grad_a[i] = if a.data[i] > 0.0 { grad[i] } else { 0.0 }
        }
        accumulate_grad(a, grad_a, out_shape)
      }
    }
    Sigmoid(a) => {
      if a.requires_grad {
        let grad_a = Array::make(grad.length(), 0.0)
        let mut i = 0
        for ; i < grad.length(); i = i + 1 {
          // s = 1 / (1 + exp(-x))
          let x = a.data[i]
          let s = 1.0 / (1.0 + exp(-x))
          grad_a[i] = grad[i] * s * (1.0 - s)
        }
        accumulate_grad(a, grad_a, out_shape)
      }
    }
    Tanh(a) => {
      if a.requires_grad {
        let grad_a = Array::make(grad.length(), 0.0)
        let mut i = 0
        for ; i < grad.length(); i = i + 1 {
          let x = a.data[i]
          let ex = exp(x)
          let enx = exp(-x)
          let th = (ex - enx) / (ex + enx)
          grad_a[i] = grad[i] * (1.0 - th * th)
        }
        accumulate_grad(a, grad_a, out_shape)
      }
    }
    MSELoss(pred, target) => {
      let size = pred.data.length()
      let factor = 2.0 / size.to_double() * grad[0]
      if pred.requires_grad {
        let grad_pred = Array::make(size, 0.0)
        let mut i = 0
        for ; i < size; i = i + 1 {
          grad_pred[i] = (pred.data[i] - target.data[i]) * factor
        }
        accumulate_grad(pred, grad_pred, pred.shape)
      }
      if target.requires_grad {
        let grad_target = Array::make(size, 0.0)
        let mut i = 0
        for ; i < size; i = i + 1 {
          grad_target[i] = -(pred.data[i] - target.data[i]) * factor
        }
        accumulate_grad(target, grad_target, target.shape)
      }
    }
  }
}

///| Runs backpropagation starting from this Tensor.
pub fn Tensor::backward(self : Tensor) -> Unit {
  // Initialize gradient of root if not set
  if self.grad.is_empty() {
    self.grad = Some(Array::make(self.data.length(), 1.0))
  }
  
  let sorted = topological_sort(self)
  let len = sorted.length()
  let mut i = len - 1
  for ; i >= 0; i = i - 1 {
    let t = sorted[i]
    match t.creator {
      None => ()
      Some(op) => {
        match t.grad {
          None => ()
          Some(g) => propagate_op_grad(op, g, t.shape)
        }
      }
    }
  }
}
