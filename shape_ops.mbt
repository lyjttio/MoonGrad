// shape_ops.mbt

///|
/// Reshape a Tensor to a new shape. The total size must remain the same.
pub fn Tensor::reshape(self : Tensor, new_shape : Array[Int]) -> Tensor {
  let mut old_size = 1
  for dim in self.shape {
    old_size = old_size * dim
  }
  if self.shape.length() == 0 {
    old_size = 1
  }
  let mut new_size = 1
  for dim in new_shape {
    new_size = new_size * dim
  }
  if new_shape.length() == 0 {
    new_size = 1
  }
  if old_size != new_size {
    panic()
  }

  let data = Array::make(old_size, 0.0)
  let out_strides = shape_to_strides(new_shape)
  for i in 0..<old_size {
    let idx = broadcast_index(
      i,
      new_shape,
      out_strides,
      self.shape,
      self.strides,
    )
    data[i] = self.data[idx]
  }

  let req_grad = self.requires_grad
  let out = Tensor::new(data, new_shape, requires_grad=req_grad)
  if req_grad {
    out.creator = Some(Reshape(self, new_shape))
  }
  out
}

///|
/// Transpose two dimensions of a Tensor.
pub fn Tensor::transpose(self : Tensor, dim0 : Int, dim1 : Int) -> Tensor {
  let len = self.shape.length()
  if dim0 >= len || dim1 >= len || dim0 < 0 || dim1 < 0 {
    panic()
  }
  let new_shape = Array::make(len, 0)
  for i in 0..<len {
    if i == dim0 {
      new_shape[i] = self.shape[dim1]
    } else if i == dim1 {
      new_shape[i] = self.shape[dim0]
    } else {
      new_shape[i] = self.shape[i]
    }
  }
  let new_strides = shape_to_strides(new_shape)

  let mut size = 1
  for dim in new_shape {
    size = size * dim
  }
  if len == 0 {
    size = 1
  }
  let data = Array::make(size, 0.0)

  for j in 0..<size {
    let mut remaining = j
    let coords = Array::make(len, 0)
    for d in 0..<len {
      coords[d] = remaining / new_strides[d]
      remaining = remaining % new_strides[d]
    }

    let orig_coords = Array::make(len, 0)
    for k in 0..<len {
      if k == dim0 {
        orig_coords[k] = coords[dim1]
      } else if k == dim1 {
        orig_coords[k] = coords[dim0]
      } else {
        orig_coords[k] = coords[k]
      }
    }

    let mut orig_idx = 0
    for m in 0..<len {
      orig_idx = orig_idx + orig_coords[m] * self.strides[m]
    }
    data[j] = self.data[orig_idx]
  }

  let req_grad = self.requires_grad
  let out = Tensor::new(data, new_shape, requires_grad=req_grad)
  if req_grad {
    out.creator = Some(Transpose(self, dim0, dim1))
  }
  out
}

///|
/// Shortcut to transpose a 2D Tensor (matrix).
pub fn Tensor::t(self : Tensor) -> Tensor {
  if self.shape.length() != 2 {
    panic()
  }
  self.transpose(0, 1)
}
