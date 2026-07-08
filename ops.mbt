// ops.mbt

///| Broadcast two shapes and return the output shape. Returns None if incompatible.
pub fn broadcast_shapes(
  shape1 : Array[Int],
  shape2 : Array[Int]
) -> Array[Int]? {
  let len1 = shape1.length()
  let len2 = shape2.length()
  let max_len = if len1 > len2 { len1 } else { len2 }
  let out_shape = Array::make(max_len, 1)
  
  let mut i = 0
  for ; i < max_len; i = i + 1 {
    let d1 = if i < max_len - len1 { 1 } else { shape1[i - (max_len - len1)] }
    let d2 = if i < max_len - len2 { 1 } else { shape2[i - (max_len - len2)] }
    if d1 == d2 {
      out_shape[i] = d1
    } else if d1 == 1 {
      out_shape[i] = d2
    } else if d2 == 1 {
      out_shape[i] = d1
    } else {
      return None
    }
  }
  Some(out_shape)
}

///| Map a flat index from broadcasted output shape back to the target shape.
fn broadcast_index(
  flat_idx : Int,
  out_shape : Array[Int],
  out_strides : Array[Int],
  target_shape : Array[Int],
  target_strides : Array[Int]
) -> Int {
  let out_len = out_shape.length()
  let target_len = target_shape.length()
  if target_len == 0 {
    return 0
  }
  let mut target_idx = 0
  let mut remaining = flat_idx
  let mut i = 0
  for ; i < out_len; i = i + 1 {
    let coord = remaining / out_strides[i]
    remaining = remaining % out_strides[i]
    
    let target_dim_idx = i - (out_len - target_len)
    if target_dim_idx >= 0 {
      let dim_size = target_shape[target_dim_idx]
      let coord_mapped = if dim_size == 1 { 0 } else { coord }
      target_idx = target_idx + coord_mapped * target_strides[target_dim_idx]
    }
  }
  target_idx
}

///| Element-wise addition of two Tensors with broadcasting support.
pub fn Tensor::op_add(self : Tensor, other : Tensor) -> Tensor {
  let out_shape = match broadcast_shapes(self.shape, other.shape) {
    Some(s) => s
    None => panic() // Broadcast failed
  }
  let out_strides = shape_to_strides(out_shape)
  let mut size = 1
  for dim in out_shape {
    size = size * dim
  }
  if out_shape.length() == 0 {
    size = 1
  }
  let data = Array::make(size, 0.0)
  let mut i = 0
  for ; i < size; i = i + 1 {
    let idx1 = broadcast_index(i, out_shape, out_strides, self.shape, self.strides)
    let idx2 = broadcast_index(i, out_shape, out_strides, other.shape, other.strides)
    data[i] = self.data[idx1] + other.data[idx2]
  }
  let req_grad = self.requires_grad || other.requires_grad
  let out = Tensor::new(data, out_shape, requires_grad: req_grad)
  if req_grad {
    out.creator = Some(Add(self, other))
  }
  out
}

///| Element-wise subtraction of two Tensors with broadcasting support.
pub fn Tensor::op_sub(self : Tensor, other : Tensor) -> Tensor {
  let out_shape = match broadcast_shapes(self.shape, other.shape) {
    Some(s) => s
    None => panic()
  }
  let out_strides = shape_to_strides(out_shape)
  let mut size = 1
  for dim in out_shape {
    size = size * dim
  }
  if out_shape.length() == 0 {
    size = 1
  }
  let data = Array::make(size, 0.0)
  let mut i = 0
  for ; i < size; i = i + 1 {
    let idx1 = broadcast_index(i, out_shape, out_strides, self.shape, self.strides)
    let idx2 = broadcast_index(i, out_shape, out_strides, other.shape, other.strides)
    data[i] = self.data[idx1] - other.data[idx2]
  }
  let req_grad = self.requires_grad || other.requires_grad
  let out = Tensor::new(data, out_shape, requires_grad: req_grad)
  if req_grad {
    out.creator = Some(Sub(self, other))
  }
  out
}

///| Element-wise multiplication of two Tensors with broadcasting support.
pub fn Tensor::op_mul(self : Tensor, other : Tensor) -> Tensor {
  let out_shape = match broadcast_shapes(self.shape, other.shape) {
    Some(s) => s
    None => panic()
  }
  let out_strides = shape_to_strides(out_shape)
  let mut size = 1
  for dim in out_shape {
    size = size * dim
  }
  if out_shape.length() == 0 {
    size = 1
  }
  let data = Array::make(size, 0.0)
  let mut i = 0
  for ; i < size; i = i + 1 {
    let idx1 = broadcast_index(i, out_shape, out_strides, self.shape, self.strides)
    let idx2 = broadcast_index(i, out_shape, out_strides, other.shape, other.strides)
    data[i] = self.data[idx1] * other.data[idx2]
  }
  let req_grad = self.requires_grad || other.requires_grad
  let out = Tensor::new(data, out_shape, requires_grad: req_grad)
  if req_grad {
    out.creator = Some(Mul(self, other))
  }
  out
}

///| Element-wise division of two Tensors with broadcasting support.
pub fn Tensor::op_div(self : Tensor, other : Tensor) -> Tensor {
  let out_shape = match broadcast_shapes(self.shape, other.shape) {
    Some(s) => s
    None => panic()
  }
  let out_strides = shape_to_strides(out_shape)
  let mut size = 1
  for dim in out_shape {
    size = size * dim
  }
  if out_shape.length() == 0 {
    size = 1
  }
  let data = Array::make(size, 0.0)
  let mut i = 0
  for ; i < size; i = i + 1 {
    let idx1 = broadcast_index(i, out_shape, out_strides, self.shape, self.strides)
    let idx2 = broadcast_index(i, out_shape, out_strides, other.shape, other.strides)
    data[i] = self.data[idx1] / other.data[idx2]
  }
  let req_grad = self.requires_grad || other.requires_grad
  let out = Tensor::new(data, out_shape, requires_grad: req_grad)
  if req_grad {
    out.creator = Some(Div(self, other))
  }
  out
}
