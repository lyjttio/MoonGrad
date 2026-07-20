// moongrad_test.mbt

///|
fn assert_approx_eq(a : Double, b : Double, epsilon : Double) -> Unit {
  let diff = a - b
  let abs_diff = if diff < 0.0 { -diff } else { diff }
  if abs_diff > epsilon {
    println("Assertion failed: \{a} is not approximately equal to \{b}")
    panic()
  }
}

///|
test "tensor creation and indexing" {
  let t = from_array([1.0, 2.0, 3.0, 4.0], [2, 2])
  assert_approx_eq(t[[0, 0]], 1.0, 0.000001)
  assert_approx_eq(t[[0, 1]], 2.0, 0.000001)
  assert_approx_eq(t[[1, 0]], 3.0, 0.000001)
  assert_approx_eq(t[[1, 1]], 4.0, 0.000001)
}

///|
test "reshape forward and out-of-bounds safety" {
  let x = from_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
  let y = x.reshape([3, 2])
  assert_approx_eq(y[[0, 0]], 1.0, 0.000001)
  assert_approx_eq(y[[0, 1]], 2.0, 0.000001)
  assert_approx_eq(y[[1, 0]], 3.0, 0.000001)
  assert_approx_eq(y[[1, 1]], 4.0, 0.000001)
  assert_approx_eq(y[[2, 0]], 5.0, 0.000001)
  assert_approx_eq(y[[2, 1]], 6.0, 0.000001)

  let z = x.reshape([1, 2, 3])
  assert_approx_eq(z[[0, 1, 2]], 6.0, 0.000001)
}

///|
test "reshape autograd backward pass" {
  let x = from_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3], requires_grad=true)
  let y = x.reshape([3, 2])
  let target = ones([3, 2])
  let loss = mse_loss(y, target)
  loss.backward()

  let gx = match x.grad {
    Some(g) => g
    None => []
  }

  let eps = 0.000001
  x.data[2] = x.data[2] + eps
  let y_p = x.reshape([3, 2])
  let loss_p = mse_loss(y_p, target)

  x.data[2] = x.data[2] - 2.0 * eps
  let y_m = x.reshape([3, 2])
  let loss_m = mse_loss(y_m, target)

  x.data[2] = x.data[2] + eps
  let num_grad = (loss_p.data[0] - loss_m.data[0]) / (2.0 * eps)
  assert_approx_eq(gx[2], num_grad, 0.00001)
}

///|
test "transpose forward and autograd backward pass" {
  let x = from_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3], requires_grad=true)
  let y = x.t()
  assert_approx_eq(y[[0, 0]], 1.0, 0.000001)
  assert_approx_eq(y[[1, 0]], 2.0, 0.000001)
  assert_approx_eq(y[[2, 0]], 3.0, 0.000001)
  assert_approx_eq(y[[0, 1]], 4.0, 0.000001)

  let target = zeros([3, 2])
  let loss = mse_loss(y, target)
  loss.backward()

  let gx = match x.grad {
    Some(g) => g
    None => []
  }

  let eps = 0.000001
  x.data[1] = x.data[1] + eps
  let y_p = x.t()
  let loss_p = mse_loss(y_p, target)

  x.data[1] = x.data[1] - 2.0 * eps
  let y_m = x.t()
  let loss_m = mse_loss(y_m, target)

  x.data[1] = x.data[1] + eps
  let num_grad = (loss_p.data[0] - loss_m.data[0]) / (2.0 * eps)
  assert_approx_eq(gx[1], num_grad, 0.00001)
}

///|
test "slice and slice_multi forward and backward" {
  let x = from_array(
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    [3, 3],
    requires_grad=true,
  )
  let s = x.slice(1, 1, 3)
  assert_approx_eq(s.shape[0].to_double(), 3.0, 0.000001)
  assert_approx_eq(s.shape[1].to_double(), 2.0, 0.000001)
  assert_approx_eq(s[[0, 0]], 2.0, 0.000001)
  assert_approx_eq(s[[0, 1]], 3.0, 0.000001)

  let target = zeros([3, 2])
  let loss = mse_loss(s, target)
  loss.backward()

  let gx = match x.grad {
    Some(g) => g
    None => []
  }

  assert_approx_eq(gx[0], 0.0, 0.000001)
  assert_approx_eq(gx[1], 2.0 / 6.0 * 2.0, 0.000001)

  let sm = x.slice_multi([(1, 3), (0, 2)])
  assert_approx_eq(sm[[0, 0]], 4.0, 0.000001)
  assert_approx_eq(sm[[1, 1]], 8.0, 0.000001)
}

///|
test "tensor broadcasting forward and backward gradients" {
  let a = from_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3], requires_grad=true)
  let b = from_array([10.0, 20.0], [2, 1], requires_grad=true)
  let c = a + b

  assert_approx_eq(c[[0, 0]], 11.0, 0.000001)
  assert_approx_eq(c[[0, 2]], 13.0, 0.000001)
  assert_approx_eq(c[[1, 0]], 24.0, 0.000001)

  let target = zeros([2, 3])
  let loss = mse_loss(c, target)
  loss.backward()

  let gb = match b.grad {
    Some(g) => g
    None => []
  }

  let eps = 0.000001
  b.data[0] = b.data[0] + eps
  let c_p = a + b
  let loss_p = mse_loss(c_p, target)

  b.data[0] = b.data[0] - 2.0 * eps
  let c_m = a + b
  let loss_m = mse_loss(c_m, target)

  b.data[0] = b.data[0] + eps
  let num_grad = (loss_p.data[0] - loss_m.data[0]) / (2.0 * eps)
  assert_approx_eq(gb[0], num_grad, 0.00001)
}

///|
test "autograd activation functions (ReLU, Sigmoid, Tanh)" {
  let x = from_array([-2.0, 0.5, 1.5], [3], requires_grad=true)

  let r = x.relu()
  assert_approx_eq(r[[0]], 0.0, 0.000001)
  assert_approx_eq(r[[1]], 0.5, 0.000001)

  let s = x.sigmoid()
  assert_approx_eq(s[[1]], 1.0 / (1.0 + exp(-0.5)), 0.000001)

  let th = x.tanh()
  let ex = exp(0.5)
  let enx = exp(-0.5)
  assert_approx_eq(th[[1]], (ex - enx) / (ex + enx), 0.000001)

  let target = zeros([3])
  let loss = mse_loss(th, target)
  loss.backward()

  let gx = match x.grad {
    Some(g) => g
    None => []
  }

  let eps = 0.000001
  x.data[1] = x.data[1] + eps
  let th_p = x.tanh()
  let loss_p = mse_loss(th_p, target)

  x.data[1] = x.data[1] - 2.0 * eps
  let th_m = x.tanh()
  let loss_m = mse_loss(th_m, target)

  x.data[1] = x.data[1] + eps
  let num_grad = (loss_p.data[0] - loss_m.data[0]) / (2.0 * eps)
  assert_approx_eq(gx[1], num_grad, 0.00001)
}

///|
test "linear layer forward and parameters" {
  let fc = Linear::new(3, 2, requires_grad=true)
  let params = fc.parameters()
  assert_approx_eq(params.length().to_double(), 2.0, 0.000001)
  assert_approx_eq(params[0].shape[0].to_double(), 3.0, 0.000001)
  assert_approx_eq(params[0].shape[1].to_double(), 2.0, 0.000001)

  let x = ones([4, 3], requires_grad=true)
  let out = fc.forward(x)
  assert_approx_eq(out.shape[0].to_double(), 4.0, 0.000001)
  assert_approx_eq(out.shape[1].to_double(), 2.0, 0.000001)

  let loss = mse_loss(out, zeros([4, 2]))
  loss.backward()

  let w_grad = match params[0].grad {
    Some(g) => g.length()
    None => 0
  }
  assert_approx_eq(w_grad.to_double(), 6.0, 0.000001)
}

///|
test "sgd optimizer step and zero_grad" {
  let w = from_array([2.0], [1], requires_grad=true)
  let opt = SGD::new([w], 0.1)

  let loss = (w - from_array([10.0], [1])) * (w - from_array([10.0], [1]))
  loss.backward()

  let initial_grad = match w.grad {
    Some(g) => g[0]
    None => 0.0
  }
  assert_approx_eq(initial_grad, -16.0, 0.000001)

  opt.step()
  assert_approx_eq(w.data[0], 3.6, 0.000001)

  opt.zero_grad()
  let cleared_grad = match w.grad {
    Some(g) => g[0]
    None => -1.0
  }
  assert_approx_eq(cleared_grad, 0.0, 0.000001)
}
