// moongrad_test.mbt

fn assert_approx_eq(a : Double, b : Double, epsilon : Double) -> Unit {
  let diff = a - b
  let abs_diff = if diff < 0.0 { -diff } else { diff }
  if abs_diff > epsilon {
    println("Assertion failed: \{a} is not approximately equal to \{b}")
    panic()
  }
}

test "tensor creation and indexing" {
  let t = from_array([1.0, 2.0, 3.0, 4.0], [2, 2])
  assert_approx_eq(t[[0, 0]], 1.0, 1e-6)
  assert_approx_eq(t[[0, 1]], 2.0, 1e-6)
  assert_approx_eq(t[[1, 0]], 3.0, 1e-6)
  assert_approx_eq(t[[1, 1]], 4.0, 1e-6)
}

test "tensor broadcasting" {
  let a = from_array([1.0, 2.0], [1, 2])
  let b = from_array([3.0, 4.0], [2, 1])
  let c = a + b
  assert_approx_eq(c[[0, 0]], 4.0, 1e-6)
  assert_approx_eq(c[[0, 1]], 5.0, 1e-6)
  assert_approx_eq(c[[1, 0]], 5.0, 1e-6)
  assert_approx_eq(c[[1, 1]], 6.0, 1e-6)
}

test "autograd simple arithmetic" {
  let x = from_array([2.0], [1], requires_grad: true)
  let y = from_array([3.0], [1], requires_grad: true)
  let z = x * y + x
  z.backward()
  let gx = match x.grad { Some(g) => g[0]; None => 0.0 }
  let gy = match y.grad { Some(g) => g[0]; None => 0.0 }
  assert_approx_eq(gx, 4.0, 1e-6)
  assert_approx_eq(gy, 2.0, 1e-6)
}

test "autograd division" {
  let x = from_array([4.0], [1], requires_grad: true)
  let y = from_array([2.0], [1], requires_grad: true)
  let z = x / y
  z.backward()
  let gx = match x.grad { Some(g) => g[0]; None => 0.0 }
  let gy = match y.grad { Some(g) => g[0]; None => 0.0 }
  assert_approx_eq(gx, 0.5, 1e-6)
  assert_approx_eq(gy, -1.0, 1e-6)
}

test "autograd matmul" {
  let x = from_array([1.0, 2.0, 3.0, 4.0], [2, 2], requires_grad: true)
  let w = from_array([0.5, -0.5, 1.0, 2.0], [2, 2], requires_grad: true)
  let y = x.matmul(w)
  let target = zeros([2, 2])
  let loss = mse_loss(y, target)
  loss.backward()
  
  // Verify prediction at index [0, 0] is 1*0.5 + 2*1 = 2.5
  assert_approx_eq(y[[0, 0]], 2.5, 1e-6)
  
  let gx = match x.grad { Some(g) => g; None => [] }
  let gw = match w.grad { Some(g) => g; None => [] }
  
  // Numerical gradient check for x[0]
  let eps = 1e-6
  x.data[0] = x.data[0] + eps
  let y_p = x.matmul(w)
  let loss_p = mse_loss(y_p, target)
  
  x.data[0] = x.data[0] - 2.0 * eps
  let y_m = x.matmul(w)
  let loss_m = mse_loss(y_m, target)
  
  x.data[0] = x.data[0] + eps // restore
  let num_grad = (loss_p.data[0] - loss_m.data[0]) / (2.0 * eps)
  assert_approx_eq(gx[0], num_grad, 1e-5)
}

test "autograd activation functions" {
  let x = from_array([-1.0, 1.0, 2.0], [3], requires_grad: true)
  let y = x.relu()
  let target = zeros([3])
  let loss = mse_loss(y, target)
  loss.backward()
  
  let gx = match x.grad { Some(g) => g; None => [] }
  
  // relu derivative at x = -1 is 0. loss derivative at y = 0 is 0.
  assert_approx_eq(gx[0], 0.0, 1e-6)
  
  // relu derivative at x = 1 is 1. loss derivative at y = 1 is 2 * 1 / 3 = 2/3.
  assert_approx_eq(gx[1], 2.0 / 3.0, 1e-6)
}
