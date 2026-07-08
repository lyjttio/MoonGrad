// cmd/main/main.mbt

fn main {
  println("==================================================")
  println("     MoonGrad: Pure MoonBit Autograd Demo         ")
  println("==================================================")
  println("Training a Multi-Layer Perceptron to solve XOR...")

  // XOR inputs (4 samples, 2 features)
  let x = @lib.from_array(
    [0.0, 0.0,
     0.0, 1.0,
     1.0, 0.0,
     1.0, 1.0],
    [4, 2]
  )
  // XOR targets (4 samples, 1 target)
  let y = @lib.from_array(
    [0.0,
     1.0,
     1.0,
     0.0],
    [4, 1]
  )

  // Network layers (2 -> 4 -> 1)
  let fc1 = @lib.Linear::new(2, 4, requires_grad=true)
  let fc2 = @lib.Linear::new(4, 1, requires_grad=true)

  // Gather parameters
  let params = @lib.Linear::parameters(fc1)
  let fc2_params = @lib.Linear::parameters(fc2)
  for i in 0..<fc2_params.length() {
    params.push(fc2_params[i])
  }

  // SGD optimizer with learning rate 1.5
  let optimizer = @lib.SGD::new(params, 1.5)

  // Train for 3000 epochs
  for epoch in 0..<3000 {
    // Forward pass
    let h1 = fc1.forward(x).sigmoid()
    let pred = fc2.forward(h1).sigmoid()

    // Calculate MSE Loss
    let loss = @lib.mse_loss(pred, y)

    // Backpropagate
    optimizer.zero_grad()
    loss.backward()

    // Update weights
    optimizer.step()

    if epoch % 300 == 0 || epoch == 2999 {
      println("Epoch: \{epoch}, Loss: \{loss.data[0]}")
    }
  }

  // Test final network predictions
  let h1 = fc1.forward(x).sigmoid()
  let final_pred = fc2.forward(h1).sigmoid()
  println("\nFinal Predictions:")
  println("Input: [0, 0] -> Output: \{final_pred.data[0]} (Target: 0.0)")
  println("Input: [0, 1] -> Output: \{final_pred.data[1]} (Target: 1.0)")
  println("Input: [1, 0] -> Output: \{final_pred.data[2]} (Target: 1.0)")
  println("Input: [1, 1] -> Output: \{final_pred.data[3]} (Target: 0.0)")
  println("==================================================")
}
