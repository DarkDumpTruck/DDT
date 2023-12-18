#include <gtest/gtest.h>
#include <torch/torch.h>

#include "core/util/common.h"

TEST(LibtorchCpuBasicTest, Simple) {
  std::cout << "hello from libtorch!" << std::endl;
  auto tensor = torch::randn({1, 2, 3, 5, 24});
  std::cout << "generated a random tensor of size: " << tensor.sizes()
            << std::endl;

  // torch::randn should be on CPU.
  ASSERT_TRUE(tensor.is_cpu());

  auto mean = torch::mean(tensor).item<double>(),
       max = torch::max(tensor).item<double>();

  // mean of randn() should be close to 0
  ASSERT_TRUE(mean > -1 && mean < 1);

  // max of randn() should be larger than 1
  ASSERT_TRUE(max > 1);
}

TEST(LibtorchCpuBasicTest, Grad) {
  auto x = torch::ones({1}, torch::requires_grad().dtype(torch::kFloat64));
  auto y = torch::sin(x);
  y.backward(torch::ones_like(x));

  // grad of sin(x) should be cos(x)
  ASSERT_DOUBLE_EQ(x.grad().item<double>(), std::cos(1));
}

TEST(LibtorchCpuBasicTest, Grad2) {
  auto x = torch::tensor({1.f}, torch::requires_grad());
  auto y = torch::tensor({2.f}, torch::requires_grad());
  auto z = torch::exp(x) * y + torch::sigmoid(x);

  auto grad_output = torch::ones_like(z);
  auto grad_x =
      torch::autograd::grad({z}, {x}, {grad_output},
                            /*retain_graph=*/true, /*create_graph=*/true)[0];

  auto e = std::exp(1), e_ = std::exp(-1);

  // grad of exp(x) * y + sigmoid(x) should be exp(x) * y + sigmoid'(x)
  ASSERT_FLOAT_EQ(grad_x.item<float>(), 2 * e + e_ / (1 + e_) / (1 + e_));

  auto grad_y = torch::autograd::grad({z}, {y})[0];
  // grad of exp(x) * y + sigmoid(x) should be exp(x)
  ASSERT_FLOAT_EQ(grad_y.item<float>(), e);
}
