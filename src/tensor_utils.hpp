#ifndef TENSOR_UTILS_H
#define TENSOR_UTILS_H

#define ACTIVATION_FUNCTION_NONE    0
#define ACTIVATION_FUNCTION_TANH    1
#define ACTIVATION_FUNCTION_RELU    2
#define ACTIVATION_FUNCTION_SOFTMAX 3

#include <vector>

using Tensor1D = std::vector<float>;
using Tensor2D = std::vector<Tensor1D>;
using Tensor3D = std::vector<Tensor2D>;

void tensor_print_1d(Tensor1D tensor);

void tensor_print_2d(Tensor2D tensor);

void tensor_print_3d(Tensor3D tensor);

Tensor1D tensor_softmax_1d(Tensor1D tensor);

Tensor2D tensor_transpose_2d(Tensor2D tensor);

Tensor2D tensor_matmul_2d(Tensor2D m1, Tensor2D m2);

Tensor3D tensor_apply_activation_3d(Tensor3D tensor, int activation_function);

Tensor3D tensor_conv_3d(Tensor3D tensor, std::vector<Tensor3D> filters, int activation_function);

Tensor3D tensor_avg_pooling_3d(Tensor3D tensor);

Tensor1D tensor_flatten_3d(Tensor3D tensor);

Tensor1D tensor_dense_1d(Tensor1D tensor, Tensor2D weights, Tensor1D bias, int activation_function);

#endif // TENSOR_UTILS_H
