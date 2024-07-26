#ifndef TENSOR_UTILS_H
#define TENSOR_UTILS_H

#define ACTIVATION_FUNCTION_NONE    0
#define ACTIVATION_FUNCTION_TANH    1
#define ACTIVATION_FUNCTION_RELU    2
#define ACTIVATION_FUNCTION_SOFTMAX 3

using Tensor1D = float*;
using Tensor2D = float**;
using Tensor3D = float***;
using Tensor4D = float****;

// Function to create a 1D tensor (array)
Tensor1D tensor_1d(int cols);

// Function to create a 2D tensor (array of arrays)
Tensor2D tensor_2d(int rows, int cols);

// Function to create a 3D tensor (array of arrays of arrays)
Tensor3D tensor_3d(int chans, int rows, int cols);

// Function to create a 3D tensor (array of arrays of arrays of arrays)
Tensor4D tensor_4d(int n, int chans, int rows, int cols);

// Function to initialize a 1D tensor with specific values
Tensor1D tensor_init_1d(int cols, float values[]);

// Function to initialize a 2D tensor with specific values
Tensor2D tensor_init_2d(int rows, int cols, float values[]);

// Function to initialize a 3D tensor with specific values
Tensor3D tensor_init_3d(int chans, int rows, int cols, float values[]);

// Function to initialize a 4D tensor with specific values
Tensor4D tensor_init_4d(int n, int chans, int rows, int cols, float values[]);

// Function to print a 1D tensor
void tensor_print_1d(Tensor1D tensor, int cols);

// Function to print a 2D tensor
void tensor_print_2d(Tensor2D tensor, int rows, int cols);

// Function to print a 3D tensor
void tensor_print_3d(Tensor3D tensor, int chans, int rows, int cols);

// Function to calculate softmax for a 1D tensor
Tensor1D tensor_softmax_1d(Tensor1D tensor, int cols);

// Function to transpose a 2D tensor
Tensor2D tensor_transpose_2d(Tensor2D tensor, int rows, int cols);

// Function to perform matrix multiplication on 2D tensors
Tensor2D tensor_matmul_2d(Tensor2D m1, int m1_rows, int m1_cols, Tensor2D m2, int m2_rows, int m2_cols);

// Function to apply an activation function to a 3D tensor
Tensor3D tensor_apply_activation_3d(Tensor3D tensor, int chans, int rows, int cols, int activation_function);

// Function to perform convolution on a 3D tensor with a set of filters
Tensor3D tensor_conv_3d(Tensor3D tensor, int t_chans, int t_rows, int t_cols, Tensor4D filters, int n_f, int f_chans, int f_rows, int f_cols, int activation_function);

// Function to perform average pooling on a 3D tensor
Tensor3D tensor_avg_pooling_3d(Tensor3D tensor, int t_chans, int t_rows, int t_cols);

// Function to flatten a 3D tensor into a 1D tensor
Tensor1D tensor_flatten_3d(Tensor3D tensor, int chans, int rows, int cols);

// Function to apply a dense layer to a 1D tensor
Tensor1D tensor_dense_1d(Tensor1D tensor, int t_cols, Tensor2D weights, int w_rows, int w_cols, Tensor1D bias, int activation_function);

// Function to deallocate a 1D tensor
void tensor_delete_1d(Tensor1D tensor);

// Function to deallocate a 2D tensor
void tensor_delete_2d(Tensor2D tensor, int rows);

// Function to deallocate a 3D tensor
void tensor_delete_3d(Tensor3D tensor, int chans, int rows);

// Function to deallocate a 4D tensor
void tensor_delete_4d(Tensor4D tensor, int n, int chans, int rows);

#endif // TENSOR_UTILS_H
