#include "tensor_utils.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>

#define ACTIVATION_FUNCTION_NONE    0
#define ACTIVATION_FUNCTION_TANH    1
#define ACTIVATION_FUNCTION_RELU    2
#define ACTIVATION_FUNCTION_SOFTMAX 3

using Tensor1D = float*;
using Tensor2D = float**;
using Tensor3D = float***;
using Tensor4D = float****;

// Function to create a 1D tensor (array)
Tensor1D tensor_1d(int cols) {
  Tensor1D res = new float[cols]();
  return res;
}

// Function to create a 2D tensor (array of arrays)
Tensor2D tensor_2d(int rows, int cols) {
  Tensor2D res = new Tensor1D[rows];
  for (int i = 0; i < rows; ++i) {
    res[i] = new float[cols]();
  }
  return res;
}

// Function to create a 3D tensor (array of arrays of arrays)
Tensor3D tensor_3d(int chans, int rows, int cols) {
  Tensor3D res = new Tensor2D[chans];
  for (int i = 0; i < chans; ++i) {
    res[i] = new Tensor1D[rows];
    for (int j = 0; j < rows; ++j) {
      res[i][j] = new float[cols]();
    }
  }
  return res;
}

// Function to create a 4D tensor (array of arrays of arrays of arrays)
Tensor4D tensor_4d(int n, int chans, int rows, int cols) {
  Tensor4D res = new Tensor3D[n];
  for (int i = 0; i < n; ++i) {
    res[i] = new Tensor2D[chans];
    for (int j = 0; j < chans; ++j) {
      res[i][j] = new Tensor1D[rows];
      for (int k = 0; k < rows; ++k) {
        res[i][j][k] = new float[cols]();
      }
    }
  }
  return res;
}

// Function to initialize a 1D tensor with specific values
Tensor1D tensor_init_1d(int cols, float values[]) {
  Tensor1D tensor = tensor_1d(cols);
  for (int i = 0; i < cols; i++) {
    tensor[i] = values[i];
  }
  return tensor;
}

// Function to initialize a 2D tensor with specific values
Tensor2D tensor_init_2d(int rows, int cols, float values[]) {
  Tensor2D tensor = tensor_2d(rows, cols);
  int index = 0;
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      tensor[i][j] = values[index++];
    }
  }
  return tensor;
}

// Function to initialize a 3D tensor with specific values
Tensor3D tensor_init_3d(int chans, int rows, int cols, float values[]) {
  Tensor3D tensor = tensor_3d(chans, rows, cols);
  int index = 0;
  for (int i = 0; i < chans; i++) {
    for (int j = 0; j < rows; j++) {
      for (int k = 0; k < cols; k++) {
        tensor[i][j][k] = values[index++];
      }
    }
  }
  return tensor;
}

// Function to initialize a 4D tensor with specific values
Tensor4D tensor_init_4d(int n, int chans, int rows, int cols, float values[]) {
  Tensor4D tensor = tensor_4d(n, chans, rows, cols);
  int index = 0;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < chans; j++) {
      for (int k = 0; k < rows; k++) {
        for (int l = 0; l < cols; l++) {
          tensor[i][j][k][l] = values[index++];
        }
      }
    }
  }
  return tensor;
}

// Function to print a 1D tensor
void tensor_print_1d(Tensor1D tensor, int cols) {
  std::cout << std::endl;
  for (int i = 0; i < cols; i++) {
    std::cout << tensor[i] << " ";
  }
  std::cout << std::endl << std::endl;
}

// Function to print a 2D tensor
void tensor_print_2d(Tensor2D tensor, int rows, int cols) {
  std::cout << std::endl;
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      std::cout << tensor[i][j] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

// Function to print a 3D tensor
void tensor_print_3d(Tensor3D tensor, int chans, int rows, int cols) {
  std::cout << std::endl;
  for (int i = 0; i < chans; i++) {
    for (int j = 0; j < rows; j++) {
      for (int k = 0; k < cols; k++) {
        std::cout << tensor[i][j][k] << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }
}

// Function to calculate softmax for a 1D tensor
Tensor1D tensor_softmax_1d(Tensor1D tensor, int cols) {
  float max_val = *std::max_element(tensor, tensor + cols);
  Tensor1D exp_values = new float[cols];
  float sum_exp = 0.0f;
  for (int i = 0; i < cols; i++) {
    exp_values[i] = std::exp(tensor[i] - max_val);
    sum_exp += exp_values[i];
  }
  for (int i = 0; i < cols; i++) {
    exp_values[i] /= sum_exp;
  }
  return exp_values;
}

// Function to transpose a 2D tensor
Tensor2D tensor_transpose_2d(Tensor2D tensor, int rows, int cols) {
  Tensor2D res = tensor_2d(cols, rows);
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      res[j][i] = tensor[i][j];
    }
  }
  return res;
}

// Function to perform matrix multiplication on 2D tensors
Tensor2D tensor_matmul_2d(Tensor2D m1, int m1_rows, int m1_cols, Tensor2D m2, int m2_rows, int m2_cols) {
  if (m1_cols != m2_rows) {
    throw std::invalid_argument("Incompatible matrices shapes for matmul.");
  }
  Tensor2D res = tensor_2d(m1_rows, m2_cols);
  for (int i = 0; i < m1_rows; ++i) {
    for (int j = 0; j < m2_cols; ++j) {
      for (int k = 0; k < m1_cols; ++k) {
        res[i][j] += m1[i][k] * m2[k][j];
      }
    }
  }
  return res;
}

// Function to apply an activation function to a 3D tensor
Tensor3D tensor_apply_activation_3d(Tensor3D tensor, int chans, int rows, int cols, int activation_function) {
  Tensor3D res = tensor_3d(chans, rows, cols);
  switch (activation_function) {
    case ACTIVATION_FUNCTION_TANH:
      for (int i = 0; i < chans; i++) {
        for (int j = 0; j < rows; j++) {
          for (int k = 0; k < cols; k++) {
            res[i][j][k] = std::tanh(tensor[i][j][k]);
          }
        }
      }
      break;
    case ACTIVATION_FUNCTION_RELU:
      for (int i = 0; i < chans; i++) {
        for (int j = 0; j < rows; j++) {
          for (int k = 0; k < cols; k++) {
            res[i][j][k] = std::max(0.0f, tensor[i][j][k]);
          }
        }
      }
      break;
    case ACTIVATION_FUNCTION_SOFTMAX:
      for (int i = 0; i < chans; i++) {
        for (int j = 0; j < rows; j++) {
          Tensor1D temp = tensor_softmax_1d(tensor[i][j], cols);
          for (int k = 0; k < cols; k++) {
            res[i][j][k] = temp[k];
          }
          delete[] temp;
        }
      }
      break;
    case ACTIVATION_FUNCTION_NONE:
    default:
      for (int i = 0; i < chans; i++) {
        for (int j = 0; j < rows; j++) {
          for (int k = 0; k < cols; k++) {
            res[i][j][k] = tensor[i][j][k];
          }
        }
      }
      break;
  }
  return res;
}

// Function to perform convolution on a 3D tensor with a set of filters
Tensor3D tensor_conv_3d(Tensor3D tensor, int t_chans, int t_rows, int t_cols, Tensor4D filters, int n_f, int f_chans, int f_rows, int f_cols, Tensor1D biases, int activation_function) {
  int res_chans = n_f;
  int res_rows = t_rows - f_rows + 1;
  int res_cols = t_cols - f_cols + 1;
  Tensor3D res = tensor_3d(res_chans, res_rows, res_cols);
  for (int on = 0; on < res_chans; on++) {
    for (int in = 0; in < t_chans; in++) {
      for (int oy = 0; oy < res_rows; oy++) {
        for (int ox = 0; ox < res_cols; ox++) {
          for (int wy = 0; wy < f_rows; wy++) {
            for (int wx = 0; wx < f_cols; wx++) {
              if (t_chans == f_chans) {
                // If the number of channels of the input tensor is equal to the number of channels of each filter, then apply one filter channel per input channel
                res[on][oy][ox] += tensor[in][oy + wy][ox + wx] * filters[on][in][wy][wx];
              } else {
                // Else, assume there is only one channel per filter and apply the first filter channel to all input channels 
                res[on][oy][ox] += tensor[in][oy + wy][ox + wx] * filters[on][0][wy][wx];
              }
            }
          }
        }
      }
    }
    // Add the bias for this output channel
    for (int oy = 0; oy < res_rows; oy++) {
      for (int ox = 0; ox < res_cols; ox++) {
        res[on][oy][ox] += biases[on];
      }
    }
  }
  res = tensor_apply_activation_3d(res, res_chans, res_rows, res_cols, activation_function);
  return res;
}

// Function to perform average pooling on a 3D tensor
Tensor3D tensor_avg_pooling_3d(Tensor3D tensor, int t_chans, int t_rows, int t_cols) {
  int res_chans = t_chans;
  int res_rows = t_rows / 2;
  int res_cols = t_cols / 2;
  Tensor3D res = tensor_3d(res_chans, res_rows, res_cols);
  for (int i = 0; i < res_chans; i++) {
    for (int j = 0; j < res_rows; j++) {
      for (int k = 0; k < res_cols; k++) {
        res[i][j][k] = (tensor[i][j * 2][k * 2] + tensor[i][j * 2 + 1][k * 2] + tensor[i][j * 2][k * 2 + 1] + tensor[i][j * 2 + 1][k * 2 + 1]) / 4;
      }
    }
  }
  return res;
}

// Function to flatten a 3D tensor into a 1D tensor
Tensor1D tensor_flatten_3d(Tensor3D tensor, int chans, int rows, int cols) {
  int res_cols = chans * rows * cols;
  Tensor1D res = tensor_1d(res_cols);
  int i = 0;
  for (int j = 0; j < chans; j++) {
    for (int k = 0; k < rows; k++) {
      for (int l = 0; l < cols; l++) {
        res[i] = tensor[j][k][l];
        i++;
      }
    }
  }
  return res;
}

// Function to calculate the index based on shape and strides
int calculate_index(int *shape, int *strides, int *indices, int dims) {
  int index = 0;
  for (int i = 0; i < dims; i++) {
    index += strides[i] * indices[i];
  }
  return index;
}

// Function to transpose a 1D tensor to a given shape permutation
Tensor1D tensor_transpose_perm_1d(Tensor1D tensor, int t_cols, int* input_shape, int* perm, int dims) {
  int res_cols = t_cols;
  Tensor1D res = tensor_1d(res_cols);
  int input_strides[dims];
  int output_strides[dims];
  int output_shape[dims];
  
  // Calculate strides for input array
  input_strides[dims - 1] = 1;
  for (int i = dims - 2; i >= 0; i--) {
    input_strides[i] = input_strides[i + 1] * input_shape[i + 1];
  }

  // Get the shape of the output array and calculate strides for the output array
  for (int i = 0; i < dims; i++) {
    output_shape[i] = input_shape[perm[i]];
  }

  output_strides[dims - 1] = 1;
  for (int i = dims - 2; i >= 0; i--) {
    output_strides[i] = output_strides[i + 1] * output_shape[i + 1];
  }

  // Iterate through each element of the input array
  int indices[dims];
  for (indices[0] = 0; indices[0] < input_shape[0]; indices[0]++) {
    for (indices[1] = 0; indices[1] < input_shape[1]; indices[1]++) {
      for (indices[2] = 0; indices[2] < input_shape[2]; indices[2]++) {
        for (indices[3] = 0; indices[3] < input_shape[3]; indices[3]++) {

          // Map the indices based on the permutation
          int new_indices[dims];
          for (int i = 0; i < dims; i++) {
              new_indices[i] = indices[perm[i]];
          }

          // Get the flat index of input and output based on the strides
          int input_idx = calculate_index(input_shape, input_strides, indices, dims);
          int output_idx = calculate_index(output_shape, output_strides, new_indices, dims);

          // Assign the transposed value
          res[output_idx] = tensor[input_idx];
        }
      }
    }
  }
  return res;
}

// Function to apply a dense layer to a 1D tensor
Tensor1D tensor_dense_1d(Tensor1D tensor, int t_cols, Tensor2D weights, int w_rows, int w_cols, Tensor1D bias, int activation_function) {
  if (t_cols != w_cols) {
    throw std::invalid_argument("Incompatible shapes for dense layer.");
  }
  Tensor1D res = tensor_1d(w_rows);
  for (int i = 0; i < w_rows; i++) {
    for (int j = 0; j < t_cols; j++) {
      res[i] += tensor[j] * weights[i][j];
    }
    res[i] += bias[i];
  }
  switch (activation_function) {
    case ACTIVATION_FUNCTION_TANH:
      for (int i = 0; i < w_rows; i++) {
        res[i] = std::tanh(res[i]);
      }
      break;
    case ACTIVATION_FUNCTION_RELU:
      for (int i = 0; i < w_rows; i++) {
        res[i] = std::max(0.0f, res[i]);
      }
      break;
    case ACTIVATION_FUNCTION_SOFTMAX: {
      Tensor1D temp = tensor_softmax_1d(res, w_rows);
      for (int i = 0; i < w_rows; i++) {
        res[i] = temp[i];
      }
      delete[] temp;
      break;
    }
    case ACTIVATION_FUNCTION_NONE:
    default:
      break;
  }
  return res;
}

// Function to deallocate a 1D tensor
void tensor_delete_1d(Tensor1D tensor) {
  delete[] tensor;
}

// Function to deallocate a 2D tensor
void tensor_delete_2d(Tensor2D tensor, int rows) {
  for (int i = 0; i < rows; ++i) {
    delete[] tensor[i];
  }
  delete[] tensor;
}

// Function to deallocate a 3D tensor
void tensor_delete_3d(Tensor3D tensor, int chans, int rows) {
  for (int i = 0; i < chans; ++i) {
    for (int j = 0; j < rows; ++j) {
      delete[] tensor[i][j];
    }
    delete[] tensor[i];
  }
  delete[] tensor;
}

// Function to deallocate a 4D tensor
void tensor_delete_4d(Tensor4D tensor, int n, int chans, int rows) {
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < chans; ++j) {
      for (int k = 0; k < rows; ++k) {
        delete[] tensor[i][j][k];
      }
      delete[] tensor[i][j];
    }
    delete[] tensor[i];
  }
  delete[] tensor;
}
