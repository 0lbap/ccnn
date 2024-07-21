#include <iostream>
#include <vector>

#define ACTIVATION_FUNCTION_NONE    0
#define ACTIVATION_FUNCTION_TANH    1
#define ACTIVATION_FUNCTION_RELU    2
#define ACTIVATION_FUNCTION_SOFTMAX 3

using Tensor1D = std::vector<float>;
using Tensor2D = std::vector<Tensor1D>;
using Tensor3D = std::vector<Tensor2D>;

void tensor_print_1d(Tensor1D tensor) {
  std::cout << std::endl;
  for (int i = 0; i < tensor.size(); i++) {
    std::cout << tensor[i] << " ";
  }
  std::cout << std::endl << std::endl;
}

void tensor_print_2d(Tensor2D tensor) {
  std::cout << std::endl;
  for (int i = 0; i < tensor.size(); i++) {
    for (int j = 0; j < tensor[0].size(); j++) {
      std::cout << tensor[i][j] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

void tensor_print_3d(Tensor3D tensor) {
  std::cout << std::endl;
  for (int i = 0; i < tensor.size(); i++) {
    for (int j = 0; j < tensor[0].size(); j++) {
      for (int k = 0; k < tensor[0][0].size(); k++) {
        std::cout << tensor[i][j][k] << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }
}

Tensor1D tensor_softmax_1d(Tensor1D tensor) {
  float max_val = *std::max_element(tensor.begin(), tensor.end());
  Tensor1D exp_values(tensor.size());
  float sum_exp = 0.0f;
  for (int i = 0; i < tensor.size(); i++) {
    exp_values[i] = std::exp(tensor[i] - max_val);
    sum_exp += exp_values[i];
  }
  for (int i = 0; i < tensor.size(); i++) {
    exp_values[i] /= sum_exp;
  }
  return exp_values;
}

Tensor2D tensor_transpose_2d(Tensor2D tensor) {
  int t_rows = tensor.size();
  int t_cols = tensor[0].size();
  int res_rows = t_cols;
  int res_cols = t_rows;
  Tensor2D res(res_rows, Tensor1D(res_cols, 0));
  for (int i = 0; i < t_rows; i++) {
    for (int j = 0; j < t_cols; j++) {
      res[j][i] = tensor[i][j];
    }
  }
  return res;
}

Tensor2D tensor_matmul_2d(Tensor2D m1, Tensor2D m2) {
  int m1_rows = m1.size();
  int m1_cols = m1[0].size();
  int m2_rows = m2.size();
  int m2_cols = m2[0].size();
  if (m1_cols != m2_rows) {
    throw std::invalid_argument("Incompatible matrices shapes for matmul.");
  }
  Tensor2D res(m1_rows, Tensor1D(m2_cols, 0));
  for (int i = 0; i < m1_rows; ++i) {
    for (int j = 0; j < m2_cols; ++j) {
      for (int k = 0; k < m1_cols; ++k) {
        res[i][j] += m1[i][k] * m2[k][j];
      }
    }
  }
  return res;
}

Tensor3D tensor_apply_activation_3d(Tensor3D tensor, int activation_function) {
  int t_chans = tensor.size();
  int t_rows = tensor[0].size();
  int t_cols = tensor[0][0].size();
  Tensor3D res = tensor;
  switch (activation_function) {
    case ACTIVATION_FUNCTION_TANH:
      for (int i = 0; i < t_chans; i++) {
        for (int j = 0; j < t_rows; j++) {
          for (int k = 0; k < t_cols; k++) {
            res[i][j][k] = std::tanh(tensor[i][j][k]);
          }
        }
      }
      break;
    case ACTIVATION_FUNCTION_RELU:
      for (int i = 0; i < t_chans; i++) {
        for (int j = 0; j < t_rows; j++) {
          for (int k = 0; k < t_cols; k++) {
            res[i][j][k] = std::max(0.0f, res[i][j][k]);
          }
        }
      }
      break;
    case ACTIVATION_FUNCTION_SOFTMAX:
      for (int i = 0; i < t_chans; i++) {
        for (int j = 0; j < t_rows; j++) {
          res[i][j] = tensor_softmax_1d(tensor[i][j]);
        }
      }
      break;
    case ACTIVATION_FUNCTION_NONE:
    default:
      break;
  }
  return res;
}

Tensor3D tensor_conv_3d(Tensor3D tensor, std::vector<Tensor3D> filters, int activation_function) {
  int n_f = filters.size();
  int f_chans = filters[0].size();
  int f_rows = filters[0][0].size();
  int f_cols = filters[0][0][0].size();
  int t_chans = tensor.size();
  int t_rows = tensor[0].size();
  int t_cols = tensor[0][0].size();
  int res_chans = n_f;
  int res_rows = t_rows - f_rows + 1;
  int res_cols = t_cols - f_cols + 1;
  Tensor3D res(res_chans, Tensor2D(res_rows, Tensor1D(res_cols, 0)));
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
  }
  res = tensor_apply_activation_3d(res, activation_function);
  return res;
}

Tensor3D tensor_avg_pooling_3d(Tensor3D tensor) {
  int t_chans = tensor.size();
  int t_rows = tensor[0].size();
  int t_cols = tensor[0][0].size();
  int res_chans = t_chans;
  int res_rows = t_rows / 2;
  int res_cols = t_cols / 2;
  Tensor3D res(res_chans, Tensor2D(res_rows, Tensor1D(res_cols, 0)));
  for (int i = 0; i < res_chans; i++) {
    for (int j = 0; j < res_rows; j++) {
      for (int k = 0; k < res_cols; k++) {
        res[i][j][k] = (tensor[i][j * 2][k * 2] + tensor[i][j * 2 + 1][k * 2] + tensor[i][j * 2][k * 2 + 1] + tensor[i][j * 2 + 1][k * 2 + 1]) / 4;
      }
    }
  }
  return res;
}

Tensor1D tensor_flatten_3d(Tensor3D tensor) {
  int t_chans = tensor.size();
  int t_rows = tensor[0].size();
  int t_cols = tensor[0][0].size();
  int res_cols = t_chans * t_rows * t_cols;
  Tensor1D res(res_cols, 0);
  int i = 0;
  for (int j = 0; j < t_chans; j++) {
    for (int k = 0; k < t_rows; k++) {
      for (int l = 0; l < t_cols; l++) {
        res[i] = tensor[j][k][l];
        i++;
      }
    }
  }
  return res;
}

Tensor1D tensor_dense_1d(Tensor1D tensor, Tensor2D weights, Tensor1D bias, int activation_function) {
  int t_cols = tensor.size();
  int res_cols = weights.size();
  Tensor1D res(res_cols, 0);
  for (int i = 0; i < res_cols; i++) {
    for (int j = 0; j < t_cols; j++) {
      res[i] += tensor[j] * weights[i][j];
    }
    res[i] += bias[i];
  }
  switch (activation_function) {
    case ACTIVATION_FUNCTION_TANH:
      for (int i = 0; i < res_cols; i++) {
        res[i] = std::tanh(res[i]);
      }
      break;
    case ACTIVATION_FUNCTION_RELU:
      for (int i = 0; i < res_cols; i++) {
        res[i] = std::max(0.0f, res[i]);
      }
      break;
    case ACTIVATION_FUNCTION_SOFTMAX:
      res = tensor_softmax_1d(res);
      break;
    case ACTIVATION_FUNCTION_NONE:
    default:
      break;
  }
  return res;
}
