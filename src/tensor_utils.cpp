#include <iostream>
#include <vector>

#define ACTIVATION_FUNCTION_TANH 0
#define ACTIVATION_FUNCTION_RELU 1
#define ACTIVATION_FUNCTION_SOFTMAX 2 // Not sure if this one works

void tensor_print_3d(std::vector< std::vector< std::vector<float> > > tensor) {
  std::cout << std::endl;
  for (int i = 0; i < tensor.size(); i++) {
    for (int j = 0; j < tensor[0].size(); j++) {
      for (int k = 0; k < tensor[0].size(); k++) {
        std::cout << tensor[i][j][k] << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }
}

std::vector< std::vector< std::vector<float> > > tensor_conv_3d(std::vector< std::vector< std::vector<float> > > tensor, std::vector< std::vector< std::vector< std::vector<float> > > > filters) {
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
  std::vector< std::vector< std::vector<float> > > res(res_chans, std::vector< std::vector<float> >(res_rows, std::vector<float>(res_cols, 0)));
  for (int on = 0; on < res_chans; on++) {
    for (int in = 0; in < t_chans; in++) {
      for (int oy = 0; oy < res_rows; oy++) {
        for (int ox = 0; ox < res_cols; ox++) {
          for (int wy = 0; wy < f_rows; wy++) {
            for (int wx = 0; wx < f_cols; wx++) {
              res[on][oy][ox] += tensor[in][oy + wy][ox + wx] * filters[on][in][wy][wx];
            }
          }
        }
      }
    }
  }
  return res;
}

std::vector< std::vector< std::vector<float> > > tensor_avg_pooling_3d(std::vector< std::vector< std::vector<float> > > tensor) {
  int t_chans = tensor.size();
  int t_rows = tensor[0].size();
  int t_cols = tensor[0][0].size();
  int res_chans = t_chans;
  int res_rows = t_rows / 2;
  int res_cols = t_cols / 2;
  std::vector< std::vector< std::vector<float> > > res(res_chans, std::vector< std::vector<float> >(res_rows, std::vector<float>(res_cols, 0)));
  for (int i = 0; i < res_chans; i++) {
    for (int j = 0; j < res_rows; j++) {
      for (int k = 0; k < res_cols; k++) {
        res[i][j][k] = (tensor[i][j * 2][k * 2] + tensor[i][j * 2 + 1][k * 2] + tensor[i][j * 2][k * 2 + 1] + tensor[i][j * 2 + 1][k * 2 + 1]) / 4;
      }
    }
  }
  return res;
}

std::vector< std::vector< std::vector<float> > > tensor_apply_activation(std::vector< std::vector< std::vector<float> > > tensor, int activation_function) {
  int t_chans = tensor.size();
  int t_rows = tensor[0].size();
  int t_cols = tensor[0][0].size();
  std::vector< std::vector< std::vector<float> > > res = tensor;
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
            if (tensor[i][j][k] <= 0) {
              res[i][j][k] = 0;
            } else {
              res[i][j][k] = tensor[i][j][k];
            }
          }
        }
      }
      break;
    case ACTIVATION_FUNCTION_SOFTMAX:
      for (int i = 0; i < t_chans; i++) {
        for (int j = 0; j < t_rows; j++) {
          float maxVal = *max_element(tensor[i][j].begin(), tensor[i][j].end());
          float sumExp = 0.0f;
          for (int k = 0; k < t_cols; k++) {
            res[i][j][k] = exp(tensor[i][j][k] - maxVal);
            sumExp += res[i][j][k];
          }
          for (int k = 0; k < res[i][j].size(); k++) {
            res[i][j][k] /= sumExp;
          }
        }
      }
      break;
  }
  return res;
}
