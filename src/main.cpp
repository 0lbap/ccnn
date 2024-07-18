#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <ostream>
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

std::vector< std::vector< std::vector<float> > > tensor_conv_2d(std::vector< std::vector<float> > tensor, std::vector< std::vector< std::vector<float> > > filter) {
  int f_chans = filter.size();
  int f_rows = filter[0].size();
  int f_cols = filter[0][0].size();
  int t_rows = tensor.size();
  int t_cols = tensor[0].size();
  int res_chans = f_chans;
  int res_rows = t_rows - f_rows + 1;
  int res_cols = t_cols - f_cols + 1;
  std::vector< std::vector< std::vector<float> > > res(res_chans, std::vector< std::vector<float> >(res_rows, std::vector<float>(res_cols, 0)));
  for (int i = 0; i < res_chans; i++) {
    for (int j = 0; j < res_rows; j++) {
      for (int k = 0; k < res_cols; k++) {
        float acc = 0;
        for (int l = 0; l < f_rows; l++) {
          for (int m = 0; m < f_cols; m++) {
            acc += tensor[j + l][k + m] * filter[i][l][m];
          }
        }
        res[i][j][k] = acc;
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
              std::cout << "aled" << std::endl;
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

int main(int, char **) {
  std::cout << "Welcome to CCNN" << std::endl;

  // Set up input tensor
  int t1_chans = 1;
  int t1_rows = 8;
  int t1_cols = 8;
  std::vector< std::vector< std::vector<float> > > t1(t1_chans, std::vector< std::vector<float> >(t1_rows, std::vector<float>(t1_cols, 1)));

  float cpt = 0;
  for (int i = 0; i < t1_rows; i++) {
    for (int j = 0; j < t1_cols; j++) {
      t1[0][i][j] = cpt;
      cpt++;
    }
  }

  // Set up filters
  int f1_chans = 2;
  int f1_rows = 3;
  int f1_cols = 3;
  std::vector< std::vector< std::vector<float> > > f1(f1_chans, std::vector< std::vector<float> >(f1_rows, std::vector<float>(f1_cols, 0)));
  f1[0] = {
    {1, 0, 1},
    {0, 1, 0},
    {1, 0, 1}
  };
  f1[1] = {
    {1, 1, 1},
    {1, 1, 1},
    {1, 1, 1}
  };

  std::cout << "------------------------------------------------------------" << std::endl;

  // Conv2D
  std::cout << "Layer 1: Conv2D" << std::endl;
  std::cout << "Input tensor: " << t1[0].size() << "x" << t1[0][0].size() << "x" << t1.size() << std::endl;
  tensor_print_3d(t1);
  std::cout << "Kernel filters: " << f1[0].size() << "x" << f1[0][0].size() << "x" << f1.size() << std::endl;
  tensor_print_3d(f1);
  std::cout << "Processing Conv2D..." << std::endl;
  std::vector< std::vector< std::vector<float> > > t2 = tensor_conv_2d(t1[0], f1); // apply convolution on first (and only) input data (since t1 depth is 1)
  t2 = tensor_apply_activation(t2, ACTIVATION_FUNCTION_RELU); // apply activation (ReLU)
  std::cout << "Output tensor: " << t2[0].size() << "x" << t2[0][0].size() << "x" << t2.size()  << std::endl;
  tensor_print_3d(t2);

  std::cout << "------------------------------------------------------------" << std::endl;

  // AveragePooling2D
  std::cout << "Layer 2: AveragePooling2D" << std::endl;
  std::cout << "Input tensor: " << t2[0].size() << "x" << t2[0][0].size() << "x" << t2.size() << std::endl;
  tensor_print_3d(t2);
  std::cout << "Processing AveragePooling2D..." << std::endl;
  std::vector< std::vector< std::vector<float> > > t3 = tensor_avg_pooling_3d(t2);
  std::cout << "Output tensor: " << t3[0].size() << "x" << t3[0][0].size() << "x" << t3.size()  << std::endl;
  tensor_print_3d(t3);

  std::cout << "------------------------------------------------------------" << std::endl;

  std::cout << "Done." << std::endl;

  return EXIT_SUCCESS;
}
