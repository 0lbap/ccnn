#include <iostream>
#include <vector>
#include "../tensor_utils.hpp"

void run_model(int batch_size, std::vector<int> profile_indices) {
  // Set up input tensor
  int t1_chans = 1;
  int t1_rows = 8;
  int t1_cols = 8;
  Tensor3D t1 = tensor_3d(t1_chans, t1_rows, t1_cols);

  float cpt = 0;
  for (int i = 0; i < t1_rows; i++) {
    for (int j = 0; j < t1_cols; j++) {
      t1[0][i][j] = cpt;
      cpt++;
    }
  }

  // Set up 2 filters of size 3x3x1 (for the first convolution)
  int n_f1 = 2;
  int f1_chans = 1;
  int f1_rows = 3;
  int f1_cols = 3;
  float f1_vals[] = {
    1, 0, 1,
    0, 1, 0,
    1, 0, 1,

    1, 1, 1,
    1, 1, 1,
    1, 1, 1
  };
  Tensor4D f1 = tensor_init_4d(n_f1, f1_chans, f1_rows, f1_cols, f1_vals);
  
  // Set up 2 biases (for the first convolution)
  int b1_cols = 2;
  float b1_vals[] = {0, 0};
  Tensor1D b1 = tensor_init_1d(b1_cols, b1_vals);

  // Conv2D
  int t2_chans = 2;
  int t2_rows = 6;
  int t2_cols = 6;
  Tensor3D t2 = tensor_conv_3d(t1, t1_chans, t1_rows, t1_cols, f1, n_f1, f1_chans, f1_rows, f1_cols, b1, ACTIVATION_FUNCTION_RELU); // apply convolution
  tensor_delete_3d(t1, t1_chans, t1_rows);

  // AveragePooling2D
  int t3_chans = 2;
  int t3_rows = 3;
  int t3_cols = 3;
  Tensor3D t3 = tensor_avg_pooling_3d(t2, t2_chans, t2_rows, t2_cols);
  tensor_delete_3d(t2, t2_chans, t2_rows);
  tensor_delete_3d(t3, t3_chans, t3_rows);

  // Transpose
  int tt1_rows = 3;
  int tt1_cols = 2;
  Tensor2D tt1 = tensor_2d(tt1_rows, tt1_cols);
  cpt = 1;
  for (int i = 0; i < tt1_rows; i++) {
    for (int j = 0; j < tt1_cols; j++) {
      tt1[i][j] = cpt;
      cpt++;
    }
  }
  int tt2_rows = tt1_cols;
  int tt2_cols = tt1_rows;
  Tensor2D tt2 = tensor_transpose_2d(tt1, tt1_rows, tt1_cols);
  tensor_delete_2d(tt1, tt1_rows);
  tensor_delete_2d(tt2, tt2_rows);

  // Matmul
  int tt3_rows = 3;
  int tt3_cols = 2;
  Tensor2D tt3 = tensor_2d(tt3_rows, tt3_cols);
  cpt = 1;
  for (int i = 0; i < tt3_rows; i++) {
    for (int j = 0; j < tt3_cols; j++) {
      tt3[i][j] = cpt;
      cpt++;
    }
  }
  int tt4_rows = 2;
  int tt4_cols = 3;
  Tensor2D tt4 = tensor_2d(tt4_rows, tt4_cols);
  cpt = 1;
  for (int i = 0; i < tt4_rows; i++) {
    for (int j = 0; j < tt4_cols; j++) {
      tt4[i][j] = cpt;
      cpt++;
    }
  }
  int tt5_rows = tt3_rows;
  int tt5_cols = tt4_cols;
  Tensor2D tt5 = tensor_matmul_2d(tt3, tt3_rows, tt3_cols, tt4, tt4_rows, tt4_cols);
  tensor_delete_2d(tt3, tt3_rows);
  tensor_delete_2d(tt4, tt4_rows);
  tensor_delete_2d(tt5, tt5_rows);

  // Matmul
  int tt6_rows = 1;
  int tt6_cols = 3;
  Tensor2D tt6 = tensor_2d(tt6_rows, tt6_cols);
  cpt = 1;
  for (int i = 0; i < tt6_cols; i++) {
    tt6[0][i] = cpt;
    cpt++;
  }
  int tt7_rows = 3;
  int tt7_cols = 3;
  Tensor2D tt7 = tensor_2d(tt7_rows, tt7_cols);
  cpt = 1;
  for (int i = 0; i < tt7_rows; i++) {
    for (int j = 0; j < tt7_cols; j++) {
      tt7[i][j] = cpt;
      cpt++;
    }
  }
  int tt8_rows = tt6_rows;
  int tt8_cols = tt7_cols;
  Tensor2D tt8 = tensor_matmul_2d(tt6, tt6_rows, tt6_cols, tt7, tt7_rows, tt7_cols);
  tensor_delete_2d(tt6, tt6_rows);
  tensor_delete_2d(tt7, tt7_rows);
  tensor_delete_2d(tt8, tt8_rows);

  // Matmul
  int tt9_rows = 3;
  int tt9_cols = 3;
  Tensor2D tt9 = tensor_2d(tt9_rows, tt9_cols);
  cpt = 1;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      tt9[i][j] = cpt;
      cpt++;
    }
  }
  int tt10_rows = 3;
  int tt10_cols = 1;
  Tensor2D tt10 = tensor_2d(tt10_rows, tt10_cols);
  cpt = 1;
  for (int i = 0; i < 3; i++) {
    tt10[i][0] = cpt;
    cpt++;
  }
  int tt11_rows = tt9_rows;
  int tt11_cols = tt10_cols;
  Tensor2D tt11 = tensor_matmul_2d(tt9, tt9_rows, tt9_cols, tt10, tt10_rows, tt10_cols);
  tensor_delete_2d(tt9, tt9_rows);
  tensor_delete_2d(tt10, tt10_rows);
  tensor_delete_2d(tt11, tt11_rows);
  
  // Dense
  int input1_cols = 3;
  float input_vals[] = {1.0, 2.0, 3.0};
  Tensor1D input1 = tensor_init_1d(input1_cols, input_vals);
  int bias1_cols = 2;
  float bias1_vals[] = {0.0, 0.0};
  Tensor1D bias1 = tensor_init_1d(bias1_cols, bias1_vals);
  int weights1_rows = 2;
  int weights1_cols = 3;
  float weights1_vals[] = {
    0.1, 0.1, 0.1,
    0.1, 0.1, 0.1
  };
  Tensor2D weights1 = tensor_init_2d(weights1_rows, weights1_cols, weights1_vals);
  int output1_cols = weights1_rows;
  Tensor1D output1 = tensor_dense_1d(input1, input1_cols, weights1, weights1_rows, weights1_cols, bias1, ACTIVATION_FUNCTION_SOFTMAX);
  tensor_delete_1d(input1);
  tensor_delete_1d(bias1);
  tensor_delete_2d(weights1, weights1_rows);
}

void run_model_debug(int batch_size) {
  // Set up input tensor
  int t1_chans = 1;
  int t1_rows = 8;
  int t1_cols = 8;
  Tensor3D t1 = tensor_3d(t1_chans, t1_rows, t1_cols);

  float cpt = 0;
  for (int i = 0; i < t1_rows; i++) {
    for (int j = 0; j < t1_cols; j++) {
      t1[0][i][j] = cpt;
      cpt++;
    }
  }

  // Set up 2 filters of size 3x3x1 (for the first convolution)
  int n_f1 = 2;
  int f1_chans = 1;
  int f1_rows = 3;
  int f1_cols = 3;
  float f1_vals[] = {
    1, 0, 1,
    0, 1, 0,
    1, 0, 1,

    1, 1, 1,
    1, 1, 1,
    1, 1, 1
  };
  Tensor4D f1 = tensor_init_4d(n_f1, f1_chans, f1_rows, f1_cols, f1_vals);
  
  // Set up 2 biases (for the first convolution)
  int b1_cols = 2;
  float b1_vals[] = {0, 0};
  Tensor1D b1 = tensor_init_1d(b1_cols, b1_vals);

  std::cout << "------------------------------------------------------------" << std::endl;

  // Conv2D
  std::cout << "Layer 1: Conv2D" << std::endl;
  std::cout << "Input tensor: " << t1_rows << "x" << t1_cols << "x" << t1_chans << std::endl;
  tensor_print_3d(t1, t1_chans, t1_rows, t1_cols);
  std::cout << "Kernel filters (" << n_f1 << "): " << f1_rows << "x" << f1_cols << "x" << f1_chans << std::endl;
  for (int i = 0; i < n_f1; i++) {
    std::cout << "Filter " << i + 1 << ":" << std::endl;
    tensor_print_3d(f1[i], f1_chans, f1_rows, f1_cols);
  }
  std::cout << "Processing Conv2D..." << std::endl;
  int t2_chans = 2;
  int t2_rows = 6;
  int t2_cols = 6;
  Tensor3D t2 = tensor_conv_3d(t1, t1_chans, t1_rows, t1_cols, f1, n_f1, f1_chans, f1_rows, f1_cols, b1, ACTIVATION_FUNCTION_RELU); // apply convolution
  std::cout << "Output tensor: " << t2_rows << "x" << t2_cols << "x" << t2_chans << std::endl;
  tensor_print_3d(t2, t2_chans, t2_rows, t2_cols);
  tensor_delete_3d(t1, t1_chans, t1_rows);

  std::cout << "------------------------------------------------------------" << std::endl;

  // AveragePooling2D
  std::cout << "Layer 2: AveragePooling2D" << std::endl;
  std::cout << "Input tensor: " << t2_rows << "x" << t2_cols << "x" << t2_chans << std::endl;
  tensor_print_3d(t2, t2_chans, t2_rows, t2_cols);
  std::cout << "Processing AveragePooling2D..." << std::endl;
  int t3_chans = 2;
  int t3_rows = 3;
  int t3_cols = 3;
  Tensor3D t3 = tensor_avg_pooling_3d(t2, t2_chans, t2_rows, t2_cols);
  std::cout << "Output tensor: " << t3_rows << "x" << t3_cols << "x" << t3_chans  << std::endl;
  tensor_print_3d(t3, t3_chans, t3_rows, t3_cols);
  tensor_delete_3d(t2, t2_chans, t2_rows);
  tensor_delete_3d(t3, t3_chans, t3_rows);

  std::cout << "------------------------------------------------------------" << std::endl;

  // Transpose
  std::cout << "Before transpose:" << std::endl;
  int tt1_rows = 3;
  int tt1_cols = 2;
  Tensor2D tt1 = tensor_2d(tt1_rows, tt1_cols);
  cpt = 1;
  for (int i = 0; i < tt1_rows; i++) {
    for (int j = 0; j < tt1_cols; j++) {
      tt1[i][j] = cpt;
      cpt++;
    }
  }
  tensor_print_2d(tt1, tt1_rows, tt1_cols);
  std::cout << "Processing transpose..." << std::endl;
  int tt2_rows = tt1_cols;
  int tt2_cols = tt1_rows;
  Tensor2D tt2 = tensor_transpose_2d(tt1, tt1_rows, tt1_cols);
  std::cout << "After transpose:" << std::endl;
  tensor_print_2d(tt2, tt2_rows, tt2_cols);
  tensor_delete_2d(tt1, tt1_rows);
  tensor_delete_2d(tt2, tt2_rows);

  std::cout << "------------------------------------------------------------" << std::endl;

  std::cout << "Before matmul:" << std::endl;
  int tt3_rows = 3;
  int tt3_cols = 2;
  Tensor2D tt3 = tensor_2d(tt3_rows, tt3_cols);
  cpt = 1;
  for (int i = 0; i < tt3_rows; i++) {
    for (int j = 0; j < tt3_cols; j++) {
      tt3[i][j] = cpt;
      cpt++;
    }
  }
  int tt4_rows = 2;
  int tt4_cols = 3;
  Tensor2D tt4 = tensor_2d(tt4_rows, tt4_cols);
  cpt = 1;
  for (int i = 0; i < tt4_rows; i++) {
    for (int j = 0; j < tt4_cols; j++) {
      tt4[i][j] = cpt;
      cpt++;
    }
  }
  tensor_print_2d(tt3, tt3_rows, tt3_cols);
  tensor_print_2d(tt4, tt4_rows, tt4_cols);
  std::cout << "Processing matmul..." << std::endl;
  int tt5_rows = tt3_rows;
  int tt5_cols = tt4_cols;
  Tensor2D tt5 = tensor_matmul_2d(tt3, tt3_rows, tt3_cols, tt4, tt4_rows, tt4_cols);
  std::cout << "After matmul:" << std::endl;
  tensor_print_2d(tt5, tt5_rows, tt5_cols);
  tensor_delete_2d(tt3, tt3_rows);
  tensor_delete_2d(tt4, tt4_rows);
  tensor_delete_2d(tt5, tt5_rows);

  std::cout << "------------------------------------------------------------" << std::endl;

  std::cout << "Before matmul:" << std::endl;
  int tt6_rows = 1;
  int tt6_cols = 3;
  Tensor2D tt6 = tensor_2d(tt6_rows, tt6_cols);
  cpt = 1;
  for (int i = 0; i < tt6_cols; i++) {
    tt6[0][i] = cpt;
    cpt++;
  }
  int tt7_rows = 3;
  int tt7_cols = 3;
  Tensor2D tt7 = tensor_2d(tt7_rows, tt7_cols);
  cpt = 1;
  for (int i = 0; i < tt7_rows; i++) {
    for (int j = 0; j < tt7_cols; j++) {
      tt7[i][j] = cpt;
      cpt++;
    }
  }
  tensor_print_2d(tt6, tt6_rows, tt6_cols);
  tensor_print_2d(tt7, tt7_rows, tt7_cols);
  std::cout << "Processing matmul..." << std::endl;
  int tt8_rows = tt6_rows;
  int tt8_cols = tt7_cols;
  Tensor2D tt8 = tensor_matmul_2d(tt6, tt6_rows, tt6_cols, tt7, tt7_rows, tt7_cols);
  std::cout << "After matmul:" << std::endl;
  tensor_print_2d(tt8, tt8_rows, tt8_cols);
  tensor_delete_2d(tt6, tt6_rows);
  tensor_delete_2d(tt7, tt7_rows);
  tensor_delete_2d(tt8, tt8_rows);

  std::cout << "------------------------------------------------------------" << std::endl;

  std::cout << "Before matmul:" << std::endl;
  int tt9_rows = 3;
  int tt9_cols = 3;
  Tensor2D tt9 = tensor_2d(tt9_rows, tt9_cols);
  cpt = 1;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      tt9[i][j] = cpt;
      cpt++;
    }
  }
  int tt10_rows = 3;
  int tt10_cols = 1;
  Tensor2D tt10 = tensor_2d(tt10_rows, tt10_cols);
  cpt = 1;
  for (int i = 0; i < 3; i++) {
    tt10[i][0] = cpt;
    cpt++;
  }
  tensor_print_2d(tt9, tt9_rows, tt9_cols);
  tensor_print_2d(tt10, tt10_rows, tt10_cols);
  std::cout << "Processing matmul..." << std::endl;
  int tt11_rows = tt9_rows;
  int tt11_cols = tt10_cols;
  Tensor2D tt11 = tensor_matmul_2d(tt9, tt9_rows, tt9_cols, tt10, tt10_rows, tt10_cols);
  std::cout << "After matmul:" << std::endl;
  tensor_print_2d(tt11, tt11_rows, tt11_cols);
  tensor_delete_2d(tt9, tt9_rows);
  tensor_delete_2d(tt10, tt10_rows);
  tensor_delete_2d(tt11, tt11_rows);

  std::cout << "------------------------------------------------------------" << std::endl;
  
  std::cout << "Before dense:" << std::endl;
  int input1_cols = 3;
  float input_vals[] = {1.0, 2.0, 3.0};
  Tensor1D input1 = tensor_init_1d(input1_cols, input_vals);
  int bias1_cols = 2;
  float bias1_vals[] = {0.0, 0.0};
  Tensor1D bias1 = tensor_init_1d(bias1_cols, bias1_vals);
  int weights1_rows = 2;
  int weights1_cols = 3;
  float weights1_vals[] = {
    0.1, 0.1, 0.1,
    0.1, 0.1, 0.1
  };
  Tensor2D weights1 = tensor_init_2d(weights1_rows, weights1_cols, weights1_vals);
  std::cout << "Input:" << std::endl;
  tensor_print_1d(input1, input1_cols);
  std::cout << "Bias:" << std::endl;
  tensor_print_1d(bias1, bias1_cols);
  std::cout << "Weights:" << std::endl;
  tensor_print_2d(weights1, weights1_rows, weights1_cols);
  int output1_cols = weights1_rows;
  Tensor1D output1 = tensor_dense_1d(input1, input1_cols, weights1, weights1_rows, weights1_cols, bias1, ACTIVATION_FUNCTION_SOFTMAX);
  std::cout << "After dense:" << std::endl;
  tensor_print_1d(output1, output1_cols);
  tensor_delete_1d(input1);
  tensor_delete_1d(bias1);
  tensor_delete_2d(weights1, weights1_rows);
}
