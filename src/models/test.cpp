#include <vector>
#include "../tensor_utils.cpp"

void test_model() {
  // Set up input tensor
  int t1_chans = 1;
  int t1_rows = 8;
  int t1_cols = 8;
  Tensor3D t1(t1_chans, Tensor2D(t1_rows, Tensor1D(t1_cols, 0)));

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
  std::vector<Tensor3D> f1(n_f1, Tensor3D(f1_chans, Tensor2D(f1_rows, Tensor1D(f1_cols, 0))));
  f1 = {
    {
      {
        {1, 0, 1},
        {0, 1, 0},
        {1, 0, 1}
      }
    }, {
      {
        {1, 1, 1},
        {1, 1, 1},
        {1, 1, 1}
      }
    }
  };

  std::cout << "------------------------------------------------------------" << std::endl;

  // Conv2D
  std::cout << "Layer 1: Conv2D" << std::endl;
  std::cout << "Input tensor: " << t1[0].size() << "x" << t1[0][0].size() << "x" << t1.size() << std::endl;
  tensor_print_3d(t1);
  std::cout << "Kernel filters (" << f1.size() << "): " << f1[0][0].size() << "x" << f1[0][0][0].size() << "x" << f1[0].size() << std::endl;
  for (int i = 0; i < n_f1; i++) {
    std::cout << "Filter " << i + 1 << ":" << std::endl;
    tensor_print_3d(f1[i]);
  }
  std::cout << "Processing Conv2D..." << std::endl;
  Tensor3D t2 = tensor_conv_3d(t1, f1, ACTIVATION_FUNCTION_RELU); // apply convolution
  std::cout << "Output tensor: " << t2[0].size() << "x" << t2[0][0].size() << "x" << t2.size()  << std::endl;
  tensor_print_3d(t2);

  std::cout << "------------------------------------------------------------" << std::endl;

  // AveragePooling2D
  std::cout << "Layer 2: AveragePooling2D" << std::endl;
  std::cout << "Input tensor: " << t2[0].size() << "x" << t2[0][0].size() << "x" << t2.size() << std::endl;
  tensor_print_3d(t2);
  std::cout << "Processing AveragePooling2D..." << std::endl;
  Tensor3D t3 = tensor_avg_pooling_3d(t2);
  std::cout << "Output tensor: " << t3[0].size() << "x" << t3[0][0].size() << "x" << t3.size()  << std::endl;
  tensor_print_3d(t3);

  std::cout << "------------------------------------------------------------" << std::endl;

  // Transpose
  std::cout << "Before transpose:" << std::endl;
  Tensor2D tt1(3, Tensor1D(2, 0));
  cpt = 1;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 2; j++) {
      tt1[i][j] = cpt;
      cpt++;
    }
  }
  tensor_print_2d(tt1);
  std::cout << "Processing transpose..." << std::endl;
  Tensor2D tt2 = tensor_transpose_2d(tt1);
  std::cout << "After transpose:" << std::endl;
  tensor_print_2d(tt2);

  std::cout << "------------------------------------------------------------" << std::endl;

  std::cout << "Before matmul:" << std::endl;
  Tensor2D tt3(3, Tensor1D(2, 0));
  cpt = 1;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 2; j++) {
      tt3[i][j] = cpt;
      cpt++;
    }
  }
  Tensor2D tt4(2, Tensor1D(3, 0));
  cpt = 1;
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 3; j++) {
      tt4[i][j] = cpt;
      cpt++;
    }
  }
  tensor_print_2d(tt3);
  tensor_print_2d(tt4);
  std::cout << "Processing matmul..." << std::endl;
  Tensor2D tt5 = tensor_matmul_2d(tt3, tt4);
  std::cout << "After matmul:" << std::endl;
  tensor_print_2d(tt5);

  std::cout << "------------------------------------------------------------" << std::endl;

  std::cout << "Before matmul:" << std::endl;
  Tensor2D tt6(1, Tensor1D(3, 0));
  cpt = 1;
  for (int i = 0; i < 3; i++) {
    tt6[0][i] = cpt;
    cpt++;
  }
  Tensor2D tt7(3, Tensor1D(3, 0));
  cpt = 1;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      tt7[i][j] = cpt;
      cpt++;
    }
  }
  tensor_print_2d(tt6);
  tensor_print_2d(tt7);
  std::cout << "Processing matmul..." << std::endl;
  Tensor2D tt8 = tensor_matmul_2d(tt6, tt7);
  std::cout << "After matmul:" << std::endl;
  tensor_print_2d(tt8);

  std::cout << "------------------------------------------------------------" << std::endl;

  std::cout << "Before matmul:" << std::endl;
  Tensor2D tt9(3, Tensor1D(3, 0));
  cpt = 1;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      tt9[i][j] = cpt;
      cpt++;
    }
  }
  Tensor2D tt10(3, Tensor1D(1, 0));
  cpt = 1;
  for (int i = 0; i < 3; i++) {
    tt10[i][0] = cpt;
    cpt++;
  }
  tensor_print_2d(tt9);
  tensor_print_2d(tt10);
  std::cout << "Processing matmul..." << std::endl;
  Tensor2D tt11 = tensor_matmul_2d(tt9, tt10);
  std::cout << "After matmul:" << std::endl;
  tensor_print_2d(tt11);

  std::cout << "------------------------------------------------------------" << std::endl;
  
  std::cout << "Before dense:" << std::endl;
  Tensor1D input1(3, 0);
  input1 = {1.0, 2.0, 3.0};
  Tensor1D bias1(2, 0);
  bias1 = {0.1, 0.2};
  Tensor2D weights1(2, Tensor1D(3, 0));
  weights1 = {
    {0.1, -0.2, -0.3},
    {-0.4, -0.5, 0.6}
  };
  std::cout << "Input:" << std::endl;
  tensor_print_1d(input1);
  std::cout << "Bias:" << std::endl;
  tensor_print_1d(bias1);
  std::cout << "Weights:" << std::endl;
  tensor_print_2d(weights1);
  Tensor1D output1 = tensor_dense_1d(input1, weights1, bias1, ACTIVATION_FUNCTION_RELU);
  std::cout << "After dense:" << std::endl;
  tensor_print_1d(output1);
}
