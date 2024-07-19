#include <vector>
#include "../tensor_utils.cpp"

void test_model() {
  // Set up input tensor
  int t1_chans = 1;
  int t1_rows = 8;
  int t1_cols = 8;
  std::vector< std::vector< std::vector<float> > > t1(t1_chans, std::vector< std::vector<float> >(t1_rows, std::vector<float>(t1_cols, 0)));

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
  std::vector< std::vector< std::vector< std::vector<float> > > > f1(n_f1, std::vector< std::vector< std::vector<float> > >(f1_chans, std::vector< std::vector<float> >(f1_rows, std::vector<float>(f1_cols, 0))));
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
  std::vector< std::vector< std::vector<float> > > t2 = tensor_conv_3d(t1, f1); // apply convolution
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

  // Transpose
  std::cout << "Before transpose:" << std::endl;
  std::vector< std::vector<float> > tt(3, std::vector<float>(2, 0));
  cpt = 1;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 2; j++) {
      tt[i][j] = cpt;
      cpt++;
    } 
  }
  tensor_print_2d(tt);
  std::cout << "Processing transpose..." << std::endl;
  std::vector< std::vector<float> > tt2 = tensor_transpose_2d(tt);
  std::cout << "After transpose:" << std::endl;
  tensor_print_2d(tt2);

}
