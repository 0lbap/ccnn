#include <vector>
#include "../tensor_utils.cpp"

void lenet_model() {
  // Set up input tensor
  int t1_chans = 1;
  int t1_rows = 32;
  int t1_cols = 32;
  std::vector< std::vector< std::vector<float> > > t1(t1_chans, std::vector< std::vector<float> >(t1_rows, std::vector<float>(t1_cols, 0)));

  // TODO: set t1 values

  // Set up 6 filters of size 5x5x1 (for the first convolution)
  int n_f1 = 6;
  int f1_chans = 1;
  int f1_rows = 5;
  int f1_cols = 5;
  std::vector< std::vector< std::vector< std::vector<float> > > > f1(n_f1, std::vector< std::vector< std::vector<float> > >(f1_chans, std::vector< std::vector<float> >(f1_rows, std::vector<float>(f1_cols, 0))));
  
  // TODO: set f1 values

  // Set up 16 filters of size 5x5x1 (for the second convolution)
  int n_f2 = 16;
  int f2_chans = 1;
  int f2_rows = 5;
  int f2_cols = 5;
  std::vector< std::vector< std::vector< std::vector<float> > > > f2(n_f2, std::vector< std::vector< std::vector<float> > >(f2_chans, std::vector< std::vector<float> >(f2_rows, std::vector<float>(f2_cols, 0))));
  
  // TODO: set f2 values

  std::cout << "------------------------------------------------------------" << std::endl;

  // TODO: Conv2D
  
  // TODO: AveragePooling2D

  // TODO: Conv2D
  
  // TODO: AveragePooling2D

  // TODO: Flatten

  // TODO: Dense

  // TODO: Dense

  // TODO: Dense
}
