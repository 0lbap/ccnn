#include <vector>
#include <iostream>
#include "../tensor_utils.hpp"

void run_model(int batch_size) {
  // Set up one input tensor
  int t1_chans = 1;
  int t1_rows = 28;
  int t1_cols = 28;
  Tensor3D t1(t1_chans, Tensor2D(t1_rows, Tensor1D(t1_cols, 0)));
  t1 = {
    {
      {  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},
      {  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},
      {  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},
      {  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},
      {  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},
      {  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   8,  38, 137, 146, 232, 254, 255, 255, 197, 109,   0,   0,   0,   0,   0,   0,   0,   0},
      {  0,   0,   0,   0,   0,   0,   0,   0,   0,  87, 107, 253, 253, 253, 253, 253, 253, 253, 253, 188,   0,   0,   0,   0,   0,   0,   0,   0},
      {  0,   0,   0,   0,   0,   0,   0,   0, 120, 237, 253, 253, 253, 248, 209, 139, 139, 230, 253, 188,   0,   0,   0,   0,   0,   0,   0,   0},
      {  0,   0,   0,   0,   0,   0,   0,   0, 112, 229, 210, 128,  96,   0,   0,   0,   0, 117, 253, 188,   0,   0,   0,   0,   0,   0,   0,   0},
      {  0,   0,   0,   0,   0,   0,   0,   0,   0, 28,   0,   0,   0,   0,   0,   0,   45, 241, 245,  82,   0,   0,   0,   0,   0,   0,   0,   0},
      {  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  60, 253, 125,   0,   0,   0,   0,   0,   0,   0,   0,   0},
      {  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  32, 134, 148,  98, 127,   4,   0,   0,   0,   0,   0,   0,   0,   0,   0},
      {  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   7, 201, 253, 200,  59,  12,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},
      {  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 122, 246, 253, 223,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},
      {  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 233, 253, 253, 236,  55,  11,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},
      {  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 233, 253, 253, 253, 253, 210,  48,   0,   0,   0,   0,   0,   0,   0,   0,   0},
      {  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  68, 193, 185, 243, 253, 253, 173,   0,   0,   0,   0,   0,   0,   0,   0,   0},
      {  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  59, 253, 253, 226,   0,   0,   0,   0,   0,   0,   0,   0,   0},
      {  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  22, 253, 253, 226,   0,   0,   0,   0,   0,   0,   0,   0,   0},
      {  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  10,  93, 253, 253, 123,   0,   0,   0,   0,   0,   0,   0,   0,   0},
      {  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  14, 180, 253, 253, 228,  42,   0,   0,   0,   0,   0,   0,   0,   0,   0},
      {  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  95, 181, 253, 253, 253,  66,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},
      {  0,   0,   0,   0,   0,   0,   0,   6,  33,  33, 100, 178, 253, 253, 253, 241,  88,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},
      {  0,   0,   0,   0,   0,   0,   0,  32, 236, 253, 253, 253, 253, 253, 228, 122,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},
      {  0,   0,   0,   0,   0,   0,   0,   0, 143, 253, 253, 253, 154,  76,  24,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},
      {  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},
      {  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},
      {  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0}
    }
  };
  for (int i = 0; i < t1_rows; i++) {
    for (int j = 0; j < t1_cols; j++) {
      t1[0][i][j] = t1[0][i][j] / 255; // Normalize values between 0 and 1
    }
  }

  // Set up input batch
  std::vector<Tensor3D> batch(batch_size, t1);

  // Set up 6 filters of size 5x5x1 (for the first convolution)
  int n_f1 = 6;
  int f1_chans = 1;
  int f1_rows = 5;
  int f1_cols = 5;
  std::vector<Tensor3D> f1(n_f1, Tensor3D(f1_chans, Tensor2D(f1_rows, Tensor1D(f1_cols, 0))));
  f1 = {
    {
      {
        { 0.1, -0.2,  0.1,  0.2,  0.1},
        { 0.0, -0.1,  0.1,  0.0,  0.2},
        {-0.1,  0.2,  0.3, -0.2, -0.1},
        { 0.0,  0.1,  0.0,  0.2,  0.0},
        {-0.2, -0.1,  0.1,  0.2, -0.1}
      }
    },
    {
      {
        { 0.0,  0.2,  0.1, -0.1, -0.2},
        { 0.1, -0.1, -0.2,  0.1,  0.2},
        {-0.1,  0.0,  0.1, -0.1,  0.0},
        { 0.2,  0.1, -0.1, -0.2,  0.1},
        { 0.1, -0.2,  0.0,  0.1,  0.1}
      }
    },
    {
      {
        { 0.2,  0.1, -0.1, -0.2,  0.0},
        {-0.1,  0.2,  0.1, -0.1,  0.0},
        { 0.0, -0.2,  0.2,  0.1, -0.1},
        {-0.1,  0.1, -0.2,  0.2,  0.1},
        { 0.1, -0.1,  0.0, -0.2,  0.1}
      }
    },
    {
      {
        { 0.1, -0.1,  0.2, -0.1,  0.1},
        {-0.2,  0.0,  0.1,  0.2, -0.1},
        { 0.1, -0.2,  0.0,  0.1, -0.1},
        { 0.0,  0.1, -0.1,  0.2,  0.1},
        {-0.1, -0.2,  0.1, -0.1,  0.2}
      }
    },
    {
      {
        {-0.1,  0.2,  0.1, -0.1,  0.0},
        { 0.1, -0.2,  0.2, -0.1,  0.1},
        {-0.2,  0.1,  0.0,  0.1, -0.2},
        { 0.2, -0.1, -0.2,  0.0,  0.1},
        {-0.1,  0.2, -0.1,  0.1,  0.0}
      }
    },
    {
      {
        { 0.0, -0.1,  0.1,  0.2,  0.1},
        { 0.2, -0.1, -0.2,  0.1,  0.0},
        {-0.1,  0.2,  0.0, -0.1,  0.1},
        { 0.1, -0.2,  0.2, -0.1,  0.2},
        { 0.0,  0.1, -0.1,  0.1, -0.2}
      }
    }
  };

  // Set up 16 filters of size 5x5x1 (for the second convolution)
  int n_f2 = 16;
  int f2_chans = 1;
  int f2_rows = 5;
  int f2_cols = 5;
  std::vector<Tensor3D> f2(n_f2, Tensor3D(f2_chans, Tensor2D(f2_rows, Tensor1D(f2_cols, 0))));
  f2 = {
    {
      {
        { 0.10, -0.20,  0.15, -0.05,  0.10},
        {-0.15,  0.25, -0.10,  0.05,  0.20},
        { 0.05, -0.10,  0.20, -0.15,  0.10},
        {-0.10,  0.20, -0.05,  0.15, -0.10},
        { 0.15, -0.20,  0.10, -0.05,  0.20}
      }
    },
    {
      {
        {-0.10,  0.15, -0.05,  0.10, -0.20},
        { 0.20, -0.15,  0.10, -0.05,  0.25},
        {-0.05,  0.10, -0.15,  0.20, -0.10},
        { 0.10, -0.20,  0.25, -0.15,  0.05},
        {-0.20,  0.05, -0.10,  0.15, -0.25}
      }
    },
    {
      {
        { 0.05, -0.10,  0.15, -0.20,  0.25},
        {-0.20,  0.15, -0.10,  0.05, -0.15},
        { 0.25, -0.05,  0.20, -0.10,  0.15},
        {-0.10,  0.25, -0.20,  0.15, -0.05},
        { 0.15, -0.10,  0.25, -0.20,  0.10}
      }
    },
    {
      {
        { 0.20, -0.05,  0.10, -0.15,  0.05},
        {-0.15,  0.25, -0.20,  0.10, -0.05},
        { 0.05, -0.10,  0.20, -0.25,  0.15},
        {-0.10,  0.05, -0.15,  0.20, -0.10},
        { 0.25, -0.20,  0.05, -0.10,  0.15}
      }
    },
    {
      {
        {-0.05,  0.10, -0.15,  0.20, -0.10},
        { 0.15, -0.20,  0.05, -0.10,  0.25},
        {-0.10,  0.15, -0.20,  0.05, -0.25},
        { 0.20, -0.05,  0.10, -0.15,  0.10},
        {-0.25,  0.20, -0.05,  0.15, -0.10}
      }
    },
    {
      {
        { 0.25, -0.10,  0.05, -0.15,  0.10},
        {-0.10,  0.20, -0.05,  0.15, -0.20},
        { 0.15, -0.25,  0.10, -0.05,  0.20},
        {-0.05,  0.15, -0.10,  0.25, -0.15},
        { 0.10, -0.20,  0.05, -0.10,  0.25}
      }
    },
    {
      {
        {-0.20,  0.25, -0.15,  0.05, -0.10},
        { 0.10, -0.05,  0.20, -0.25,  0.15},
        {-0.15,  0.10, -0.05,  0.20, -0.25},
        { 0.05, -0.20,  0.15, -0.10,  0.25},
        {-0.10,  0.15, -0.25,  0.05, -0.20}
      }
    },
    {
      {
        { 0.10, -0.20,  0.15, -0.05,  0.20},
        {-0.25,  0.10, -0.05,  0.15, -0.20},
        { 0.20, -0.25,  0.10, -0.15,  0.05},
        {-0.10,  0.20, -0.25,  0.05, -0.10},
        { 0.15, -0.05,  0.20, -0.10,  0.25}
      }
    },
    {
      {
        {-0.10,  0.25, -0.15,  0.10, -0.20},
        { 0.05, -0.10,  0.20, -0.15,  0.25},
        {-0.20,  0.05, -0.10,  0.25, -0.15},
        { 0.10, -0.25,  0.15, -0.20,  0.05},
        {-0.15,  0.10, -0.20,  0.05,  0.25}
      }
    },
    {
      {
        { 0.20, -0.10,  0.05, -0.15,  0.10},
        {-0.05,  0.15, -0.20,  0.25, -0.10},
        { 0.10, -0.20,  0.15, -0.05,  0.20},
        {-0.15,  0.10, -0.05,  0.25, -0.20},
        { 0.25, -0.15,  0.10, -0.20,  0.05}
      }
    },
    {
      {
        {-0.25,  0.10, -0.05,  0.20, -0.15},
        { 0.05, -0.20,  0.15, -0.10,  0.25},
        {-0.15,  0.25, -0.10,  0.05, -0.20},
        { 0.20, -0.10,  0.25, -0.05,  0.10},
        {-0.10,  0.20, -0.25,  0.15, -0.05}
      }
    },
    {
      {
        { 0.05, -0.15,  0.10, -0.20,  0.15},
        {-0.10,  0.25, -0.20,  0.05, -0.10},
        { 0.15, -0.10,  0.05, -0.20,  0.25},
        {-0.20,  0.15, -0.25,  0.10, -0.05},
        { 0.10, -0.20,  0.15, -0.10,  0.05}
      }
    },
    {
      {
        {-0.05,  0.10, -0.15,  0.25, -0.20},
        { 0.15, -0.20,  0.05, -0.10,  0.20},
        {-0.10,  0.25, -0.05,  0.15, -0.25},
        { 0.20, -0.10,  0.15, -0.20,  0.10},
        {-0.15,  0.05, -0.20,  0.10,  0.25}
      }
    },
    {
      {
        { 0.25, -0.20,  0.10, -0.15,  0.05},
        {-0.15,  0.10, -0.05,  0.20, -0.25},
        { 0.10, -0.25,  0.15, -0.20,  0.05},
        {-0.20,  0.05, -0.15,  0.10, -0.25},
        { 0.15, -0.10,  0.20, -0.05,  0.10}
      }
    },
    {
      {
        {-0.10,  0.05, -0.20,  0.15, -0.25},
        { 0.20, -0.15,  0.10, -0.05,  0.25},
        {-0.05,  0.10, -0.20,  0.25, -0.15},
        { 0.15, -0.20,  0.05, -0.10,  0.20},
        {-0.25,  0.15, -0.05,  0.10, -0.20}
      }
    },
    {
      {
        { 0.15, -0.10,  0.20, -0.05,  0.10},
        {-0.25,  0.05, -0.15,  0.20, -0.10},
        { 0.10, -0.20,  0.05, -0.15,  0.25},
        {-0.05,  0.15, -0.10,  0.25, -0.20},
        { 0.20, -0.25,  0.10, -0.05,  0.15}
      }
    }
  };

  // Set up fully connected layers output shapes
  int fc_0 = 256;
  int fc_1 = 120;
  int fc_2 = 84;
  int fc_3 = 10;

  // Set up fully connected layers filters
  Tensor2D f_fc1(fc_1, Tensor1D(fc_0, 1));
  Tensor2D f_fc2(fc_2, Tensor1D(fc_1, 1));
  Tensor2D f_fc3(fc_3, Tensor1D(fc_2, 1));

  // Set up fully connected layers biases
  Tensor1D b_fc1(fc_1, 0);
  Tensor1D b_fc2(fc_2, 0);
  Tensor1D b_fc3(fc_3, 0);

  // Conv2D
  std::vector<Tensor3D> output1(batch_size, Tensor3D(6, Tensor2D(24, Tensor1D(24, 0))));
  for (int batch_id = 0; batch_id < batch_size; batch_id++) {
    output1[batch_id] = tensor_conv_3d(batch[batch_id], f1, ACTIVATION_FUNCTION_RELU);
  }

  // AveragePooling2D
  std::vector<Tensor3D> output2(batch_size, Tensor3D(6, Tensor2D(12, Tensor1D(12, 0))));
  for (int batch_id = 0; batch_id < batch_size; batch_id++) {
    output2[batch_id] = tensor_avg_pooling_3d(output1[batch_id]);
  }

  // Conv2D
  std::vector<Tensor3D> output3(batch_size, Tensor3D(16, Tensor2D(8, Tensor1D(8, 0))));
  for (int batch_id = 0; batch_id < batch_size; batch_id++) {
    output3[batch_id] = tensor_conv_3d(output2[batch_id], f2, ACTIVATION_FUNCTION_RELU);
  }

  // AveragePooling2D
  std::vector<Tensor3D> output4(batch_size, Tensor3D(16, Tensor2D(4, Tensor1D(4, 0))));
  for (int batch_id = 0; batch_id < batch_size; batch_id++) {
    output4[batch_id] = tensor_avg_pooling_3d(output3[batch_id]);
  }

  // Flatten
  std::vector<Tensor1D> output5(batch_size, Tensor1D(256, 0));
  for (int batch_id = 0; batch_id < batch_size; batch_id++) {
    output5[batch_id] = tensor_flatten_3d(output4[batch_id]);
  }

  // Dense
  std::vector<Tensor1D> output6(batch_size, Tensor1D(120, 0));
  for (int batch_id = 0; batch_id < batch_size; batch_id++) {
    output6[batch_id] = tensor_dense_1d(output5[batch_id], f_fc1, b_fc1, ACTIVATION_FUNCTION_RELU);
  }

  // Dense
  std::vector<Tensor1D> output7(batch_size, Tensor1D(84, 0));
  for (int batch_id = 0; batch_id < batch_size; batch_id++) {
    output7[batch_id] = tensor_dense_1d(output6[batch_id], f_fc2, b_fc2, ACTIVATION_FUNCTION_RELU);
  }

  // Dense
  std::vector<Tensor1D> output8(batch_size, Tensor1D(10, 0));
  for (int batch_id = 0; batch_id < batch_size; batch_id++) {
    output8[batch_id] = tensor_dense_1d(output7[batch_id], f_fc3, b_fc3, ACTIVATION_FUNCTION_SOFTMAX);
  }
}

void run_model_debug(int batch_size) {
  // Set up one input tensor
  int t1_chans = 1;
  int t1_rows = 28;
  int t1_cols = 28;
  Tensor3D t1(t1_chans, Tensor2D(t1_rows, Tensor1D(t1_cols, 0)));
  t1 = {
    {
      {  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},
      {  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},
      {  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},
      {  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},
      {  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},
      {  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   8,  38, 137, 146, 232, 254, 255, 255, 197, 109,   0,   0,   0,   0,   0,   0,   0,   0},
      {  0,   0,   0,   0,   0,   0,   0,   0,   0,  87, 107, 253, 253, 253, 253, 253, 253, 253, 253, 188,   0,   0,   0,   0,   0,   0,   0,   0},
      {  0,   0,   0,   0,   0,   0,   0,   0, 120, 237, 253, 253, 253, 248, 209, 139, 139, 230, 253, 188,   0,   0,   0,   0,   0,   0,   0,   0},
      {  0,   0,   0,   0,   0,   0,   0,   0, 112, 229, 210, 128,  96,   0,   0,   0,   0, 117, 253, 188,   0,   0,   0,   0,   0,   0,   0,   0},
      {  0,   0,   0,   0,   0,   0,   0,   0,   0, 28,   0,   0,   0,   0,   0,   0,   45, 241, 245,  82,   0,   0,   0,   0,   0,   0,   0,   0},
      {  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  60, 253, 125,   0,   0,   0,   0,   0,   0,   0,   0,   0},
      {  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  32, 134, 148,  98, 127,   4,   0,   0,   0,   0,   0,   0,   0,   0,   0},
      {  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   7, 201, 253, 200,  59,  12,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},
      {  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 122, 246, 253, 223,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},
      {  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 233, 253, 253, 236,  55,  11,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},
      {  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 233, 253, 253, 253, 253, 210,  48,   0,   0,   0,   0,   0,   0,   0,   0,   0},
      {  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  68, 193, 185, 243, 253, 253, 173,   0,   0,   0,   0,   0,   0,   0,   0,   0},
      {  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  59, 253, 253, 226,   0,   0,   0,   0,   0,   0,   0,   0,   0},
      {  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  22, 253, 253, 226,   0,   0,   0,   0,   0,   0,   0,   0,   0},
      {  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  10,  93, 253, 253, 123,   0,   0,   0,   0,   0,   0,   0,   0,   0},
      {  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  14, 180, 253, 253, 228,  42,   0,   0,   0,   0,   0,   0,   0,   0,   0},
      {  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  95, 181, 253, 253, 253,  66,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},
      {  0,   0,   0,   0,   0,   0,   0,   6,  33,  33, 100, 178, 253, 253, 253, 241,  88,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},
      {  0,   0,   0,   0,   0,   0,   0,  32, 236, 253, 253, 253, 253, 253, 228, 122,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},
      {  0,   0,   0,   0,   0,   0,   0,   0, 143, 253, 253, 253, 154,  76,  24,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},
      {  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},
      {  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},
      {  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0}
    }
  };
  for (int i = 0; i < t1_rows; i++) {
    for (int j = 0; j < t1_cols; j++) {
      t1[0][i][j] = t1[0][i][j] / 255; // Normalize values between 0 and 1
    }
  }

  // Set up input batch
  std::vector<Tensor3D> batch(batch_size, t1);

  // Set up 6 filters of size 5x5x1 (for the first convolution)
  int n_f1 = 6;
  int f1_chans = 1;
  int f1_rows = 5;
  int f1_cols = 5;
  std::vector<Tensor3D> f1(n_f1, Tensor3D(f1_chans, Tensor2D(f1_rows, Tensor1D(f1_cols, 0))));
  f1 = {
    {
      {
        { 0.1, -0.2,  0.1,  0.2,  0.1},
        { 0.0, -0.1,  0.1,  0.0,  0.2},
        {-0.1,  0.2,  0.3, -0.2, -0.1},
        { 0.0,  0.1,  0.0,  0.2,  0.0},
        {-0.2, -0.1,  0.1,  0.2, -0.1}
      }
    },
    {
      {
        { 0.0,  0.2,  0.1, -0.1, -0.2},
        { 0.1, -0.1, -0.2,  0.1,  0.2},
        {-0.1,  0.0,  0.1, -0.1,  0.0},
        { 0.2,  0.1, -0.1, -0.2,  0.1},
        { 0.1, -0.2,  0.0,  0.1,  0.1}
      }
    },
    {
      {
        { 0.2,  0.1, -0.1, -0.2,  0.0},
        {-0.1,  0.2,  0.1, -0.1,  0.0},
        { 0.0, -0.2,  0.2,  0.1, -0.1},
        {-0.1,  0.1, -0.2,  0.2,  0.1},
        { 0.1, -0.1,  0.0, -0.2,  0.1}
      }
    },
    {
      {
        { 0.1, -0.1,  0.2, -0.1,  0.1},
        {-0.2,  0.0,  0.1,  0.2, -0.1},
        { 0.1, -0.2,  0.0,  0.1, -0.1},
        { 0.0,  0.1, -0.1,  0.2,  0.1},
        {-0.1, -0.2,  0.1, -0.1,  0.2}
      }
    },
    {
      {
        {-0.1,  0.2,  0.1, -0.1,  0.0},
        { 0.1, -0.2,  0.2, -0.1,  0.1},
        {-0.2,  0.1,  0.0,  0.1, -0.2},
        { 0.2, -0.1, -0.2,  0.0,  0.1},
        {-0.1,  0.2, -0.1,  0.1,  0.0}
      }
    },
    {
      {
        { 0.0, -0.1,  0.1,  0.2,  0.1},
        { 0.2, -0.1, -0.2,  0.1,  0.0},
        {-0.1,  0.2,  0.0, -0.1,  0.1},
        { 0.1, -0.2,  0.2, -0.1,  0.2},
        { 0.0,  0.1, -0.1,  0.1, -0.2}
      }
    }
  };

  // Set up 16 filters of size 5x5x1 (for the second convolution)
  int n_f2 = 16;
  int f2_chans = 1;
  int f2_rows = 5;
  int f2_cols = 5;
  std::vector<Tensor3D> f2(n_f2, Tensor3D(f2_chans, Tensor2D(f2_rows, Tensor1D(f2_cols, 0))));
  f2 = {
    {
      {
        { 0.10, -0.20,  0.15, -0.05,  0.10},
        {-0.15,  0.25, -0.10,  0.05,  0.20},
        { 0.05, -0.10,  0.20, -0.15,  0.10},
        {-0.10,  0.20, -0.05,  0.15, -0.10},
        { 0.15, -0.20,  0.10, -0.05,  0.20}
      }
    },
    {
      {
        {-0.10,  0.15, -0.05,  0.10, -0.20},
        { 0.20, -0.15,  0.10, -0.05,  0.25},
        {-0.05,  0.10, -0.15,  0.20, -0.10},
        { 0.10, -0.20,  0.25, -0.15,  0.05},
        {-0.20,  0.05, -0.10,  0.15, -0.25}
      }
    },
    {
      {
        { 0.05, -0.10,  0.15, -0.20,  0.25},
        {-0.20,  0.15, -0.10,  0.05, -0.15},
        { 0.25, -0.05,  0.20, -0.10,  0.15},
        {-0.10,  0.25, -0.20,  0.15, -0.05},
        { 0.15, -0.10,  0.25, -0.20,  0.10}
      }
    },
    {
      {
        { 0.20, -0.05,  0.10, -0.15,  0.05},
        {-0.15,  0.25, -0.20,  0.10, -0.05},
        { 0.05, -0.10,  0.20, -0.25,  0.15},
        {-0.10,  0.05, -0.15,  0.20, -0.10},
        { 0.25, -0.20,  0.05, -0.10,  0.15}
      }
    },
    {
      {
        {-0.05,  0.10, -0.15,  0.20, -0.10},
        { 0.15, -0.20,  0.05, -0.10,  0.25},
        {-0.10,  0.15, -0.20,  0.05, -0.25},
        { 0.20, -0.05,  0.10, -0.15,  0.10},
        {-0.25,  0.20, -0.05,  0.15, -0.10}
      }
    },
    {
      {
        { 0.25, -0.10,  0.05, -0.15,  0.10},
        {-0.10,  0.20, -0.05,  0.15, -0.20},
        { 0.15, -0.25,  0.10, -0.05,  0.20},
        {-0.05,  0.15, -0.10,  0.25, -0.15},
        { 0.10, -0.20,  0.05, -0.10,  0.25}
      }
    },
    {
      {
        {-0.20,  0.25, -0.15,  0.05, -0.10},
        { 0.10, -0.05,  0.20, -0.25,  0.15},
        {-0.15,  0.10, -0.05,  0.20, -0.25},
        { 0.05, -0.20,  0.15, -0.10,  0.25},
        {-0.10,  0.15, -0.25,  0.05, -0.20}
      }
    },
    {
      {
        { 0.10, -0.20,  0.15, -0.05,  0.20},
        {-0.25,  0.10, -0.05,  0.15, -0.20},
        { 0.20, -0.25,  0.10, -0.15,  0.05},
        {-0.10,  0.20, -0.25,  0.05, -0.10},
        { 0.15, -0.05,  0.20, -0.10,  0.25}
      }
    },
    {
      {
        {-0.10,  0.25, -0.15,  0.10, -0.20},
        { 0.05, -0.10,  0.20, -0.15,  0.25},
        {-0.20,  0.05, -0.10,  0.25, -0.15},
        { 0.10, -0.25,  0.15, -0.20,  0.05},
        {-0.15,  0.10, -0.20,  0.05,  0.25}
      }
    },
    {
      {
        { 0.20, -0.10,  0.05, -0.15,  0.10},
        {-0.05,  0.15, -0.20,  0.25, -0.10},
        { 0.10, -0.20,  0.15, -0.05,  0.20},
        {-0.15,  0.10, -0.05,  0.25, -0.20},
        { 0.25, -0.15,  0.10, -0.20,  0.05}
      }
    },
    {
      {
        {-0.25,  0.10, -0.05,  0.20, -0.15},
        { 0.05, -0.20,  0.15, -0.10,  0.25},
        {-0.15,  0.25, -0.10,  0.05, -0.20},
        { 0.20, -0.10,  0.25, -0.05,  0.10},
        {-0.10,  0.20, -0.25,  0.15, -0.05}
      }
    },
    {
      {
        { 0.05, -0.15,  0.10, -0.20,  0.15},
        {-0.10,  0.25, -0.20,  0.05, -0.10},
        { 0.15, -0.10,  0.05, -0.20,  0.25},
        {-0.20,  0.15, -0.25,  0.10, -0.05},
        { 0.10, -0.20,  0.15, -0.10,  0.05}
      }
    },
    {
      {
        {-0.05,  0.10, -0.15,  0.25, -0.20},
        { 0.15, -0.20,  0.05, -0.10,  0.20},
        {-0.10,  0.25, -0.05,  0.15, -0.25},
        { 0.20, -0.10,  0.15, -0.20,  0.10},
        {-0.15,  0.05, -0.20,  0.10,  0.25}
      }
    },
    {
      {
        { 0.25, -0.20,  0.10, -0.15,  0.05},
        {-0.15,  0.10, -0.05,  0.20, -0.25},
        { 0.10, -0.25,  0.15, -0.20,  0.05},
        {-0.20,  0.05, -0.15,  0.10, -0.25},
        { 0.15, -0.10,  0.20, -0.05,  0.10}
      }
    },
    {
      {
        {-0.10,  0.05, -0.20,  0.15, -0.25},
        { 0.20, -0.15,  0.10, -0.05,  0.25},
        {-0.05,  0.10, -0.20,  0.25, -0.15},
        { 0.15, -0.20,  0.05, -0.10,  0.20},
        {-0.25,  0.15, -0.05,  0.10, -0.20}
      }
    },
    {
      {
        { 0.15, -0.10,  0.20, -0.05,  0.10},
        {-0.25,  0.05, -0.15,  0.20, -0.10},
        { 0.10, -0.20,  0.05, -0.15,  0.25},
        {-0.05,  0.15, -0.10,  0.25, -0.20},
        { 0.20, -0.25,  0.10, -0.05,  0.15}
      }
    }
  };

  // Set up fully connected layers output shapes
  int fc_0 = 256;
  int fc_1 = 120;
  int fc_2 = 84;
  int fc_3 = 10;

  // Set up fully connected layers filters
  Tensor2D f_fc1(fc_1, Tensor1D(fc_0, 1));
  Tensor2D f_fc2(fc_2, Tensor1D(fc_1, 1));
  Tensor2D f_fc3(fc_3, Tensor1D(fc_2, 1));

  // Set up fully connected layers biases
  Tensor1D b_fc1(fc_1, 0);
  Tensor1D b_fc2(fc_2, 0);
  Tensor1D b_fc3(fc_3, 0);

  // Conv2D
  std::vector<Tensor3D> output1(batch_size, Tensor3D(6, Tensor2D(24, Tensor1D(24, 0))));
  for (int batch_id = 0; batch_id < batch_size; batch_id++) {
    std::cout << "------------------------------------------------------------" << std::endl << std::endl;
    std::cout << "Batch " << batch_id + 1 << std::endl;
    std::cout << "Layer 1: Conv2D" << std::endl;
    std::cout << "Input tensor: " << batch[batch_id][0].size() << "x" << batch[batch_id][0][0].size() << "x" << batch[batch_id].size() << std::endl;
    tensor_print_3d(batch[batch_id]);
    std::cout << "Kernel filters (" << f1.size() << "): " << f1[0][0].size() << "x" << f1[0][0][0].size() << "x" << f1[0].size() << std::endl;
    for (int i = 0; i < n_f1; i++) {
      std::cout << "Filter " << i + 1 << ":" << std::endl;
      tensor_print_3d(f1[i]);
    }
    std::cout << "Processing Conv2D..." << std::endl;
    output1[batch_id] = tensor_conv_3d(batch[batch_id], f1, ACTIVATION_FUNCTION_RELU);
    std::cout << "Output tensor: " << output1[batch_id][0].size() << "x" << output1[batch_id][0][0].size() << "x" << output1[batch_id].size() << std::endl;
    tensor_print_3d(output1[batch_id]);
  }

  // AveragePooling2D
  std::vector<Tensor3D> output2(batch_size, Tensor3D(6, Tensor2D(12, Tensor1D(12, 0))));
  for (int batch_id = 0; batch_id < batch_size; batch_id++) {
    std::cout << "------------------------------------------------------------" << std::endl << std::endl;
    std::cout << "Batch " << batch_id + 1 << std::endl;
    std::cout << "Layer 2: AveragePooling2D" << std::endl;
    std::cout << "Input tensor: " << output1[batch_id][0].size() << "x" << output1[batch_id][0][0].size() << "x" << output1[batch_id].size() << std::endl;
    tensor_print_3d(output1[batch_id]);
    std::cout << "Processing AveragePooling2D..." << std::endl;
    output2[batch_id] = tensor_avg_pooling_3d(output1[batch_id]);
    std::cout << "Output tensor: " << output2[batch_id][0].size() << "x" << output2[batch_id][0][0].size() << "x" << output2[batch_id].size() << std::endl;
    tensor_print_3d(output2[batch_id]);
  }

  // Conv2D
  std::vector<Tensor3D> output3(batch_size, Tensor3D(16, Tensor2D(8, Tensor1D(8, 0))));
  for (int batch_id = 0; batch_id < batch_size; batch_id++) {
    std::cout << "------------------------------------------------------------" << std::endl << std::endl;
    std::cout << "Batch " << batch_id + 1 << std::endl;
    std::cout << "Layer 3: Conv2D" << std::endl;
    std::cout << "Input tensor: " << output2[batch_id][0].size() << "x" << output2[batch_id][0][0].size() << "x" << output2[batch_id].size() << std::endl;
    tensor_print_3d(output2[batch_id]);
    std::cout << "Kernel filters (" << f2.size() << "): " << f2[0][0].size() << "x" << f2[0][0][0].size() << "x" << f2[0].size() << std::endl;
    for (int i = 0; i < n_f2; i++) {
      std::cout << "Filter " << i + 1 << ":" << std::endl;
      tensor_print_3d(f2[i]);
    }
    std::cout << "Processing Conv2D..." << std::endl;
    output3[batch_id] = tensor_conv_3d(output2[batch_id], f2, ACTIVATION_FUNCTION_RELU);
    std::cout << "Output tensor: " << output3[batch_id][0].size() << "x" << output3[batch_id][0][0].size() << "x" << output3[batch_id].size() << std::endl;
    tensor_print_3d(output3[batch_id]);
  }

  // AveragePooling2D
  std::vector<Tensor3D> output4(batch_size, Tensor3D(16, Tensor2D(4, Tensor1D(4, 0))));
  for (int batch_id = 0; batch_id < batch_size; batch_id++) {
    std::cout << "------------------------------------------------------------" << std::endl << std::endl;
    std::cout << "Batch " << batch_id + 1 << std::endl;
    std::cout << "Layer 4: AveragePooling2D" << std::endl;
    std::cout << "Input tensor: " << output3[batch_id][0].size() << "x" << output3[batch_id][0][0].size() << "x" << output3[batch_id].size() << std::endl;
    tensor_print_3d(output3[batch_id]);
    std::cout << "Processing AveragePooling2D..." << std::endl;
    output4[batch_id] = tensor_avg_pooling_3d(output3[batch_id]);
    std::cout << "Output tensor: " << output4[batch_id][0].size() << "x" << output4[batch_id][0][0].size() << "x" << output4[batch_id].size() << std::endl;
    tensor_print_3d(output4[batch_id]);
  }

  // Flatten
  std::vector<Tensor1D> output5(batch_size, Tensor1D(256, 0));
  for (int batch_id = 0; batch_id < batch_size; batch_id++) {
    std::cout << "------------------------------------------------------------" << std::endl;
    std::cout << "Batch " << batch_id + 1 << std::endl;
    std::cout << "Layer 5: Flatten" << std::endl;
    std::cout << "Input tensor: " << output4[batch_id][0].size() << "x" << output4[batch_id][0][0].size() << "x" << output4[batch_id].size() << std::endl;
    tensor_print_3d(output4[batch_id]);
    std::cout << "Processing Flatten..." << std::endl;
    output5[batch_id] = tensor_flatten_3d(output4[batch_id]);
    std::cout << "Output tensor: " << output5[batch_id].size() << std::endl;
    tensor_print_1d(output5[batch_id]);
  }

  // Dense
  std::vector<Tensor1D> output6(batch_size, Tensor1D(120, 0));
  for (int batch_id = 0; batch_id < batch_size; batch_id++) {
    std::cout << "------------------------------------------------------------" << std::endl;
    std::cout << "Batch " << batch_id + 1 << std::endl;
    std::cout << "Layer 6: Dense" << std::endl;
    std::cout << "Input tensor: " << output5[batch_id].size() << std::endl;
    tensor_print_1d(output5[batch_id]);
    std::cout << "Processing Dense..." << std::endl;
    output6[batch_id] = tensor_dense_1d(output5[batch_id], f_fc1, b_fc1, ACTIVATION_FUNCTION_RELU);
    std::cout << "Output tensor: " << output6[batch_id].size() << std::endl;
    tensor_print_1d(output6[batch_id]);
  }

  // Dense
  std::vector<Tensor1D> output7(batch_size, Tensor1D(84, 0));
  for (int batch_id = 0; batch_id < batch_size; batch_id++) {
    std::cout << "------------------------------------------------------------" << std::endl;
    std::cout << "Batch " << batch_id + 1 << std::endl;
    std::cout << "Layer 7: Dense" << std::endl;
    std::cout << "Input tensor: " << output6[batch_id].size() << std::endl;
    tensor_print_1d(output6[batch_id]);
    std::cout << "Processing Dense..." << std::endl;
    output7[batch_id] = tensor_dense_1d(output6[batch_id], f_fc2, b_fc2, ACTIVATION_FUNCTION_RELU);
    std::cout << "Output tensor: " << output7[batch_id].size() << std::endl;
    tensor_print_1d(output7[batch_id]);
  }

  // Dense
  std::vector<Tensor1D> output8(batch_size, Tensor1D(10, 0));
  for (int batch_id = 0; batch_id < batch_size; batch_id++) {
    std::cout << "------------------------------------------------------------" << std::endl;
    std::cout << "Batch " << batch_id + 1 << std::endl;
    std::cout << "Layer 8: Dense" << std::endl;
    std::cout << "Input tensor: " << output7[batch_id].size() << std::endl;
    tensor_print_1d(output7[batch_id]);
    std::cout << "Processing Dense..." << std::endl;
    output8[batch_id] = tensor_dense_1d(output7[batch_id], f_fc3, b_fc3, ACTIVATION_FUNCTION_SOFTMAX);
    std::cout << "Output tensor: " << output8[batch_id].size() << std::endl;
    tensor_print_1d(output8[batch_id]);
  }
}
