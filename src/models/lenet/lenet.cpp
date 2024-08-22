#include <algorithm>
#include <iostream>
#include <vector>
#include "../../tensor_utils.hpp"
#include "../../profiling_utils.hpp"
#include "data/t1_vals.hpp"
#include "data/f1_vals.hpp"
#include "data/f2_vals.hpp"
#include "data/f_fc1_vals.hpp"
#include "data/f_fc2_vals.hpp"
#include "data/f_fc3_vals.hpp"
#include "data/b1_vals.hpp"
#include "data/b2_vals.hpp"
#include "data/b_fc1_vals.hpp"
#include "data/b_fc2_vals.hpp"
#include "data/b_fc3_vals.hpp"

void run_model(int batch_size, std::vector<int> profile_indices) {
  // Structure of data used for profiling
  ProfilingData pd;

  // Variable used to check if a layer need profiling
  bool need_to_profile_layer = false;

  if (!profile_indices.empty()) {
    // Initialize profiling
    pd = profiling_init();
    profiling_print_header();
  }

  // Set up one input tensor
  int t1_chans = 1;
  int t1_rows = 28;
  int t1_cols = 28;
  Tensor3D t1 = tensor_init_3d(t1_chans, t1_rows, t1_cols, t1_vals);

  // Set up input batch
  float batch_vals[batch_size * t1_rows * t1_cols];
  for (int i = 0; i < batch_size; ++i) {
    for (int j = 0; j < t1_rows * t1_cols; ++j) {
      batch_vals[i * t1_rows * t1_cols + j] = t1_vals[j];
    }
  }
  Tensor4D batch = tensor_init_4d(batch_size, t1_chans, t1_rows, t1_cols, batch_vals);
  tensor_delete_3d(t1, t1_chans, t1_rows);

  // Set up 6 filters of size 5x5x1 (for the first convolution)
  int n_f1 = 6;
  int f1_chans = 1;
  int f1_rows = 5;
  int f1_cols = 5;
  Tensor4D f1 = tensor_init_4d(n_f1, f1_chans, f1_rows, f1_cols, f1_vals);
  
  // Set up 6 biases (for the first convolution)
  int b1_cols = 6;
  Tensor1D b1 = tensor_init_1d(b1_cols, b1_vals);

  // Set up 16 filters of size 5x5x6 (for the second convolution)
  int n_f2 = 16;
  int f2_chans = 6;
  int f2_rows = 5;
  int f2_cols = 5;
  Tensor4D f2 = tensor_init_4d(n_f2, f2_chans, f2_rows, f2_cols, f2_vals);

  // Set up 16 biases (for the second convolution)
  int b2_cols = 16;
  Tensor1D b2 = tensor_init_1d(b2_cols, b2_vals);

  // Set up fully connected layers output shapes
  int fc_0 = 256;
  int fc_1 = 120;
  int fc_2 = 84;
  int fc_3 = 10;

  // Set up fully connected layers filters
  Tensor2D f_fc1 = tensor_init_2d(fc_1, fc_0, f_fc1_vals);
  Tensor2D f_fc2 = tensor_init_2d(fc_2, fc_1, f_fc2_vals);
  Tensor2D f_fc3 = tensor_init_2d(fc_3, fc_2, f_fc3_vals);

  // Set up fully connected layers biases
  Tensor1D b_fc1 = tensor_init_1d(fc_1, b_fc1_vals);
  Tensor1D b_fc2 = tensor_init_1d(fc_2, b_fc2_vals);
  Tensor1D b_fc3 = tensor_init_1d(fc_3, b_fc3_vals);

  // Conv2D
  need_to_profile_layer = std::find(profile_indices.begin(), profile_indices.end(), 1) != profile_indices.end();
  if (need_to_profile_layer) {
    profiling_start(pd);
    pd.name = "Conv2D";
  }
  int output1_chans = 6;
  int output1_rows = 24;
  int output1_cols = 24;
  Tensor4D output1 = tensor_4d(batch_size, output1_chans, output1_rows, output1_cols);
  for (int batch_id = 0; batch_id < batch_size; batch_id++) {
    output1[batch_id] = tensor_conv_3d(batch[batch_id], t1_chans, t1_rows, t1_cols, f1, n_f1, f1_chans, f1_rows, f1_cols, b1, ACTIVATION_FUNCTION_RELU);
  }
  tensor_delete_4d(batch, batch_size, t1_chans, t1_rows);
  tensor_delete_4d(f1, n_f1, f1_chans, f1_rows);
  if (need_to_profile_layer) {
    profiling_stop(pd);
    profiling_print_results(pd);
  }

  // AveragePooling2D
  need_to_profile_layer = std::find(profile_indices.begin(), profile_indices.end(), 2) != profile_indices.end();
  if (need_to_profile_layer) {
    profiling_start(pd);
    pd.name = "AvgPooling2D";
  }
  int output2_chans = 6;
  int output2_rows = 12;
  int output2_cols = 12;
  Tensor4D output2 = tensor_4d(batch_size, output2_chans, output2_rows, output2_cols);
  for (int batch_id = 0; batch_id < batch_size; batch_id++) {
    output2[batch_id] = tensor_avg_pooling_3d(output1[batch_id], output1_chans, output1_rows, output1_cols);
  }
  tensor_delete_4d(output1, batch_size, output1_chans, output1_rows);
  if (need_to_profile_layer) {
    profiling_stop(pd);
    profiling_print_results(pd);
  }

  // Conv2D
  need_to_profile_layer = std::find(profile_indices.begin(), profile_indices.end(), 3) != profile_indices.end();
  if (need_to_profile_layer) {
    profiling_start(pd);
    pd.name = "Conv2D";
  }
  int output3_chans = 16;
  int output3_rows = 8;
  int output3_cols = 8;
  Tensor4D output3 = tensor_4d(batch_size, output3_chans, output3_rows, output3_cols);
  for (int batch_id = 0; batch_id < batch_size; batch_id++) {
    output3[batch_id] = tensor_conv_3d(output2[batch_id], output2_chans, output2_rows, output2_cols, f2, n_f2, f2_chans, f2_rows, f2_cols, b2, ACTIVATION_FUNCTION_RELU);
  }
  tensor_delete_4d(output2, batch_size, output2_chans, output2_rows);
  tensor_delete_4d(f2, n_f2, f2_chans, f2_rows);
  if (need_to_profile_layer) {
    profiling_stop(pd);
    profiling_print_results(pd);
  }

  // AveragePooling2D
  need_to_profile_layer = std::find(profile_indices.begin(), profile_indices.end(), 4) != profile_indices.end();
  if (need_to_profile_layer) {
    profiling_start(pd);
    pd.name = "AvgPooling2D";
  }
  int output4_chans = 16;
  int output4_rows = 4;
  int output4_cols = 4;
  Tensor4D output4 = tensor_4d(batch_size, output4_chans, output4_rows, output4_cols);
  for (int batch_id = 0; batch_id < batch_size; batch_id++) {
    output4[batch_id] = tensor_avg_pooling_3d(output3[batch_id], output3_chans, output3_rows, output3_cols);
  }
  tensor_delete_4d(output3, batch_size, output3_chans, output3_rows);
  if (need_to_profile_layer) {
    profiling_stop(pd);
    profiling_print_results(pd);
  }

  // Flatten
  need_to_profile_layer = std::find(profile_indices.begin(), profile_indices.end(), 5) != profile_indices.end();
  if (need_to_profile_layer) {
    profiling_start(pd);
    pd.name = "Flatten";
  }
  int output5_cols = 256;
  Tensor2D output5 = tensor_2d(batch_size, output5_cols);
  for (int batch_id = 0; batch_id < batch_size; batch_id++) {
    output5[batch_id] = tensor_flatten_3d(output4[batch_id], output4_chans, output4_rows, output4_cols);
  }
  tensor_delete_4d(output4, batch_size, output4_chans, output4_rows);
  if (need_to_profile_layer) {
    profiling_stop(pd);
    profiling_print_results(pd);
  }

  // Reorder the tensor
  int output5bis_cols = 256;
  Tensor2D output5bis = tensor_2d(batch_size, output5bis_cols);
  int input_shape[4] = {1, output4_chans, output4_rows, output4_cols}; // Dimensions (1, 16, 4, 4)
  int perm[4] = {0, 2, 3, 1}; // Desired permutation of dimensions (1, 4, 4, 16)
  for (int batch_id = 0; batch_id < batch_size; batch_id++) {
    output5bis[batch_id] = tensor_transpose_perm_1d(output5[batch_id], output5_cols, input_shape, perm, 4);
  }
  tensor_delete_2d(output5, batch_size);

  // Dense
  need_to_profile_layer = std::find(profile_indices.begin(), profile_indices.end(), 6) != profile_indices.end();
  if (need_to_profile_layer) {
    profiling_start(pd);
    pd.name = "Dense";
  }
  int output6_cols = 120;
  Tensor2D output6 = tensor_2d(batch_size, output6_cols);
  for (int batch_id = 0; batch_id < batch_size; batch_id++) {
    output6[batch_id] = tensor_dense_1d(output5bis[batch_id], output5bis_cols, f_fc1, fc_1, fc_0,  b_fc1, ACTIVATION_FUNCTION_RELU);
  }
  tensor_delete_2d(output5bis, batch_size);
  tensor_delete_2d(f_fc1, fc_1);
  tensor_delete_1d(b_fc1);
  if (need_to_profile_layer) {
    profiling_stop(pd);
    profiling_print_results(pd);
  }

  // Dense
  need_to_profile_layer = std::find(profile_indices.begin(), profile_indices.end(), 7) != profile_indices.end();
  if (need_to_profile_layer) {
    profiling_start(pd);
    pd.name = "Dense";
  }
  int output7_cols = 84;
  Tensor2D output7 = tensor_2d(batch_size, output7_cols);
  for (int batch_id = 0; batch_id < batch_size; batch_id++) {
    output7[batch_id] = tensor_dense_1d(output6[batch_id], output6_cols, f_fc2, fc_2, fc_1, b_fc2, ACTIVATION_FUNCTION_RELU);
  }
  tensor_delete_2d(output6, batch_size);
  tensor_delete_2d(f_fc2, fc_2);
  tensor_delete_1d(b_fc2);
  if (need_to_profile_layer) {
    profiling_stop(pd);
    profiling_print_results(pd);
  }

  // Dense
  need_to_profile_layer = std::find(profile_indices.begin(), profile_indices.end(), 8) != profile_indices.end();
  if (need_to_profile_layer) {
    profiling_start(pd);
    pd.name = "Dense";
  }
  int output8_cols = 10;
  Tensor2D output8 = tensor_2d(batch_size, output8_cols);
  for (int batch_id = 0; batch_id < batch_size; batch_id++) {
    output8[batch_id] = tensor_dense_1d(output7[batch_id], output7_cols, f_fc3, fc_3, fc_2, b_fc3, ACTIVATION_FUNCTION_SOFTMAX);
  }
  tensor_delete_2d(output7, batch_size);
  tensor_delete_2d(f_fc3, fc_3);
  tensor_delete_1d(b_fc3);
  if (need_to_profile_layer) {
    profiling_stop(pd);
    profiling_print_results(pd);
  }

  tensor_delete_2d(output8, batch_size);

  if (!profile_indices.empty()) {
    // Shutdown profiling
    profiling_shutdown();
  }
}

void run_model_debug(int batch_size) {
  // Set up one input tensor
  int t1_chans = 1;
  int t1_rows = 28;
  int t1_cols = 28;
  Tensor3D t1 = tensor_init_3d(t1_chans, t1_rows, t1_cols, t1_vals);

  // Set up input batch
  float batch_vals[batch_size * t1_rows * t1_cols];
  for (int i = 0; i < batch_size; ++i) {
    for (int j = 0; j < t1_rows * t1_cols; ++j) {
      batch_vals[i * t1_rows * t1_cols + j] = t1_vals[j];
    }
  }
  Tensor4D batch = tensor_init_4d(batch_size, t1_chans, t1_rows, t1_cols, batch_vals);
  tensor_delete_3d(t1, t1_chans, t1_rows);

  // Set up 6 filters of size 5x5x1 (for the first convolution)
  int n_f1 = 6;
  int f1_chans = 1;
  int f1_rows = 5;
  int f1_cols = 5;
  Tensor4D f1 = tensor_init_4d(n_f1, f1_chans, f1_rows, f1_cols, f1_vals);
  
  // Set up 6 biases (for the first convolution)
  int b1_cols = 6;
  Tensor1D b1 = tensor_init_1d(b1_cols, b1_vals);

  // Set up 16 filters of size 5x5x6 (for the second convolution)
  int n_f2 = 16;
  int f2_chans = 6;
  int f2_rows = 5;
  int f2_cols = 5;
  Tensor4D f2 = tensor_init_4d(n_f2, f2_chans, f2_rows, f2_cols, f2_vals);

  // Set up 16 biases (for the second convolution)
  int b2_cols = 16;
  Tensor1D b2 = tensor_init_1d(b2_cols, b2_vals);

  // Set up fully connected layers output shapes
  int fc_0 = 256;
  int fc_1 = 120;
  int fc_2 = 84;
  int fc_3 = 10;

  // Set up fully connected layers filters
  Tensor2D f_fc1 = tensor_init_2d(fc_1, fc_0, f_fc1_vals);
  Tensor2D f_fc2 = tensor_init_2d(fc_2, fc_1, f_fc2_vals);
  Tensor2D f_fc3 = tensor_init_2d(fc_3, fc_2, f_fc3_vals);

  // Set up fully connected layers biases
  Tensor1D b_fc1 = tensor_init_1d(fc_1, b_fc1_vals);
  Tensor1D b_fc2 = tensor_init_1d(fc_2, b_fc2_vals);
  Tensor1D b_fc3 = tensor_init_1d(fc_3, b_fc3_vals);

  // Conv2D
  int output1_chans = 6;
  int output1_rows = 24;
  int output1_cols = 24;
  Tensor4D output1 = tensor_4d(batch_size, output1_chans, output1_rows, output1_cols);
  for (int batch_id = 0; batch_id < batch_size; batch_id++) {
    std::cout << "------------------------------------------------------------" << std::endl << std::endl;
    std::cout << "Batch " << batch_id + 1 << std::endl;
    std::cout << "Layer 1: Conv2D" << std::endl;
    std::cout << "Input tensor: " << t1_rows << "x" << t1_cols << "x" << t1_chans << std::endl;
    tensor_print_3d(batch[batch_id], t1_chans, t1_rows, t1_cols);
    std::cout << "Kernel filters (" << n_f1 << "): " << f1_rows << "x" << f1_cols << "x" << f1_chans << std::endl;
    for (int i = 0; i < n_f1; i++) {
      std::cout << "Filter " << i + 1 << ":" << std::endl;
      tensor_print_3d(f1[i], f1_chans, f1_rows, f1_cols);
    }
    std::cout << "Biases (" << b1_cols << "): " << std::endl;
   tensor_print_1d(b1, b1_cols);
    std::cout << "Processing Conv2D..." << std::endl;
    output1[batch_id] = tensor_conv_3d(batch[batch_id], t1_chans, t1_rows, t1_cols, f1, n_f1, f1_chans, f1_rows, f1_cols, b1, ACTIVATION_FUNCTION_RELU);
    std::cout << "Output tensor: " << output1_rows << "x" << output1_cols << "x" << output1_chans << std::endl;
    tensor_print_3d(output1[batch_id], output1_chans, output1_rows, output1_cols);
  }
  tensor_delete_4d(batch, batch_size, t1_chans, t1_rows);
  tensor_delete_4d(f1, n_f1, f1_chans, f1_rows);
  
  // AveragePooling2D
  int output2_chans = 6;
  int output2_rows = 12;
  int output2_cols = 12;
  Tensor4D output2 = tensor_4d(batch_size, output2_chans, output2_rows, output2_cols);
  for (int batch_id = 0; batch_id < batch_size; batch_id++) {
    std::cout << "------------------------------------------------------------" << std::endl << std::endl;
    std::cout << "Batch " << batch_id + 1 << std::endl;
    std::cout << "Layer 2: AveragePooling2D" << std::endl;
    std::cout << "Input tensor: " << output1_rows << "x" << output1_cols << "x" << output1_chans << std::endl;
    tensor_print_3d(output1[batch_id], output1_chans, output1_rows, output1_cols);
    std::cout << "Processing AveragePooling2D..." << std::endl;
    output2[batch_id] = tensor_avg_pooling_3d(output1[batch_id], output1_chans, output1_rows, output1_cols);
    std::cout << "Output tensor: " << output2_rows << "x" << output2_cols << "x" << output2_chans << std::endl;
    tensor_print_3d(output2[batch_id], output2_chans, output2_rows, output2_cols);
  }
  tensor_delete_4d(output1, batch_size, output1_chans, output1_rows);

  // Conv2D
  int output3_chans = 16;
  int output3_rows = 8;
  int output3_cols = 8;
  Tensor4D output3 = tensor_4d(batch_size, output3_chans, output3_rows, output3_cols);
  for (int batch_id = 0; batch_id < batch_size; batch_id++) {
    std::cout << "------------------------------------------------------------" << std::endl << std::endl;
    std::cout << "Batch " << batch_id + 1 << std::endl;
    std::cout << "Layer 3: Conv2D" << std::endl;
    std::cout << "Input tensor: " << output2_rows << "x" << output2_cols << "x" << output2_chans << std::endl;
    tensor_print_3d(output2[batch_id], output2_chans, output2_rows, output2_cols);
    std::cout << "Kernel filters (" << n_f2 << "): " << f2_rows << "x" << f2_cols << "x" << f2_chans << std::endl;
    for (int i = 0; i < n_f2; i++) {
      std::cout << "Filter " << i + 1 << ":" << std::endl;
      tensor_print_3d(f2[i], f2_chans, f2_rows, f2_cols);
    }
    std::cout << "Biases (" << b2_cols << "): " << std::endl;
    tensor_print_1d(b2, b2_cols);
    std::cout << "Processing Conv2D..." << std::endl;
    output3[batch_id] = tensor_conv_3d(output2[batch_id], output2_chans, output2_rows, output2_cols, f2, n_f2, f2_chans, f2_rows, f2_cols, b2, ACTIVATION_FUNCTION_RELU);
    std::cout << "Output tensor: " << output3_rows << "x" << output3_cols << "x" << output3_chans << std::endl;
    tensor_print_3d(output3[batch_id], output3_chans, output3_rows, output3_cols);
  }
  tensor_delete_4d(output2, batch_size, output2_chans, output2_rows);
  tensor_delete_4d(f2, n_f2, f2_chans, f2_rows);

  // AveragePooling2D
  int output4_chans = 16;
  int output4_rows = 4;
  int output4_cols = 4;
  Tensor4D output4 = tensor_4d(batch_size, output4_chans, output4_rows, output4_cols);
  for (int batch_id = 0; batch_id < batch_size; batch_id++) {
    std::cout << "------------------------------------------------------------" << std::endl << std::endl;
    std::cout << "Batch " << batch_id + 1 << std::endl;
    std::cout << "Layer 4: AveragePooling2D" << std::endl;
    std::cout << "Input tensor: " << output3_rows << "x" << output3_cols << "x" << output3_chans << std::endl;
    tensor_print_3d(output3[batch_id], output3_chans, output3_rows, output3_cols);
    std::cout << "Processing AveragePooling2D..." << std::endl;
    output4[batch_id] = tensor_avg_pooling_3d(output3[batch_id], output3_chans, output3_rows, output3_cols);
    std::cout << "Output tensor: " << output4_rows << "x" << output4_cols << "x" << output4_chans << std::endl;
    tensor_print_3d(output4[batch_id], output4_chans, output4_rows, output4_cols);
  }
  tensor_delete_4d(output3, batch_size, output3_chans, output3_rows);

  // Flatten
  int output5_cols = 256;
  Tensor2D output5 = tensor_2d(batch_size, output5_cols);
  for (int batch_id = 0; batch_id < batch_size; batch_id++) {
    std::cout << "------------------------------------------------------------" << std::endl;
    std::cout << "Batch " << batch_id + 1 << std::endl;
    std::cout << "Layer 5: Flatten" << std::endl;
    std::cout << "Input tensor: " << output4_rows << "x" << output4_cols << "x" << output4_chans << std::endl;
    tensor_print_3d(output4[batch_id], output4_chans, output4_rows, output4_cols);
    std::cout << "Processing Flatten..." << std::endl;
    output5[batch_id] = tensor_flatten_3d(output4[batch_id], output4_chans, output4_rows, output4_cols);
    std::cout << "Output tensor: " << output5_cols << std::endl;
    tensor_print_1d(output5[batch_id], output5_cols);
  }
  tensor_delete_4d(output4, batch_size, output4_chans, output4_rows);

  // Reorder the tensor
  int output5bis_cols = 256;
  Tensor2D output5bis = tensor_2d(batch_size, output5bis_cols);
  int input_shape[4] = {1, output4_chans, output4_rows, output4_cols}; // Dimensions (1, 16, 4, 4)
  int perm[4] = {0, 2, 3, 1}; // Desired permutation of dimensions (1, 4, 4, 16)
  for (int batch_id = 0; batch_id < batch_size; batch_id++) {
    std::cout << "------------------------------------------------------------" << std::endl;
    std::cout << "Batch " << batch_id + 1 << std::endl;
    std::cout << "Layer 5bis: Reordering" << std::endl;
    std::cout << "Input tensor: " << output5_cols << std::endl;
    tensor_print_1d(output5[batch_id], output5_cols);
    std::cout << "Processing Reordering..." << std::endl;
    output5bis[batch_id] = tensor_transpose_perm_1d(output5[batch_id], output5_cols, input_shape, perm, 4);
    std::cout << "Output tensor: " << output5bis_cols << std::endl;
    tensor_print_1d(output5bis[batch_id], output5bis_cols);
  }
  tensor_delete_2d(output5, batch_size);

  // Dense
  int output6_cols = 120;
  Tensor2D output6 = tensor_2d(batch_size, output6_cols);
  for (int batch_id = 0; batch_id < batch_size; batch_id++) {
    std::cout << "------------------------------------------------------------" << std::endl;
    std::cout << "Batch " << batch_id + 1 << std::endl;
    std::cout << "Layer 6: Dense" << std::endl;
    std::cout << "Input tensor: " << output5bis_cols << std::endl;
    tensor_print_1d(output5bis[batch_id], output5bis_cols);
    std::cout << "Processing Dense..." << std::endl;
    output6[batch_id] = tensor_dense_1d(output5bis[batch_id], output5bis_cols, f_fc1, fc_1, fc_0,  b_fc1, ACTIVATION_FUNCTION_RELU);
    std::cout << "Output tensor: " << output6_cols << std::endl;
    tensor_print_1d(output6[batch_id], output6_cols);
  }
  tensor_delete_2d(output5bis, batch_size);
  tensor_delete_2d(f_fc1, fc_1);
  tensor_delete_1d(b_fc1);

  // Dense
  int output7_cols = 84;
  Tensor2D output7 = tensor_2d(batch_size, output7_cols);
  for (int batch_id = 0; batch_id < batch_size; batch_id++) {
    std::cout << "------------------------------------------------------------" << std::endl;
    std::cout << "Batch " << batch_id + 1 << std::endl;
    std::cout << "Layer 7: Dense" << std::endl;
    std::cout << "Input tensor: " << output6_cols << std::endl;
    tensor_print_1d(output6[batch_id], output6_cols);
    std::cout << "Processing Dense..." << std::endl;
    output7[batch_id] = tensor_dense_1d(output6[batch_id], output6_cols, f_fc2, fc_2, fc_1, b_fc2, ACTIVATION_FUNCTION_RELU);
    std::cout << "Output tensor: " << output7_cols << std::endl;
    tensor_print_1d(output7[batch_id], output7_cols);
  }
  tensor_delete_2d(output6, batch_size);
  tensor_delete_2d(f_fc2, fc_2);
  tensor_delete_1d(b_fc2);

  // Dense
  int output8_cols = 10;
  Tensor2D output8 = tensor_2d(batch_size, output8_cols);
  for (int batch_id = 0; batch_id < batch_size; batch_id++) {
    std::cout << "------------------------------------------------------------" << std::endl;
    std::cout << "Batch " << batch_id + 1 << std::endl;
    std::cout << "Layer 8: Dense" << std::endl;
    std::cout << "Input tensor: " << output7_cols<< std::endl;
    tensor_print_1d(output7[batch_id], output7_cols);
    std::cout << "Processing Dense..." << std::endl;
    output8[batch_id] = tensor_dense_1d(output7[batch_id], output7_cols, f_fc3, fc_3, fc_2, b_fc3, ACTIVATION_FUNCTION_SOFTMAX);
    std::cout << "Output tensor: " << output8_cols << std::endl;
    tensor_print_1d(output8[batch_id], output8_cols);
  }
  tensor_delete_2d(output7, batch_size);
  tensor_delete_2d(f_fc3, fc_3);
  tensor_delete_1d(b_fc3);

  tensor_delete_2d(output8, batch_size);
}
