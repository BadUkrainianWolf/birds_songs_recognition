/**
  ******************************************************************************
  * @file    model.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifdef __cplusplus
extern "C" {
#endif

#ifndef SINGLE_FILE
#include "number.h"
#include "model.h"
// #include <chrono>

 // InputLayer is excluded
#include "conv1d_36.c"
#include "weights/conv1d_36.c" // InputLayer is excluded
#include "max_pooling1d_27.c" // InputLayer is excluded
#include "conv1d_37.c"
#include "weights/conv1d_37.c" // InputLayer is excluded
#include "max_pooling1d_28.c" // InputLayer is excluded
#include "conv1d_38.c"
#include "weights/conv1d_38.c" // InputLayer is excluded
#include "max_pooling1d_29.c" // InputLayer is excluded
#include "conv1d_39.c"
#include "weights/conv1d_39.c" // InputLayer is excluded
#include "average_pooling1d_9.c" // InputLayer is excluded
#include "flatten_9.c" // InputLayer is excluded
#include "dense_27.c"
#include "weights/dense_27.c" // InputLayer is excluded
#include "dense_28.c"
#include "weights/dense_28.c" // InputLayer is excluded
#include "dense_29.c"
#include "weights/dense_29.c"
#endif


void cnn(
  const input_t input,
  dense_29_output_type dense_29_output) {
  
  // Output array allocation
  static union {
    conv1d_36_output_type conv1d_36_output;
    conv1d_37_output_type conv1d_37_output;
    conv1d_38_output_type conv1d_38_output;
    conv1d_39_output_type conv1d_39_output;
    dense_27_output_type dense_27_output;
  } activations1;

  static union {
    max_pooling1d_27_output_type max_pooling1d_27_output;
    max_pooling1d_28_output_type max_pooling1d_28_output;
    max_pooling1d_29_output_type max_pooling1d_29_output;
    average_pooling1d_9_output_type average_pooling1d_9_output;
    flatten_9_output_type flatten_9_output;
    dense_28_output_type dense_28_output;
  } activations2;


// Model layers call chain 
  
  
  conv1d_36( // First layer uses input passed as model parameter
    input,
    conv1d_36_kernel,
    conv1d_36_bias,
    activations1.conv1d_36_output
    );
  
  
  max_pooling1d_27(
    activations1.conv1d_36_output,
    activations2.max_pooling1d_27_output
    );
  
  
  conv1d_37(
    activations2.max_pooling1d_27_output,
    conv1d_37_kernel,
    conv1d_37_bias,
    activations1.conv1d_37_output
    );
  
  
  max_pooling1d_28(
    activations1.conv1d_37_output,
    activations2.max_pooling1d_28_output
    );
  
  
  conv1d_38(
    activations2.max_pooling1d_28_output,
    conv1d_38_kernel,
    conv1d_38_bias,
    activations1.conv1d_38_output
    );
  
  
  max_pooling1d_29(
    activations1.conv1d_38_output,
    activations2.max_pooling1d_29_output
    );
  
  
  conv1d_39(
    activations2.max_pooling1d_29_output,
    conv1d_39_kernel,
    conv1d_39_bias,
    activations1.conv1d_39_output
    );
  
  
  average_pooling1d_9(
    activations1.conv1d_39_output,
    activations2.average_pooling1d_9_output
    );
  
  
  flatten_9(
    activations2.average_pooling1d_9_output,
    activations2.flatten_9_output
    );
  
  
  dense_27(
    activations2.flatten_9_output,
    dense_27_kernel,
    dense_27_bias,
    activations1.dense_27_output
    );
  
  
  dense_28(
    activations1.dense_27_output,
    dense_28_kernel,
    dense_28_bias,
    activations2.dense_28_output
    );
  
  
  dense_29(
    activations2.dense_28_output,
    dense_29_kernel,
    dense_29_bias,// Last layer uses output passed as model parameter
    dense_29_output
    );
}

#ifdef __cplusplus
} // extern "C"
#endif