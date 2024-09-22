#define SINGLE_FILE
/**
  ******************************************************************************
  * @file    defines.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, Université Côte d'Azur, LEAT, France
  * @version 2.1.0
  * @date    10 january 2024
  * @brief   Global C pre-processor definitions to use to build all source files (incl. CMSIS-NN)
  */

/* CMSIS-NN round mode definition */
#if defined(WITH_CMSIS_NN) || defined(WITH_NMSIS_NN)


#define ARM_NN_TRUNCATE 1
#define RISCV_NN_TRUNCATE 1

#endif // defined(WITH_CMSIS_NN) || defined(WITH_NMSIS_NN)
/**
  ******************************************************************************
  * @file    number.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    2 february 2021
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifdef __cplusplus
extern "C" {
#endif

#ifndef __NUMBER_H__
#define __NUMBER_H__

#include <stdint.h>
#include <stddef.h>
#include <math.h>

#ifdef TRAPV_SHIFT
#include <limits.h>
#include <stdio.h>
#include <assert.h>
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#endif

#define _clamp_to(type, number) clamp_to_number_t_ ## type (number)
#define clamp_to(type, number) _clamp_to(type, number)
#define _scale(type, number, scale_factor, round_mode) scale_number_t_ ## type (number, scale_factor, round_mode)
#define scale(type, number, scale_factor, round_mode) _scale(type, number, scale_factor, round_mode)
#define _scale_and_clamp_to(type, number, scale_factor, round_mode) scale_and_clamp_to_number_t_ ## type (number, scale_factor, round_mode)
#define scale_and_clamp_to(type, number, scale_factor, round_mode) _scale_and_clamp_to(type, number, scale_factor, round_mode)

typedef enum {
  ROUND_MODE_NONE,
  ROUND_MODE_FLOOR,
  ROUND_MODE_NEAREST,
} round_mode_t;

// Idea 1: Write the smallest min max interval of the net, could be an issue for hybrid int type network
// Idea 2: listing any interval and add type in name in a switch case like <- better but painfull
// #define NUMBER_MIN		// Max value for this numeric type
// #define NUMBER_MAX		// Min value for this numeric type

// // Idea 1: List of all types and write any corresponding function 
// typedef  number_t;		// Standard size numeric type used for weights and activations
// typedef  long_number_t;	// Long numeric type used for intermediate results

#define NUMBER_MIN_INT16_T -32768
#define NUMBER_MAX_INT16_T 32767

static inline int32_t min_int16_t(
    int32_t a,
    int32_t b) {
	if (a <= b)
		return a;
	return b;
}

static inline int32_t max_int16_t(
    int32_t a,
    int32_t b) {
	if (a >= b)
		return a;
	return b;
}

static inline int32_t scale_number_t_int16_t(
  int32_t number, int scale_factor, round_mode_t round_mode) {


  if (scale_factor <= 0) {
#ifdef TRAPV_SHIFT
    // Check for possible overflow of left shift
    if (number > INT32_MAX >> -scale_factor) {
      fprintf(stderr,
              "Error: scale() overflow, number=%d, scale_factor=%d, limit=%d\n",
              number,
              scale_factor,
              INT16_MAX >> -scale_factor);
      assert(number <= INT32_MAX >> -scale_factor);
    }
#endif
    // No rounding to apply when shifting left
    return number << - scale_factor;
  } else {
    if (round_mode == ROUND_MODE_NEAREST) {
      number += (1 << (scale_factor - 1)); // +0.5 in fixed-point
    }
    return number >> scale_factor;
  }
}
static inline int16_t clamp_to_number_t_int16_t(
  int32_t number) {
	return (int16_t) max_int16_t(
      NUMBER_MIN_INT16_T,
      min_int16_t(
        NUMBER_MAX_INT16_T, number));
}
static inline int16_t scale_and_clamp_to_number_t_int16_t(
  int32_t number, int scale_factor, round_mode_t round_mode) {
#ifdef WITH_CMSIS_NN
  // Not really CMSIS-NN but use SSAT anyway
  if (scale_factor <= 0) {
    // No rounding to apply when shifting left
    return __SSAT(number << - scale_factor, sizeof(int16_t) * 8);
  } else {
    if (round_mode == ROUND_MODE_NEAREST) {
      number += (1 << (scale_factor - 1)); // +0.5 in fixed-point
    }
    return __SSAT(number >> scale_factor, sizeof(int16_t) * 8);
  }
#else
  number = scale_number_t_int16_t(number, scale_factor, round_mode);
  return clamp_to_number_t_int16_t(number);
#endif
}

#define NUMBER_MIN_INT32_T -2147483648
#define NUMBER_MAX_INT32_T 2147483647

static inline int64_t min_int32_t(
    int64_t a,
    int64_t b) {
	if (a <= b)
		return a;
	return b;
}

static inline int64_t max_int32_t(
    int64_t a,
    int64_t b) {
	if (a >= b)
		return a;
	return b;
}

static inline int64_t scale_number_t_int32_t(
  int64_t number, int scale_factor, round_mode_t round_mode) {


  if (scale_factor <= 0) {
#ifdef TRAPV_SHIFT
    // Check for possible overflow of left shift
    if (number > INT64_MAX >> -scale_factor) {
      fprintf(stderr,
              "Error: scale() overflow, number=%ld, scale_factor=%d, limit=%d\n",
              number,
              scale_factor,
              INT16_MAX >> -scale_factor);
      assert(number <= INT64_MAX >> -scale_factor);
    }
#endif
    // No rounding to apply when shifting left
    return number << - scale_factor;
  } else {
    if (round_mode == ROUND_MODE_NEAREST) {
      number += (1 << (scale_factor - 1)); // +0.5 in fixed-point
    }
    return number >> scale_factor;
  }
}
static inline int32_t clamp_to_number_t_int32_t(
  int64_t number) {
	return (int32_t) max_int32_t(
      NUMBER_MIN_INT32_T,
      min_int32_t(
        NUMBER_MAX_INT32_T, number));
}
static inline int32_t scale_and_clamp_to_number_t_int32_t(
  int64_t number, int scale_factor, round_mode_t round_mode) {
#ifdef WITH_CMSIS_NN
  // Not really CMSIS-NN but use SSAT anyway
  if (scale_factor <= 0) {
    // No rounding to apply when shifting left
    return __SSAT(number << - scale_factor, sizeof(int32_t) * 8);
  } else {
    if (round_mode == ROUND_MODE_NEAREST) {
      number += (1 << (scale_factor - 1)); // +0.5 in fixed-point
    }
    return __SSAT(number >> scale_factor, sizeof(int32_t) * 8);
  }
#else
  number = scale_number_t_int32_t(number, scale_factor, round_mode);
  return clamp_to_number_t_int32_t(number);
#endif
}




static inline void int64_t_to_float(int64_t * tabint, float * tabfloat, long tabdim, int scale_factor){
  for (int i=0; i<tabdim; i++){
    tabfloat[i] = (float)tabint[i] / (1<<scale_factor);
  }
}

static inline void int32_t_to_float(int32_t * tabint, float * tabfloat, long tabdim, int scale_factor){
  for (int i=0; i<tabdim; i++){
    tabfloat[i] = (float)tabint[i] / (1<<scale_factor);
  }
}

static inline void int16_t_to_float(int16_t * tabint, float * tabfloat, long tabdim, int scale_factor){
  for (int i=0; i<tabdim; i++){
    tabfloat[i] = ((float)tabint[i]) / (1<<scale_factor);
  }
}

static inline void int8_t_to_float(int8_t * tabint, float * tabfloat, long tabdim, int scale_factor){
  for (int i=0; i<tabdim; i++){
    tabfloat[i] = ((float)tabint[i]) / (1<<scale_factor);
  }
}
#endif //__NUMBER_H__

#ifdef __cplusplus
} // extern "C"
#endif
/**
  ******************************************************************************
  * @file    conv1d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 january 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _CONV1D_36_H_
#define _CONV1D_36_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS      1
#define INPUT_SAMPLES       16000
#define CONV_FILTERS        8
#define CONV_KERNEL_SIZE    20
#define CONV_STRIDE         10

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

typedef int16_t conv1d_36_output_type[CONV_OUTSAMPLES][CONV_FILTERS];

#if 0
void conv1d_36(
  const number_t input[INPUT_SAMPLES][INPUT_CHANNELS],                    // IN
  const number_t kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS],  // IN

  const number_t bias[CONV_FILTERS],						                          // IN

  number_t output[CONV_OUTSAMPLES][CONV_FILTERS]);                       // OUT
#endif

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES

#endif//_CONV1D_36_H_
/**
  ******************************************************************************
  * @file    conv.cc
  * @author  Sébastien Bilavarn, LEAT, CNRS, Université Côte d'Azur, France
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "conv1d_36.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#elif defined(WITH_NMSIS_NN)
#include "riscv_nnfunctions.h"
#endif

#define INPUT_CHANNELS      1
#define INPUT_SAMPLES       16000
#define CONV_FILTERS        8
#define CONV_KERNEL_SIZE    20
#define CONV_STRIDE         10
#define CONV_GROUPS         1
#define CHANNELS_PER_GROUP  (INPUT_CHANNELS / CONV_GROUPS)
#define FILTERS_PER_GROUP   (CONV_FILTERS / CONV_GROUPS)

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

#define ACTIVATION_RELU

// For fixed point quantization
#define WEIGHTS_SCALE_FACTOR 7
#define BIASES_SCALE_FACTOR 7
#define TMP_SCALE_FACTOR 7
#define INPUT_SCALE_FACTOR 7
#define OUTPUT_SCALE_FACTOR 7
#define OUTPUT_ROUND_MODE ROUND_MODE_FLOOR
#define NUMBER_T int16_t
#define LONG_NUMBER_T int32_t


static inline void conv1d_36(
  const NUMBER_T input[INPUT_SAMPLES][INPUT_CHANNELS],                    // IN
  const NUMBER_T kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS / CONV_GROUPS],  // IN

  const NUMBER_T bias[CONV_FILTERS],						                          // IN

  NUMBER_T output[CONV_OUTSAMPLES][CONV_FILTERS]) {                       // OUT

#if !defined(WITH_CMSIS_NN) && !defined(WITH_NMSIS_NN)
  unsigned short pos_x, z, k; 	// loop indexes for output volume
  unsigned short x;
  int input_x;
  LONG_NUMBER_T output_acc;

  for (pos_x = 0; pos_x < CONV_OUTSAMPLES; pos_x++) { 
    for (k = 0; k < CONV_FILTERS; k++) { 
      output_acc = 0;

      for (x = 0; x < CONV_KERNEL_SIZE; x++) {
        input_x = pos_x * CONV_STRIDE - ZEROPADDING_LEFT + x;

        if (input_x >= 0 && input_x < INPUT_SAMPLES) { // ZeroPadding1D
          for (z = 0; z < INPUT_CHANNELS / CONV_GROUPS; z++) {
            output_acc += (LONG_NUMBER_T)input[input_x][z + (k / FILTERS_PER_GROUP) * CHANNELS_PER_GROUP] * (LONG_NUMBER_T)kernel[k][x][z];
          }
        }
      }

    // Scale for possible additional precision of bias
    output_acc = scale(NUMBER_T, output_acc, WEIGHTS_SCALE_FACTOR - TMP_SCALE_FACTOR, OUTPUT_ROUND_MODE);

    // Scale bias to match accumulator
    output_acc += scale(NUMBER_T, (LONG_NUMBER_T)bias[k], BIASES_SCALE_FACTOR - TMP_SCALE_FACTOR - INPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);

      
#ifdef ACTIVATION_LINEAR
      output[pos_x][k] = scale_and_clamp_to(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
#elif defined(ACTIVATION_RELU) || defined(ACTIVATION_RELU6)
      // Activation function: ReLU
      if (output_acc < 0) {
        output[pos_x][k] = 0;
      } else {
#if defined(ACTIVATION_RELU6)
        if (output_acc > scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE)) {
          output_acc = scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE);
        }
#endif
        output[pos_x][k] = scale_and_clamp_to(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
      }
#else
#error "Unsupported activation function"
#endif
    }
  }

#else

#if BIASES_SCALE_FACTOR > WEIGHTS_SCALE_FACTOR
#error "CMSIS-NN does not support BIASES_SCALE_FACTOR larger than WEIGHTS_SCALE_FACTOR"
#endif

  static q15_t bufferA[2*INPUT_CHANNELS*CONV_KERNEL_SIZE];
#if INPUT_CHANNELS % 2 == 0 && CONV_FILTERS % 2 == 0
#ifdef WITH_CMSIS_NN
  arm_convolve_HWC_q15_fast_nonsquare(
#elif defined(WITH_NMSIS_NN)
  riscv_convolve_HWC_q15_fast_nonsquare(
#endif
#else
#ifdef WITH_CMSIS_NN
  arm_convolve_HWC_q15_basic_nonsquare(
#elif defined(WITH_NMSIS_NN)
  riscv_convolve_HWC_q15_basic_nonsquare(
#endif
#endif
                                      (q15_t*)input, //Im_in
                                      INPUT_SAMPLES, //dim_im_in_x
                                      1, //dim_im_in_y
                                      INPUT_CHANNELS, //ch_im_in
                                      (q15_t*)kernel, //wt
                                      CONV_FILTERS, //ch_im_out
                                      CONV_KERNEL_SIZE, //dim_kernel_x
                                      1, //dim_kernel_y
                                      ZEROPADDING_LEFT, //padding_x, left and right must be equal
                                      0, //padding_y
                                      CONV_STRIDE, //stride_x
                                      1, //stride_y
                                      (q15_t*)bias, //bias
                                      INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - BIASES_SCALE_FACTOR, //bias_shift
                                      INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, //out_shift
                                      (q15_t*)output, //Im_out
                                      CONV_OUTSAMPLES, //dim_im_out_x
                                      1, //dim_im_out_y
                                      bufferA, //bufferA
                                      NULL //bufferB, unused
                                      );
#ifdef ACTIVATION_RELU
#ifdef WITH_CMSIS_NN
  arm_relu_q15((q15_t*)output, CONV_FILTERS * CONV_OUTSAMPLES);
#elif defined(WITH_NMSIS_NN)
  riscv_relu_q15((q15_t*)output, CONV_FILTERS * CONV_OUTSAMPLES);
#endif
#elif !defined(ACTIVATION_LINEAR)
#error "Unsupported activation with CMSIS-NN"
#endif


#endif
}

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef CONV_GROUPS
#undef CHANNELS_PER_GROUP
#undef FILTERS_PER_GROUP
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES
#undef ACTIVATION_RELU
#undef WEIGHTS_SCALE_FACTOR
#undef BIASES_SCALE_FACTOR
#undef TMP_SCALE_FACTOR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
#undef NUMBER_T
#undef LONG_NUMBER_T
/**
  ******************************************************************************
  * @file    weights/conv1d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

#define INPUT_CHANNELS    1
#define CONV_FILTERS      8
#define CONV_KERNEL_SIZE  20
#define CONV_GROUPS       1


const int16_t  conv1d_36_bias[CONV_FILTERS] = {-12, -22, -14, -35, -24, -25, -16, -26}
;

const int16_t  conv1d_36_kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS / CONV_GROUPS] = {{{-24}
, {21}
, {17}
, {-7}
, {40}
, {-34}
, {-27}
, {19}
, {-45}
, {-31}
, {68}
, {3}
, {0}
, {38}
, {-33}
, {-54}
, {30}
, {-23}
, {-47}
, {89}
}
, {{-34}
, {-53}
, {-12}
, {-29}
, {-1}
, {-21}
, {-48}
, {-22}
, {10}
, {21}
, {19}
, {9}
, {7}
, {29}
, {71}
, {35}
, {18}
, {19}
, {36}
, {53}
}
, {{-36}
, {9}
, {-50}
, {-32}
, {21}
, {-35}
, {-14}
, {47}
, {39}
, {7}
, {7}
, {50}
, {16}
, {-20}
, {17}
, {6}
, {2}
, {-7}
, {18}
, {2}
}
, {{-12}
, {-17}
, {-6}
, {-35}
, {37}
, {57}
, {-12}
, {-12}
, {-26}
, {-59}
, {0}
, {74}
, {32}
, {-14}
, {-26}
, {-34}
, {-19}
, {-7}
, {20}
, {36}
}
, {{-22}
, {-51}
, {-59}
, {6}
, {-11}
, {37}
, {33}
, {25}
, {28}
, {24}
, {23}
, {-45}
, {-48}
, {-35}
, {-4}
, {18}
, {16}
, {9}
, {-17}
, {1}
}
, {{-7}
, {-5}
, {10}
, {-1}
, {-22}
, {-4}
, {18}
, {43}
, {-33}
, {-48}
, {27}
, {41}
, {11}
, {-44}
, {-2}
, {3}
, {21}
, {34}
, {-47}
, {-1}
}
, {{34}
, {19}
, {24}
, {-44}
, {-30}
, {-34}
, {27}
, {51}
, {19}
, {4}
, {-32}
, {-24}
, {10}
, {25}
, {-12}
, {-38}
, {-32}
, {-15}
, {14}
, {40}
}
, {{24}
, {-50}
, {-2}
, {-20}
, {-6}
, {26}
, {14}
, {25}
, {-25}
, {-21}
, {-34}
, {-70}
, {27}
, {39}
, {-14}
, {23}
, {-41}
, {-58}
, {-31}
, {39}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_GROUPS
/**
  ******************************************************************************
  * @file    maxpool1d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 january 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _MAX_POOLING1D_27_H_
#define _MAX_POOLING1D_27_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS  8
#define INPUT_SAMPLES   1599
#define POOL_SIZE       2
#define POOL_STRIDE     2
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

typedef int16_t max_pooling1d_27_output_type[POOL_LENGTH][INPUT_CHANNELS];

#if 0
void max_pooling1d_27(
  const number_t input[INPUT_SAMPLES][INPUT_CHANNELS], 	    // IN
  number_t output[POOL_LENGTH][INPUT_CHANNELS]); 	// OUT
#endif

#undef INPUT_CHANNELS  
#undef INPUT_SAMPLES
#undef POOL_SIZE
#undef POOL_STRIDE
#undef POOL_PAD
#undef POOL_LENGTH

#endif//_MAX_POOLING1D_27_H_
/**
  ******************************************************************************
  * @file    maxpool.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "max_pooling1d_27.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#endif

#define INPUT_CHANNELS  8
#define INPUT_SAMPLES   1599
#define POOL_SIZE       2
#define POOL_STRIDE     2
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

#define ACTIVATION_LINEAR

// For fixed point quantization
#define INPUT_SCALE_FACTOR 7
#define OUTPUT_SCALE_FACTOR 7
#define OUTPUT_ROUND_MODE ROUND_MODE_FLOOR
#define NUMBER_T int16_t
#define LONG_NUMBER_T int32_t


static inline void max_pooling1d_27(
  const NUMBER_T input[INPUT_SAMPLES][INPUT_CHANNELS], 	    // IN
  NUMBER_T output[POOL_LENGTH][INPUT_CHANNELS]) {	// OUT

  unsigned short pos_x, k; 	// loop indexes for output volume
  unsigned int x;
  static LONG_NUMBER_T max[INPUT_CHANNELS];

  for (pos_x = 0; pos_x < POOL_LENGTH; pos_x++) {
    for (k = 0; k < INPUT_CHANNELS; k++) {
#ifdef ACTIVATION_LINEAR
      max[k] = input[pos_x*POOL_STRIDE][k];
      x = 1;
#elif defined(ACTIVATION_RELU)
      max[k] = 0;
      x = 0;
#else
#error "Unsupported activation function"
#endif
    }

    for (; x < POOL_SIZE; x++) {
      for (k = 0; k < INPUT_CHANNELS; k++) {
        if (max[k] < input[(pos_x * POOL_STRIDE) + x][k])
          max[k] = input[(pos_x * POOL_STRIDE) + x][k];
      }
    }

    for (k = 0; k < INPUT_CHANNELS; k++) {
      output[pos_x][k] = scale_and_clamp_to(NUMBER_T, max[k], INPUT_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
    }
  }
}

#undef INPUT_CHANNELS  
#undef INPUT_SAMPLES
#undef POOL_SIZE
#undef POOL_STRIDE
#undef POOL_PAD
#undef POOL_LENGTH
#undef ACTIVATION_LINEAR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
#undef NUMBER_T
#undef LONG_NUMBER_T
/**
  ******************************************************************************
  * @file    conv1d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 january 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _CONV1D_37_H_
#define _CONV1D_37_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS      8
#define INPUT_SAMPLES       799
#define CONV_FILTERS        16
#define CONV_KERNEL_SIZE    8
#define CONV_STRIDE         4

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

typedef int16_t conv1d_37_output_type[CONV_OUTSAMPLES][CONV_FILTERS];

#if 0
void conv1d_37(
  const number_t input[INPUT_SAMPLES][INPUT_CHANNELS],                    // IN
  const number_t kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS],  // IN

  const number_t bias[CONV_FILTERS],						                          // IN

  number_t output[CONV_OUTSAMPLES][CONV_FILTERS]);                       // OUT
#endif

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES

#endif//_CONV1D_37_H_
/**
  ******************************************************************************
  * @file    conv.cc
  * @author  Sébastien Bilavarn, LEAT, CNRS, Université Côte d'Azur, France
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "conv1d_37.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#elif defined(WITH_NMSIS_NN)
#include "riscv_nnfunctions.h"
#endif

#define INPUT_CHANNELS      8
#define INPUT_SAMPLES       799
#define CONV_FILTERS        16
#define CONV_KERNEL_SIZE    8
#define CONV_STRIDE         4
#define CONV_GROUPS         1
#define CHANNELS_PER_GROUP  (INPUT_CHANNELS / CONV_GROUPS)
#define FILTERS_PER_GROUP   (CONV_FILTERS / CONV_GROUPS)

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

#define ACTIVATION_RELU

// For fixed point quantization
#define WEIGHTS_SCALE_FACTOR 7
#define BIASES_SCALE_FACTOR 7
#define TMP_SCALE_FACTOR 7
#define INPUT_SCALE_FACTOR 7
#define OUTPUT_SCALE_FACTOR 7
#define OUTPUT_ROUND_MODE ROUND_MODE_FLOOR
#define NUMBER_T int16_t
#define LONG_NUMBER_T int32_t


static inline void conv1d_37(
  const NUMBER_T input[INPUT_SAMPLES][INPUT_CHANNELS],                    // IN
  const NUMBER_T kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS / CONV_GROUPS],  // IN

  const NUMBER_T bias[CONV_FILTERS],						                          // IN

  NUMBER_T output[CONV_OUTSAMPLES][CONV_FILTERS]) {                       // OUT

#if !defined(WITH_CMSIS_NN) && !defined(WITH_NMSIS_NN)
  unsigned short pos_x, z, k; 	// loop indexes for output volume
  unsigned short x;
  int input_x;
  LONG_NUMBER_T output_acc;

  for (pos_x = 0; pos_x < CONV_OUTSAMPLES; pos_x++) { 
    for (k = 0; k < CONV_FILTERS; k++) { 
      output_acc = 0;

      for (x = 0; x < CONV_KERNEL_SIZE; x++) {
        input_x = pos_x * CONV_STRIDE - ZEROPADDING_LEFT + x;

        if (input_x >= 0 && input_x < INPUT_SAMPLES) { // ZeroPadding1D
          for (z = 0; z < INPUT_CHANNELS / CONV_GROUPS; z++) {
            output_acc += (LONG_NUMBER_T)input[input_x][z + (k / FILTERS_PER_GROUP) * CHANNELS_PER_GROUP] * (LONG_NUMBER_T)kernel[k][x][z];
          }
        }
      }

    // Scale for possible additional precision of bias
    output_acc = scale(NUMBER_T, output_acc, WEIGHTS_SCALE_FACTOR - TMP_SCALE_FACTOR, OUTPUT_ROUND_MODE);

    // Scale bias to match accumulator
    output_acc += scale(NUMBER_T, (LONG_NUMBER_T)bias[k], BIASES_SCALE_FACTOR - TMP_SCALE_FACTOR - INPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);

      
#ifdef ACTIVATION_LINEAR
      output[pos_x][k] = scale_and_clamp_to(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
#elif defined(ACTIVATION_RELU) || defined(ACTIVATION_RELU6)
      // Activation function: ReLU
      if (output_acc < 0) {
        output[pos_x][k] = 0;
      } else {
#if defined(ACTIVATION_RELU6)
        if (output_acc > scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE)) {
          output_acc = scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE);
        }
#endif
        output[pos_x][k] = scale_and_clamp_to(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
      }
#else
#error "Unsupported activation function"
#endif
    }
  }

#else

#if BIASES_SCALE_FACTOR > WEIGHTS_SCALE_FACTOR
#error "CMSIS-NN does not support BIASES_SCALE_FACTOR larger than WEIGHTS_SCALE_FACTOR"
#endif

  static q15_t bufferA[2*INPUT_CHANNELS*CONV_KERNEL_SIZE];
#if INPUT_CHANNELS % 2 == 0 && CONV_FILTERS % 2 == 0
#ifdef WITH_CMSIS_NN
  arm_convolve_HWC_q15_fast_nonsquare(
#elif defined(WITH_NMSIS_NN)
  riscv_convolve_HWC_q15_fast_nonsquare(
#endif
#else
#ifdef WITH_CMSIS_NN
  arm_convolve_HWC_q15_basic_nonsquare(
#elif defined(WITH_NMSIS_NN)
  riscv_convolve_HWC_q15_basic_nonsquare(
#endif
#endif
                                      (q15_t*)input, //Im_in
                                      INPUT_SAMPLES, //dim_im_in_x
                                      1, //dim_im_in_y
                                      INPUT_CHANNELS, //ch_im_in
                                      (q15_t*)kernel, //wt
                                      CONV_FILTERS, //ch_im_out
                                      CONV_KERNEL_SIZE, //dim_kernel_x
                                      1, //dim_kernel_y
                                      ZEROPADDING_LEFT, //padding_x, left and right must be equal
                                      0, //padding_y
                                      CONV_STRIDE, //stride_x
                                      1, //stride_y
                                      (q15_t*)bias, //bias
                                      INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - BIASES_SCALE_FACTOR, //bias_shift
                                      INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, //out_shift
                                      (q15_t*)output, //Im_out
                                      CONV_OUTSAMPLES, //dim_im_out_x
                                      1, //dim_im_out_y
                                      bufferA, //bufferA
                                      NULL //bufferB, unused
                                      );
#ifdef ACTIVATION_RELU
#ifdef WITH_CMSIS_NN
  arm_relu_q15((q15_t*)output, CONV_FILTERS * CONV_OUTSAMPLES);
#elif defined(WITH_NMSIS_NN)
  riscv_relu_q15((q15_t*)output, CONV_FILTERS * CONV_OUTSAMPLES);
#endif
#elif !defined(ACTIVATION_LINEAR)
#error "Unsupported activation with CMSIS-NN"
#endif


#endif
}

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef CONV_GROUPS
#undef CHANNELS_PER_GROUP
#undef FILTERS_PER_GROUP
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES
#undef ACTIVATION_RELU
#undef WEIGHTS_SCALE_FACTOR
#undef BIASES_SCALE_FACTOR
#undef TMP_SCALE_FACTOR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
#undef NUMBER_T
#undef LONG_NUMBER_T
/**
  ******************************************************************************
  * @file    weights/conv1d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

#define INPUT_CHANNELS    8
#define CONV_FILTERS      16
#define CONV_KERNEL_SIZE  8
#define CONV_GROUPS       1


const int16_t  conv1d_37_bias[CONV_FILTERS] = {-20, -2, 13, -2, -2, 9, -35, 6, -58, 5, -57, 4, -83, -28, -15, 1}
;

const int16_t  conv1d_37_kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS / CONV_GROUPS] = {{{13, 7, -5, -6, -24, -29, 4, -15}
, {-23, 0, -16, -8, -32, -21, -8, -19}
, {-32, -19, -18, -25, 10, -22, -12, 10}
, {-20, -5, -26, -30, -7, -18, -2, -29}
, {-21, 9, -24, 5, -28, 11, -21, -10}
, {-18, -28, 6, 10, -4, -13, -18, -22}
, {-27, -21, -5, -19, -26, 3, -2, -4}
, {5, -7, 12, -23, -6, -14, -10, -16}
}
, {{-17, -73, -60, -19, -36, -46, -65, -47}
, {-76, -54, -52, -56, 2, -61, -80, -15}
, {-64, -1, 9, -83, -5, -70, -84, -57}
, {-37, -17, -40, -24, -62, -66, -13, -47}
, {26, 8, -9, -4, -43, -23, 14, -8}
, {31, 4, 5, 22, 27, 9, 18, 19}
, {75, 12, 2, 53, -12, 32, 29, 36}
, {57, 44, 9, 39, 7, 50, 7, 15}
}
, {{-53, -30, -34, -18, -9, 8, 32, -1}
, {-3, -44, -25, 10, -38, -20, 15, 49}
, {-1, -18, -51, 59, 9, 1, -19, 38}
, {-24, -48, -41, 10, -1, -14, 43, 28}
, {-25, -7, -32, 27, -30, -26, 34, 4}
, {1, -50, -69, 54, 7, -27, 5, -12}
, {-18, -2, -58, 29, 3, -30, 20, -11}
, {-37, -6, -60, 55, 19, 9, 59, 26}
}
, {{47, -10, 26, 22, 15, 45, -21, -28}
, {45, 6, -20, 28, 11, 41, 5, 33}
, {17, 43, 29, -12, 21, 66, 15, 7}
, {-2, 22, 13, 17, -12, -13, 33, -31}
, {18, -36, -9, -37, 25, 1, -1, 1}
, {-24, -26, -19, -74, 10, -7, -76, -30}
, {-58, -52, -41, -141, -7, -15, -116, -105}
, {-50, 4, -63, -102, -59, -26, -91, -82}
}
, {{-31, 9, 29, -22, 17, -38, 10, -49}
, {45, 50, 41, 3, -9, -39, -13, -37}
, {48, -4, -26, 17, -81, -20, -2, -71}
, {47, 44, 23, 5, -45, -111, -22, -20}
, {25, 25, 14, 3, -27, -52, -26, -35}
, {-41, 27, -6, -27, -66, -39, 15, -94}
, {12, 26, 29, 52, 1, -33, 10, -44}
, {12, -7, 24, 33, 10, -2, 22, 9}
}
, {{35, -7, 12, 17, -16, -10, -5, 27}
, {19, -28, 41, -25, -22, 3, -27, 59}
, {42, -29, 19, 4, -8, -5, -23, -4}
, {13, -53, 47, 7, 35, 12, -1, 23}
, {60, -62, -5, 8, 36, 17, -27, -6}
, {68, -39, 21, 1, -9, 16, -40, 37}
, {21, 17, -17, 0, -12, 7, -7, 2}
, {24, 14, 9, 25, -9, -27, -31, 31}
}
, {{-20, -81, -62, -52, -16, -23, -1, -1}
, {10, -28, -37, 17, -67, 6, -17, -47}
, {0, 6, -3, -39, -69, -32, -3, -21}
, {-10, 15, -10, -16, -42, -17, -22, -34}
, {-12, -74, -25, -18, 23, 2, -51, -1}
, {-78, 37, 24, -9, 36, 5, 7, -47}
, {-37, -104, -59, -51, -65, -66, 36, -18}
, {-94, -22, -63, -43, -37, -37, -50, -17}
}
, {{-82, -33, -16, -36, -66, -79, -48, -128}
, {-62, -8, -26, -81, -124, 0, -102, -128}
, {-92, -31, -53, -67, -35, -128, -40, -67}
, {-15, 21, 5, -25, 19, -2, -37, -54}
, {-86, -84, -92, -70, -8, -42, -37, -32}
, {-27, 0, 7, -41, -10, -46, -55, -79}
, {90, -12, 41, 8, -32, 30, -16, -63}
, {-38, 107, 26, -9, -119, -53, -3, -60}
}
, {{-27, -51, -36, -39, -41, -45, -20, -26}
, {-44, -76, -101, -17, -25, -4, -28, -24}
, {-49, -71, -27, -27, -46, -33, -26, -63}
, {-32, -69, -41, -25, -40, -51, -18, -15}
, {-24, -21, -46, -47, -14, -33, -14, -15}
, {-49, -99, -10, -38, -5, 27, -49, -27}
, {-43, -52, -76, -5, 18, -13, -25, -21}
, {-15, -41, -37, 3, 1, 33, -8, -42}
}
, {{11, 36, 32, -21, 29, -14, -13, -16}
, {-28, -11, -1, 7, 17, -7, -18, -6}
, {-5, -51, -64, -4, 17, 14, 23, -28}
, {-3, -17, -68, 27, 50, 17, 18, 21}
, {-12, -16, -27, 27, 26, 14, 38, 29}
, {-43, 22, -27, -1, 20, 11, 52, 0}
, {-33, -9, -17, -4, 1, 19, 66, 43}
, {-29, -8, -57, -23, 12, 1, 3, -11}
}
, {{21, 8, 21, -3, 18, 0, -2, 0}
, {36, 65, -32, -15, -17, 43, 32, -26}
, {-22, 37, 35, -26, -21, -40, 26, 19}
, {-28, 39, 9, -4, 3, 36, 4, -18}
, {-33, 24, -19, -1, 6, 42, 3, 0}
, {30, 33, -36, -45, -4, 21, -25, -27}
, {-22, 37, 11, -27, 11, 33, -3, -4}
, {-23, -20, -13, 11, -35, 26, -18, -26}
}
, {{-6, -84, -25, -33, 35, -23, -19, 21}
, {-82, -68, 19, -31, 58, -7, -33, -8}
, {37, 100, 62, 24, -25, -12, 1, -23}
, {8, -40, -23, -10, -51, -34, 21, -128}
, {-38, -53, -1, -30, 66, -66, 25, -14}
, {29, 33, 58, 4, 8, -86, -22, -13}
, {-6, -6, -21, -26, 21, -77, 24, -85}
, {-9, -39, 28, -3, 102, 7, 14, -30}
}
, {{-22, -111, -37, -30, -22, 7, -2, -12}
, {-14, -50, -47, -10, -45, -10, -51, -31}
, {-27, -41, -8, -15, -81, -48, -20, -48}
, {3, -34, -48, 10, -41, -19, -6, -45}
, {18, 3, -65, -3, -35, 3, -12, -19}
, {-67, 42, 44, 21, -18, 1, -45, -40}
, {-49, -39, -72, -47, -23, -10, -67, -11}
, {-59, -67, -48, -52, -11, -1, -49, -18}
}
, {{46, 0, 17, 14, -91, -28, -37, 28}
, {-37, -49, -5, -67, 45, -15, -30, 1}
, {56, -82, -5, 32, 2, -29, -4, -5}
, {0, -18, 5, 39, -28, -40, -26, -24}
, {-89, -9, 27, 2, -20, -66, -22, -13}
, {105, -25, 5, -1, -70, -30, -52, 22}
, {1, -53, -63, -107, 64, -22, 17, 35}
, {32, -26, 15, 5, -57, 14, -72, -49}
}
, {{0, -30, -2, -17, 4, 5, -5, -20}
, {-13, -7, 3, -3, -22, -13, -19, 6}
, {-32, -32, 3, 3, 6, -14, 8, -30}
, {-1, 10, -27, -4, -10, 8, -2, -1}
, {-9, -5, -14, 10, -8, -20, -21, -16}
, {-27, -9, 4, -11, -33, -6, -20, -26}
, {-27, -30, -17, -18, -27, 0, -31, -12}
, {-9, -2, -30, -24, -31, -23, -25, -1}
}
, {{12, -9, -4, 28, 55, 55, 57, 65}
, {-7, -42, -8, 36, 27, 39, 32, 15}
, {-34, -36, 44, 22, 53, -24, 14, 23}
, {12, -47, 9, -16, -53, -87, 32, -4}
, {-78, -41, -83, -82, -59, -38, -32, -55}
, {-37, -114, -49, -64, -78, -110, -92, -70}
, {-158, -51, -64, -153, -110, -71, -197, -54}
, {-108, -34, 77, -147, -40, 3, -76, -12}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_GROUPS
/**
  ******************************************************************************
  * @file    maxpool1d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 january 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _MAX_POOLING1D_28_H_
#define _MAX_POOLING1D_28_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS  16
#define INPUT_SAMPLES   198
#define POOL_SIZE       2
#define POOL_STRIDE     2
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

typedef int16_t max_pooling1d_28_output_type[POOL_LENGTH][INPUT_CHANNELS];

#if 0
void max_pooling1d_28(
  const number_t input[INPUT_SAMPLES][INPUT_CHANNELS], 	    // IN
  number_t output[POOL_LENGTH][INPUT_CHANNELS]); 	// OUT
#endif

#undef INPUT_CHANNELS  
#undef INPUT_SAMPLES
#undef POOL_SIZE
#undef POOL_STRIDE
#undef POOL_PAD
#undef POOL_LENGTH

#endif//_MAX_POOLING1D_28_H_
/**
  ******************************************************************************
  * @file    maxpool.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "max_pooling1d_28.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#endif

#define INPUT_CHANNELS  16
#define INPUT_SAMPLES   198
#define POOL_SIZE       2
#define POOL_STRIDE     2
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

#define ACTIVATION_LINEAR

// For fixed point quantization
#define INPUT_SCALE_FACTOR 7
#define OUTPUT_SCALE_FACTOR 7
#define OUTPUT_ROUND_MODE ROUND_MODE_FLOOR
#define NUMBER_T int16_t
#define LONG_NUMBER_T int32_t


static inline void max_pooling1d_28(
  const NUMBER_T input[INPUT_SAMPLES][INPUT_CHANNELS], 	    // IN
  NUMBER_T output[POOL_LENGTH][INPUT_CHANNELS]) {	// OUT

  unsigned short pos_x, k; 	// loop indexes for output volume
  unsigned int x;
  static LONG_NUMBER_T max[INPUT_CHANNELS];

  for (pos_x = 0; pos_x < POOL_LENGTH; pos_x++) {
    for (k = 0; k < INPUT_CHANNELS; k++) {
#ifdef ACTIVATION_LINEAR
      max[k] = input[pos_x*POOL_STRIDE][k];
      x = 1;
#elif defined(ACTIVATION_RELU)
      max[k] = 0;
      x = 0;
#else
#error "Unsupported activation function"
#endif
    }

    for (; x < POOL_SIZE; x++) {
      for (k = 0; k < INPUT_CHANNELS; k++) {
        if (max[k] < input[(pos_x * POOL_STRIDE) + x][k])
          max[k] = input[(pos_x * POOL_STRIDE) + x][k];
      }
    }

    for (k = 0; k < INPUT_CHANNELS; k++) {
      output[pos_x][k] = scale_and_clamp_to(NUMBER_T, max[k], INPUT_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
    }
  }
}

#undef INPUT_CHANNELS  
#undef INPUT_SAMPLES
#undef POOL_SIZE
#undef POOL_STRIDE
#undef POOL_PAD
#undef POOL_LENGTH
#undef ACTIVATION_LINEAR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
#undef NUMBER_T
#undef LONG_NUMBER_T
/**
  ******************************************************************************
  * @file    conv1d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 january 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _CONV1D_38_H_
#define _CONV1D_38_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS      16
#define INPUT_SAMPLES       99
#define CONV_FILTERS        32
#define CONV_KERNEL_SIZE    4
#define CONV_STRIDE         2

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

typedef int16_t conv1d_38_output_type[CONV_OUTSAMPLES][CONV_FILTERS];

#if 0
void conv1d_38(
  const number_t input[INPUT_SAMPLES][INPUT_CHANNELS],                    // IN
  const number_t kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS],  // IN

  const number_t bias[CONV_FILTERS],						                          // IN

  number_t output[CONV_OUTSAMPLES][CONV_FILTERS]);                       // OUT
#endif

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES

#endif//_CONV1D_38_H_
/**
  ******************************************************************************
  * @file    conv.cc
  * @author  Sébastien Bilavarn, LEAT, CNRS, Université Côte d'Azur, France
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "conv1d_38.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#elif defined(WITH_NMSIS_NN)
#include "riscv_nnfunctions.h"
#endif

#define INPUT_CHANNELS      16
#define INPUT_SAMPLES       99
#define CONV_FILTERS        32
#define CONV_KERNEL_SIZE    4
#define CONV_STRIDE         2
#define CONV_GROUPS         1
#define CHANNELS_PER_GROUP  (INPUT_CHANNELS / CONV_GROUPS)
#define FILTERS_PER_GROUP   (CONV_FILTERS / CONV_GROUPS)

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

#define ACTIVATION_RELU

// For fixed point quantization
#define WEIGHTS_SCALE_FACTOR 7
#define BIASES_SCALE_FACTOR 7
#define TMP_SCALE_FACTOR 7
#define INPUT_SCALE_FACTOR 7
#define OUTPUT_SCALE_FACTOR 7
#define OUTPUT_ROUND_MODE ROUND_MODE_FLOOR
#define NUMBER_T int16_t
#define LONG_NUMBER_T int32_t


static inline void conv1d_38(
  const NUMBER_T input[INPUT_SAMPLES][INPUT_CHANNELS],                    // IN
  const NUMBER_T kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS / CONV_GROUPS],  // IN

  const NUMBER_T bias[CONV_FILTERS],						                          // IN

  NUMBER_T output[CONV_OUTSAMPLES][CONV_FILTERS]) {                       // OUT

#if !defined(WITH_CMSIS_NN) && !defined(WITH_NMSIS_NN)
  unsigned short pos_x, z, k; 	// loop indexes for output volume
  unsigned short x;
  int input_x;
  LONG_NUMBER_T output_acc;

  for (pos_x = 0; pos_x < CONV_OUTSAMPLES; pos_x++) { 
    for (k = 0; k < CONV_FILTERS; k++) { 
      output_acc = 0;

      for (x = 0; x < CONV_KERNEL_SIZE; x++) {
        input_x = pos_x * CONV_STRIDE - ZEROPADDING_LEFT + x;

        if (input_x >= 0 && input_x < INPUT_SAMPLES) { // ZeroPadding1D
          for (z = 0; z < INPUT_CHANNELS / CONV_GROUPS; z++) {
            output_acc += (LONG_NUMBER_T)input[input_x][z + (k / FILTERS_PER_GROUP) * CHANNELS_PER_GROUP] * (LONG_NUMBER_T)kernel[k][x][z];
          }
        }
      }

    // Scale for possible additional precision of bias
    output_acc = scale(NUMBER_T, output_acc, WEIGHTS_SCALE_FACTOR - TMP_SCALE_FACTOR, OUTPUT_ROUND_MODE);

    // Scale bias to match accumulator
    output_acc += scale(NUMBER_T, (LONG_NUMBER_T)bias[k], BIASES_SCALE_FACTOR - TMP_SCALE_FACTOR - INPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);

      
#ifdef ACTIVATION_LINEAR
      output[pos_x][k] = scale_and_clamp_to(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
#elif defined(ACTIVATION_RELU) || defined(ACTIVATION_RELU6)
      // Activation function: ReLU
      if (output_acc < 0) {
        output[pos_x][k] = 0;
      } else {
#if defined(ACTIVATION_RELU6)
        if (output_acc > scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE)) {
          output_acc = scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE);
        }
#endif
        output[pos_x][k] = scale_and_clamp_to(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
      }
#else
#error "Unsupported activation function"
#endif
    }
  }

#else

#if BIASES_SCALE_FACTOR > WEIGHTS_SCALE_FACTOR
#error "CMSIS-NN does not support BIASES_SCALE_FACTOR larger than WEIGHTS_SCALE_FACTOR"
#endif

  static q15_t bufferA[2*INPUT_CHANNELS*CONV_KERNEL_SIZE];
#if INPUT_CHANNELS % 2 == 0 && CONV_FILTERS % 2 == 0
#ifdef WITH_CMSIS_NN
  arm_convolve_HWC_q15_fast_nonsquare(
#elif defined(WITH_NMSIS_NN)
  riscv_convolve_HWC_q15_fast_nonsquare(
#endif
#else
#ifdef WITH_CMSIS_NN
  arm_convolve_HWC_q15_basic_nonsquare(
#elif defined(WITH_NMSIS_NN)
  riscv_convolve_HWC_q15_basic_nonsquare(
#endif
#endif
                                      (q15_t*)input, //Im_in
                                      INPUT_SAMPLES, //dim_im_in_x
                                      1, //dim_im_in_y
                                      INPUT_CHANNELS, //ch_im_in
                                      (q15_t*)kernel, //wt
                                      CONV_FILTERS, //ch_im_out
                                      CONV_KERNEL_SIZE, //dim_kernel_x
                                      1, //dim_kernel_y
                                      ZEROPADDING_LEFT, //padding_x, left and right must be equal
                                      0, //padding_y
                                      CONV_STRIDE, //stride_x
                                      1, //stride_y
                                      (q15_t*)bias, //bias
                                      INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - BIASES_SCALE_FACTOR, //bias_shift
                                      INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, //out_shift
                                      (q15_t*)output, //Im_out
                                      CONV_OUTSAMPLES, //dim_im_out_x
                                      1, //dim_im_out_y
                                      bufferA, //bufferA
                                      NULL //bufferB, unused
                                      );
#ifdef ACTIVATION_RELU
#ifdef WITH_CMSIS_NN
  arm_relu_q15((q15_t*)output, CONV_FILTERS * CONV_OUTSAMPLES);
#elif defined(WITH_NMSIS_NN)
  riscv_relu_q15((q15_t*)output, CONV_FILTERS * CONV_OUTSAMPLES);
#endif
#elif !defined(ACTIVATION_LINEAR)
#error "Unsupported activation with CMSIS-NN"
#endif


#endif
}

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef CONV_GROUPS
#undef CHANNELS_PER_GROUP
#undef FILTERS_PER_GROUP
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES
#undef ACTIVATION_RELU
#undef WEIGHTS_SCALE_FACTOR
#undef BIASES_SCALE_FACTOR
#undef TMP_SCALE_FACTOR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
#undef NUMBER_T
#undef LONG_NUMBER_T
/**
  ******************************************************************************
  * @file    weights/conv1d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

#define INPUT_CHANNELS    16
#define CONV_FILTERS      32
#define CONV_KERNEL_SIZE  4
#define CONV_GROUPS       1


const int16_t  conv1d_38_bias[CONV_FILTERS] = {-5, 17, 33, -41, 25, -99, -24, 54, 32, -12, -3, -9, 39, -79, 4, 8, -37, 1, -76, -7, 38, 37, 40, -11, -370, 19, -10, 62, -25, 25, 58, 24}
;

const int16_t  conv1d_38_kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS / CONV_GROUPS] = {{{8, 2, -77, -3, -102, 19, -30, 24, -45, -41, 4, -127, -3, 19, 0, -39}
, {14, -33, -60, -3, -127, 6, -38, -68, -31, -33, -6, -110, -27, -67, -31, -59}
, {3, -9, 58, -58, -28, -32, -86, -26, 3, 46, -3, -86, -54, 17, 27, -5}
, {10, -39, 35, 39, -1, -47, 150, -130, -26, -3, -7, 54, 98, -93, 5, -76}
}
, {{-28, -32, -27, 40, -65, -43, -15, -101, -58, -24, -64, 24, 10, 40, -21, 7}
, {-9, 61, -169, -155, 43, 2, -4, 42, 5, -49, 13, 13, 53, 34, 10, -90}
, {-66, -10, -37, 88, -27, 50, -6, 29, -27, -2, 54, 11, -40, -73, -30, 46}
, {-35, 12, -36, -63, -70, -44, -71, -90, 76, 4, -46, -63, -30, 17, -13, -55}
}
, {{-12, -8, -45, -20, 4, -81, -121, -205, 22, 0, 18, 35, -61, 20, 7, -24}
, {-8, -53, -14, 14, -30, 27, -11, -24, -11, 9, -45, 132, -108, -27, 14, -9}
, {0, -3, -19, -33, 18, 8, -47, -52, 2, -22, -7, 182, -71, -9, -32, 59}
, {27, -27, -55, -53, 3, -26, -46, 66, 29, 7, -14, 78, -87, 28, 16, -136}
}
, {{-11, 43, -65, -52, -53, -65, -15, -27, 5, -129, -47, -53, -17, -25, 2, 53}
, {11, 3, -78, -3, 19, 43, -16, -29, -34, -58, -28, 6, -21, -37, 0, -39}
, {-6, -65, -63, 14, 9, -43, -4, 13, 1, -58, 10, -164, -2, -25, -4, 59}
, {-9, 34, -43, 17, 51, -31, -19, 50, -31, -95, 44, -81, 4, -50, 4, 50}
}
, {{-2, -17, -50, 10, -6, -11, 4, -52, 0, -28, -30, -50, -10, 27, 19, 13}
, {33, 12, -19, -5, 17, 11, 6, -55, -45, -19, 20, -119, -16, -25, 26, -144}
, {-14, 68, -100, 30, -9, 16, -47, 0, -111, -34, 1, -1, 7, 20, 12, -63}
, {-13, 18, -92, 22, -51, 48, 14, 42, -73, 11, 22, -75, -38, -8, -5, -16}
}
, {{10, -57, 44, -30, -18, -43, -28, 9, -25, 14, -64, 18, -17, 1, -1, 23}
, {-28, -4, -33, -21, -51, -58, -20, 5, 7, -144, -36, -29, 12, -24, -15, 52}
, {-15, 27, -39, -5, -40, -108, -25, -2, -15, -82, -60, -190, -21, -11, -28, -75}
, {-3, 14, 14, -23, -93, -35, -10, 5, -25, -25, -48, -127, 0, 10, 0, -13}
}
, {{12, -8, 41, 38, -107, 17, 49, -103, -27, 29, 8, 49, 19, -64, -7, -42}
, {-23, -84, -1, 20, -48, 7, -8, -35, -63, 44, -3, -44, -8, -112, 5, 73}
, {6, -1, 4, -45, 37, -68, -30, -67, -16, -86, -61, -40, 30, -148, 12, 31}
, {30, 60, -116, -65, 35, -37, 25, 108, 3, -152, -2, 42, 37, -18, 9, -150}
}
, {{-9, -5, 36, -104, -83, -18, -37, 49, -7, 30, 12, -196, -13, -48, -30, 23}
, {-11, -78, 8, -44, -98, -9, 11, -32, -12, 6, -12, -62, -10, -2, 5, -1}
, {5, -5, -32, -89, -54, 37, -7, 26, 18, -21, -71, -58, 9, 11, 2, -79}
, {-24, -7, 26, -82, -65, -13, -20, 78, 15, 8, -66, -108, 11, -3, -16, 0}
}
, {{19, 53, -39, -14, -9, -11, 39, -57, -5, -7, 7, -70, -16, 37, 12, 14}
, {10, -1, -27, 55, -65, -6, -11, -53, -38, -22, -57, -16, -5, 42, -7, 74}
, {-13, 69, -55, -16, 16, -57, -3, 84, -6, 17, -14, -61, 9, -43, 8, 24}
, {11, 10, -18, 11, 7, -24, -11, 22, 12, -56, 22, -146, -1, 19, -5, 40}
}
, {{-14, 5, -2, -9, -31, -49, 2, -23, -22, -5, -19, 4, 4, -18, -22, -30}
, {2, 6, -32, -22, -9, -3, -17, -4, 2, -4, -35, -4, -23, 4, 9, 3}
, {6, -22, -23, -16, -28, -7, -6, 3, -10, -28, -23, -41, 14, -7, -23, -24}
, {10, -29, -15, 12, -12, -17, 10, 5, -15, -26, -7, -21, -18, 0, -23, -20}
}
, {{-23, -51, 32, 22, -11, -76, -7, -47, -20, 6, -48, 81, -12, -11, -3, -71}
, {17, 38, 14, 7, 55, -18, 6, -38, -28, -30, -58, -35, -1, 31, -21, -142}
, {-22, 11, 8, -68, 20, 56, -15, -70, -14, -23, -43, -30, 6, -15, -3, -119}
, {1, -68, -66, 20, -92, -20, -9, -13, 0, -5, 76, -26, 30, -19, 1, -6}
}
, {{8, -36, -44, 28, -80, 45, 15, -12, 12, -18, -47, -17, -22, -12, 3, 58}
, {-15, -29, -65, -83, -60, -131, 13, -3, 8, -14, -88, 8, -30, -27, -20, -81}
, {3, -39, -1, -6, -4, -123, 9, -33, 12, 0, -57, -6, 8, 5, -12, 16}
, {-22, -4, -7, -13, -12, -79, 6, 0, -28, -55, -3, 11, -24, 9, -11, -76}
}
, {{-20, 45, -91, -51, -99, -32, 66, 65, -24, -57, -25, -5, 65, -15, 0, 29}
, {-4, -44, 54, -70, -68, -27, -23, -58, -15, 62, -16, -46, 19, -33, -10, -83}
, {-20, -24, 23, 18, 83, -12, -7, 9, 3, 6, -64, -29, 4, -31, 21, -17}
, {-14, 8, -11, 46, -60, 32, 1, -99, 20, -22, -63, 19, -33, 29, -17, 10}
}
, {{-17, -26, -29, -83, -53, -38, -9, -21, 10, -42, -89, -79, -28, 1, -29, -65}
, {-23, -27, -2, -31, -39, -59, 1, -4, 13, -28, -61, -114, 9, -15, -22, 29}
, {15, -5, -24, -8, -29, -69, -28, 4, 6, -55, 3, -58, 4, -14, -17, 27}
, {-21, 91, -84, -23, -129, -34, 24, 57, 12, -26, -43, -32, 1, 50, -26, -52}
}
, {{-4, -44, -95, -79, -50, -51, -16, -40, -7, 2, -40, -125, -25, 9, -21, 63}
, {-30, 8, -37, 70, -125, 47, 12, -17, 15, 33, -19, -8, 3, 10, -26, 95}
, {2, -32, -90, -32, -62, -65, 9, 1, -30, -48, -74, 32, -23, -15, 1, 66}
, {-28, 6, -42, -50, 22, -107, -6, -10, -2, -20, -9, -35, -39, -28, -17, -88}
}
, {{-4, -55, 13, -26, 29, 54, -58, -160, -12, -15, -96, 17, 6, 91, 4, 123}
, {-5, 32, -1, 27, -22, -57, 27, 51, -34, -26, -112, 37, 12, 14, -10, 24}
, {-7, 81, 25, -69, -76, -15, -40, -22, 50, -8, -75, 44, -5, -2, 9, -41}
, {-11, 2, 22, -3, -16, 28, 62, -48, -15, 5, -112, 53, 1, 1, -17, 159}
}
, {{-1, -14, 12, 12, -20, 16, -20, -52, -43, 18, 14, 87, 6, 9, 13, 18}
, {1, -70, -39, 36, -33, 54, -15, -208, 6, 1, 9, 71, -48, 43, 18, 22}
, {-1, -33, -41, -51, -29, 31, -19, 29, -49, 8, -11, -9, 7, -52, -1, -11}
, {4, 51, -18, 1, 4, 14, -48, 35, -3, 18, 3, 103, -82, 27, 22, 6}
}
, {{-11, -51, 16, -18, -54, 66, -7, 57, -23, 40, 4, 10, 0, -74, -25, -10}
, {5, -65, 7, -58, -74, 12, -15, -82, 18, 16, -13, -26, -23, -64, 7, 8}
, {-9, -111, 15, -66, -91, -35, -25, -40, 15, -4, -11, -41, -5, 59, -19, 14}
, {-30, -6, -33, -30, -63, 7, -26, -192, -5, 10, -54, -21, -59, -28, -3, 67}
}
, {{-29, -85, 38, -22, -67, -143, -10, -19, 12, -63, -70, -18, -13, -22, 5, -19}
, {-8, 53, -14, -20, -34, -64, 15, 106, -27, -53, -21, -57, -27, -9, -18, 30}
, {1, 71, -10, 29, 57, -85, 2, 6, -37, -73, -33, 90, -25, -62, 3, -84}
, {-9, -28, -51, -43, -120, 29, -20, -28, -5, 52, 4, 112, 27, 60, -2, 66}
}
, {{5, 55, -7, -15, 14, -100, -71, -66, 86, 24, -30, 32, -11, 11, -28, 19}
, {-9, 13, 24, 49, -68, -100, 85, 3, 88, 43, -32, -55, 65, -14, 23, 50}
, {-8, 117, 10, 14, -1, -45, 28, 89, 35, -16, -10, 35, 42, 69, 25, 92}
, {15, 70, 12, 36, 3, -91, -10, 34, 58, -11, -34, 22, -41, -53, 24, 97}
}
, {{-20, -69, -59, -96, 41, -39, -43, 12, 20, -37, 16, 44, 61, 3, 8, 60}
, {-38, -61, -67, -41, 68, -16, 11, 60, 23, -51, -9, -88, -11, -91, 8, -58}
, {63, -44, 4, -50, 34, 6, -6, 43, -51, -1, 18, -44, -5, -118, 18, 62}
, {-4, -59, -32, -13, 11, 31, 34, 0, -22, -8, 41, 33, 232, -44, -12, 78}
}
, {{-7, -8, -9, -26, -65, -36, 14, 43, -49, -48, -2, -29, -24, 15, -22, 27}
, {-10, 39, -79, 35, 35, -30, -14, 16, 32, -39, -5, 64, -27, -26, 10, 83}
, {-27, 41, 4, -21, -78, 19, -42, 23, -8, -8, -13, 48, -48, -14, -8, 8}
, {17, 25, 78, -77, -92, 34, -86, -65, 56, 61, 9, -57, 48, -34, -1, 3}
}
, {{-8, 1, 62, 21, -78, -38, 14, 20, -43, 17, -25, -129, 44, -40, -3, 28}
, {22, 28, -17, -29, 15, 70, 4, 51, -4, -90, -92, 6, -30, 12, -30, -57}
, {-8, 25, -144, -12, 48, 5, 40, 52, -4, -104, 10, 76, 35, -2, -21, -32}
, {-1, 15, -164, -2, -21, -102, -69, 6, -43, -71, 3, 58, -5, -27, -1, -86}
}
, {{-16, 6, -19, -3, -37, 5, -11, 73, -7, -18, -1, -6, 0, -25, 24, 121}
, {-12, 7, 8, -5, -70, -18, -83, -7, 73, 17, -44, -15, -37, -84, 15, 2}
, {-8, 63, -29, 92, 55, 14, 4, -43, 45, 6, 32, -22, -20, 33, -11, 85}
, {20, 73, -39, 73, -63, -19, 25, 49, -37, -4, -11, -49, -14, 36, -2, 95}
}
, {{-11, 33, 44, 24, 25, 31, 94, 112, -76, 15, 32, 58, -62, 17, 25, 40}
, {-5, -24, 22, -4, -7, 17, -20, 15, -20, 27, 43, 8, -59, -30, 1, -3}
, {-6, -50, -15, -11, 19, 1, -1, 87, -39, -12, 0, 53, 41, 50, -11, -39}
, {-5, -52, -33, 11, 22, -20, 93, -46, 2, -20, -36, -17, -24, 10, -37, -53}
}
, {{-20, 32, 45, -6, -39, 33, -3, -60, -20, 41, 1, 49, 5, 42, 18, 78}
, {9, 25, -3, 19, -19, -37, -46, 75, -19, -23, -61, -103, -42, -155, 6, 128}
, {-7, -31, -132, -94, -21, -54, 62, 34, -20, -74, 27, 45, 3, -93, -2, 67}
, {1, 5, -203, -49, -53, -26, -41, 10, 19, -32, -2, 15, -60, -121, -20, -76}
}
, {{-1, -31, -43, -60, 31, 26, 21, 60, -2, -27, -23, -147, -4, 24, 4, -43}
, {19, -118, -9, 80, -59, -7, -17, -21, 9, 26, 20, 2, 24, -44, 13, 79}
, {7, 128, -9, -95, -22, -32, 67, 70, -5, -32, -77, 4, 45, 30, 11, -56}
, {-13, -28, -25, 40, -57, 11, 28, 26, -28, -49, 20, 10, 6, -26, -15, 9}
}
, {{11, -2, -144, 32, 63, 37, -15, -26, 8, -57, 36, -40, 19, 10, -6, 0}
, {-7, -46, -29, -11, -50, 47, 6, -97, -19, 23, -93, -36, 18, 21, -22, -115}
, {-5, 87, -48, 6, -5, -22, 23, 23, -38, -23, -63, -10, 85, -25, -4, 53}
, {-22, -22, -64, -49, -1, -27, -28, 46, -51, -18, -89, 36, 15, -20, -3, -99}
}
, {{-18, 89, 28, 25, -5, -64, 66, -13, 66, 59, 18, -74, -40, -3, -2, 62}
, {-14, 33, 10, 71, -36, -17, -6, -5, 99, 26, -1, -15, -25, -17, 8, -7}
, {10, 48, -11, 20, -31, -26, 20, -30, 40, -10, 36, -26, -7, -21, 6, -1}
, {-7, 2, 8, 28, 25, -11, -50, 39, 29, 12, 34, 23, -4, -4, -8, -1}
}
, {{7, -42, -112, -15, -24, -96, 8, -23, 30, -151, -80, 80, -26, -95, 9, 17}
, {16, -44, -100, 103, 34, -216, 78, 162, -46, -164, -104, 73, 136, 3, -3, 82}
, {20, 78, 25, 10, 30, -13, -14, 6, 41, -2, 14, 8, 4, 43, -14, 19}
, {-2, 46, 12, 21, -36, 29, -65, 49, 51, 43, 9, -22, 21, -39, -1, 69}
}
, {{-19, -119, -2, 48, 15, 34, 34, 55, 36, -27, -4, 5, -29, -85, -18, 49}
, {3, -57, -80, 9, -51, 15, 17, -5, 15, -69, 39, 5, 1, -31, -12, -149}
, {8, -71, -114, -14, -24, -22, -53, -38, -22, -104, 37, -89, -25, -147, 3, -87}
, {-21, -122, -64, 24, -58, -36, -9, 7, -19, -27, 3, 40, 6, -71, -1, -83}
}
, {{-21, -20, 0, 55, 11, 16, 2, -111, 24, 30, 59, -116, 45, 38, -5, 3}
, {-13, -40, -14, 43, -55, 0, -16, -86, 4, 36, 51, -25, -31, 65, -1, -49}
, {-7, -55, -42, 5, -71, -18, 8, 143, 8, -9, -24, -32, -6, -129, -4, 26}
, {-1, -111, 19, -34, 7, -112, 29, -13, 0, -84, -99, -41, -30, 14, 13, -50}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_GROUPS
/**
  ******************************************************************************
  * @file    maxpool1d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 january 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _MAX_POOLING1D_29_H_
#define _MAX_POOLING1D_29_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS  32
#define INPUT_SAMPLES   48
#define POOL_SIZE       2
#define POOL_STRIDE     2
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

typedef int16_t max_pooling1d_29_output_type[POOL_LENGTH][INPUT_CHANNELS];

#if 0
void max_pooling1d_29(
  const number_t input[INPUT_SAMPLES][INPUT_CHANNELS], 	    // IN
  number_t output[POOL_LENGTH][INPUT_CHANNELS]); 	// OUT
#endif

#undef INPUT_CHANNELS  
#undef INPUT_SAMPLES
#undef POOL_SIZE
#undef POOL_STRIDE
#undef POOL_PAD
#undef POOL_LENGTH

#endif//_MAX_POOLING1D_29_H_
/**
  ******************************************************************************
  * @file    maxpool.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "max_pooling1d_29.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#endif

#define INPUT_CHANNELS  32
#define INPUT_SAMPLES   48
#define POOL_SIZE       2
#define POOL_STRIDE     2
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

#define ACTIVATION_LINEAR

// For fixed point quantization
#define INPUT_SCALE_FACTOR 7
#define OUTPUT_SCALE_FACTOR 7
#define OUTPUT_ROUND_MODE ROUND_MODE_FLOOR
#define NUMBER_T int16_t
#define LONG_NUMBER_T int32_t


static inline void max_pooling1d_29(
  const NUMBER_T input[INPUT_SAMPLES][INPUT_CHANNELS], 	    // IN
  NUMBER_T output[POOL_LENGTH][INPUT_CHANNELS]) {	// OUT

  unsigned short pos_x, k; 	// loop indexes for output volume
  unsigned int x;
  static LONG_NUMBER_T max[INPUT_CHANNELS];

  for (pos_x = 0; pos_x < POOL_LENGTH; pos_x++) {
    for (k = 0; k < INPUT_CHANNELS; k++) {
#ifdef ACTIVATION_LINEAR
      max[k] = input[pos_x*POOL_STRIDE][k];
      x = 1;
#elif defined(ACTIVATION_RELU)
      max[k] = 0;
      x = 0;
#else
#error "Unsupported activation function"
#endif
    }

    for (; x < POOL_SIZE; x++) {
      for (k = 0; k < INPUT_CHANNELS; k++) {
        if (max[k] < input[(pos_x * POOL_STRIDE) + x][k])
          max[k] = input[(pos_x * POOL_STRIDE) + x][k];
      }
    }

    for (k = 0; k < INPUT_CHANNELS; k++) {
      output[pos_x][k] = scale_and_clamp_to(NUMBER_T, max[k], INPUT_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
    }
  }
}

#undef INPUT_CHANNELS  
#undef INPUT_SAMPLES
#undef POOL_SIZE
#undef POOL_STRIDE
#undef POOL_PAD
#undef POOL_LENGTH
#undef ACTIVATION_LINEAR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
#undef NUMBER_T
#undef LONG_NUMBER_T
/**
  ******************************************************************************
  * @file    conv1d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 january 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _CONV1D_39_H_
#define _CONV1D_39_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS      32
#define INPUT_SAMPLES       24
#define CONV_FILTERS        64
#define CONV_KERNEL_SIZE    2
#define CONV_STRIDE         1

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

typedef int16_t conv1d_39_output_type[CONV_OUTSAMPLES][CONV_FILTERS];

#if 0
void conv1d_39(
  const number_t input[INPUT_SAMPLES][INPUT_CHANNELS],                    // IN
  const number_t kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS],  // IN

  const number_t bias[CONV_FILTERS],						                          // IN

  number_t output[CONV_OUTSAMPLES][CONV_FILTERS]);                       // OUT
#endif

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES

#endif//_CONV1D_39_H_
/**
  ******************************************************************************
  * @file    conv.cc
  * @author  Sébastien Bilavarn, LEAT, CNRS, Université Côte d'Azur, France
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "conv1d_39.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#elif defined(WITH_NMSIS_NN)
#include "riscv_nnfunctions.h"
#endif

#define INPUT_CHANNELS      32
#define INPUT_SAMPLES       24
#define CONV_FILTERS        64
#define CONV_KERNEL_SIZE    2
#define CONV_STRIDE         1
#define CONV_GROUPS         1
#define CHANNELS_PER_GROUP  (INPUT_CHANNELS / CONV_GROUPS)
#define FILTERS_PER_GROUP   (CONV_FILTERS / CONV_GROUPS)

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

#define ACTIVATION_RELU

// For fixed point quantization
#define WEIGHTS_SCALE_FACTOR 7
#define BIASES_SCALE_FACTOR 7
#define TMP_SCALE_FACTOR 7
#define INPUT_SCALE_FACTOR 7
#define OUTPUT_SCALE_FACTOR 7
#define OUTPUT_ROUND_MODE ROUND_MODE_FLOOR
#define NUMBER_T int16_t
#define LONG_NUMBER_T int32_t


static inline void conv1d_39(
  const NUMBER_T input[INPUT_SAMPLES][INPUT_CHANNELS],                    // IN
  const NUMBER_T kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS / CONV_GROUPS],  // IN

  const NUMBER_T bias[CONV_FILTERS],						                          // IN

  NUMBER_T output[CONV_OUTSAMPLES][CONV_FILTERS]) {                       // OUT

#if !defined(WITH_CMSIS_NN) && !defined(WITH_NMSIS_NN)
  unsigned short pos_x, z, k; 	// loop indexes for output volume
  unsigned short x;
  int input_x;
  LONG_NUMBER_T output_acc;

  for (pos_x = 0; pos_x < CONV_OUTSAMPLES; pos_x++) { 
    for (k = 0; k < CONV_FILTERS; k++) { 
      output_acc = 0;

      for (x = 0; x < CONV_KERNEL_SIZE; x++) {
        input_x = pos_x * CONV_STRIDE - ZEROPADDING_LEFT + x;

        if (input_x >= 0 && input_x < INPUT_SAMPLES) { // ZeroPadding1D
          for (z = 0; z < INPUT_CHANNELS / CONV_GROUPS; z++) {
            output_acc += (LONG_NUMBER_T)input[input_x][z + (k / FILTERS_PER_GROUP) * CHANNELS_PER_GROUP] * (LONG_NUMBER_T)kernel[k][x][z];
          }
        }
      }

    // Scale for possible additional precision of bias
    output_acc = scale(NUMBER_T, output_acc, WEIGHTS_SCALE_FACTOR - TMP_SCALE_FACTOR, OUTPUT_ROUND_MODE);

    // Scale bias to match accumulator
    output_acc += scale(NUMBER_T, (LONG_NUMBER_T)bias[k], BIASES_SCALE_FACTOR - TMP_SCALE_FACTOR - INPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);

      
#ifdef ACTIVATION_LINEAR
      output[pos_x][k] = scale_and_clamp_to(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
#elif defined(ACTIVATION_RELU) || defined(ACTIVATION_RELU6)
      // Activation function: ReLU
      if (output_acc < 0) {
        output[pos_x][k] = 0;
      } else {
#if defined(ACTIVATION_RELU6)
        if (output_acc > scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE)) {
          output_acc = scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE);
        }
#endif
        output[pos_x][k] = scale_and_clamp_to(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
      }
#else
#error "Unsupported activation function"
#endif
    }
  }

#else

#if BIASES_SCALE_FACTOR > WEIGHTS_SCALE_FACTOR
#error "CMSIS-NN does not support BIASES_SCALE_FACTOR larger than WEIGHTS_SCALE_FACTOR"
#endif

  static q15_t bufferA[2*INPUT_CHANNELS*CONV_KERNEL_SIZE];
#if INPUT_CHANNELS % 2 == 0 && CONV_FILTERS % 2 == 0
#ifdef WITH_CMSIS_NN
  arm_convolve_HWC_q15_fast_nonsquare(
#elif defined(WITH_NMSIS_NN)
  riscv_convolve_HWC_q15_fast_nonsquare(
#endif
#else
#ifdef WITH_CMSIS_NN
  arm_convolve_HWC_q15_basic_nonsquare(
#elif defined(WITH_NMSIS_NN)
  riscv_convolve_HWC_q15_basic_nonsquare(
#endif
#endif
                                      (q15_t*)input, //Im_in
                                      INPUT_SAMPLES, //dim_im_in_x
                                      1, //dim_im_in_y
                                      INPUT_CHANNELS, //ch_im_in
                                      (q15_t*)kernel, //wt
                                      CONV_FILTERS, //ch_im_out
                                      CONV_KERNEL_SIZE, //dim_kernel_x
                                      1, //dim_kernel_y
                                      ZEROPADDING_LEFT, //padding_x, left and right must be equal
                                      0, //padding_y
                                      CONV_STRIDE, //stride_x
                                      1, //stride_y
                                      (q15_t*)bias, //bias
                                      INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - BIASES_SCALE_FACTOR, //bias_shift
                                      INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, //out_shift
                                      (q15_t*)output, //Im_out
                                      CONV_OUTSAMPLES, //dim_im_out_x
                                      1, //dim_im_out_y
                                      bufferA, //bufferA
                                      NULL //bufferB, unused
                                      );
#ifdef ACTIVATION_RELU
#ifdef WITH_CMSIS_NN
  arm_relu_q15((q15_t*)output, CONV_FILTERS * CONV_OUTSAMPLES);
#elif defined(WITH_NMSIS_NN)
  riscv_relu_q15((q15_t*)output, CONV_FILTERS * CONV_OUTSAMPLES);
#endif
#elif !defined(ACTIVATION_LINEAR)
#error "Unsupported activation with CMSIS-NN"
#endif


#endif
}

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef CONV_GROUPS
#undef CHANNELS_PER_GROUP
#undef FILTERS_PER_GROUP
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES
#undef ACTIVATION_RELU
#undef WEIGHTS_SCALE_FACTOR
#undef BIASES_SCALE_FACTOR
#undef TMP_SCALE_FACTOR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
#undef NUMBER_T
#undef LONG_NUMBER_T
/**
  ******************************************************************************
  * @file    weights/conv1d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

#define INPUT_CHANNELS    32
#define CONV_FILTERS      64
#define CONV_KERNEL_SIZE  2
#define CONV_GROUPS       1


const int16_t  conv1d_39_bias[CONV_FILTERS] = {-12, 6, 24, 18, 40, -10, -20, 53, 39, -9, -6, 145, -30, 23, -4, 124, -17, 37, -10, -7, 26, -41, -4, -14, -9, 37, -9, 4, -15, -13, 12, 0, 26, 69, 26, -37, 12, 51, -11, -102, 28, -9, 67, 39, -49, -3, 85, 78, 8, -6, 13, 24, -4, 33, 16, 44, 53, -10, -11, 93, 91, 4, -26, -4}
;

const int16_t  conv1d_39_kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS / CONV_GROUPS] = {{{2, -30, -36, 7, -21, 14, -17, -11, -18, -6, -27, -6, -25, -14, -18, -19, -18, -11, -1, -1, 0, -10, -9, -15, -33, -24, 11, -4, -25, -9, -28, -6}
, {16, -27, -39, -8, -27, 14, -5, 9, -33, -3, -4, -8, 9, 7, 9, -80, -20, 8, -21, 1, -11, -28, -21, -23, -6, -28, -11, -3, -18, 6, -11, -25}
}
, {{-3, 40, -150, -3, -11, -2, -23, -12, -14, -26, -20, 36, -8, 42, 61, -63, -54, -17, -41, -47, -45, 34, 12, -26, -69, 21, -50, 25, -47, -37, -61, 42}
, {45, -19, 14, -129, -12, 63, 11, 34, -53, -22, 7, -33, 30, -43, -94, -27, 42, 16, -63, 3, -51, 11, 15, 43, 21, 38, -25, -15, 30, 3, -14, 27}
}
, {{-77, 71, -97, 21, -53, 28, -175, 13, 18, 10, -36, 15, 16, 34, -97, 72, -21, 26, 5, -21, -56, -9, -71, 40, -41, 59, 0, -114, 3, 30, -56, -166}
, {-37, 5, -113, 39, -59, -6, -56, 41, 46, -6, -5, -1, -4, -27, 21, 48, -26, 30, -75, 11, -36, -1, -24, 20, -31, -82, 47, -147, 22, -46, -79, -161}
}
, {{-52, 8, -114, 7, 57, 31, 15, 19, 15, -24, 6, 20, 5, -91, 7, -53, 42, -71, 0, -55, -56, -21, 54, -5, 2, 47, 2, 40, -47, -173, 38, -128}
, {-44, -49, -47, -7, 44, 16, -99, -167, -66, -14, -85, -99, -36, 16, -28, -53, 10, -24, 18, -73, 29, -76, -106, -19, -10, -80, -15, 97, 17, -115, 40, -51}
}
, {{-20, -108, 39, -64, -58, -80, -27, 0, -26, 27, -21, -21, -11, 3, -1, -47, -67, -41, -79, 5, -60, -183, 15, -106, -88, 24, -112, 26, 28, 32, 43, -4}
, {-48, -26, 48, -65, -34, -77, -16, -212, 11, 24, -54, -59, -104, 75, -35, -109, -77, -35, 11, -1, -27, 6, 171, -93, -38, 168, -108, 63, -87, 117, 127, 95}
}
, {{-19, -15, -29, 3, -6, -1, -40, 2, -30, 18, 1, -24, 13, -6, -17, -10, -15, -2, -1, -9, -18, -33, 19, -6, -8, -25, 1, -3, -35, -17, -5, -24}
, {-32, -13, -8, 4, -9, -23, -10, -11, 8, -17, 5, -3, -9, -28, -7, -31, -14, 9, -2, -4, -48, -23, 15, -8, -31, 1, 1, -18, -28, -4, -24, -9}
}
, {{29, -183, -18, -2, -69, -24, 59, -158, -20, -29, -91, -34, -53, -113, 107, -113, -61, -160, -22, -76, -26, -152, 52, -151, -155, -60, -30, -14, 0, 101, -16, -74}
, {-83, -132, -17, -156, 10, -21, -81, -153, -47, 14, -100, -1, -140, -30, -21, -70, -58, 21, 12, 29, 15, -33, 30, -106, -87, -15, -70, -138, -69, 112, -74, -58}
}
, {{61, -66, -58, -43, -96, 12, -81, 36, -31, 19, -4, -134, -7, -24, -57, -26, -1, 68, -36, -25, -111, -18, -24, -21, 4, -21, -33, -71, 48, 7, -66, -61}
, {-10, -111, 41, -48, -274, -121, -49, -25, -4, 22, -13, 45, -7, 73, 19, -18, -40, -14, -155, -40, -161, 37, -164, -121, -76, -78, -44, -94, -57, -99, -120, 0}
}
, {{52, -20, 22, -58, -1, -19, -30, -70, -64, 10, -34, -19, -3, -17, -45, 31, -76, -72, 25, -26, -15, 15, -24, -26, -180, -111, -12, 20, -60, 31, -124, 23}
, {-15, -15, 8, 1, 36, -20, -98, 60, 31, -38, 64, -37, 66, 133, 100, 30, 41, -17, -46, 2, -47, 27, -49, -3, 6, -111, -38, 9, -15, -135, 39, -23}
}
, {{4, -21, -22, 10, -1, 5, -3, -30, -3, 10, -12, 14, -24, -27, 12, -23, -18, -1, 2, -38, -24, 9, -5, -15, -20, -35, -29, 8, -4, -8, 4, -25}
, {3, 8, -32, -11, -16, -7, -23, -31, -16, -1, -1, -24, -20, -28, -29, -22, -11, -12, -7, -26, -55, -27, -30, -31, -19, -6, -1, 2, -29, 1, -19, -12}
}
, {{23, 1, -19, -26, -23, -19, -20, -15, -4, -10, -2, -13, -20, -16, -18, 1, -14, -11, -29, -4, -24, -21, -2, -26, -19, -24, 11, -20, -6, -13, -30, 7}
, {8, -30, -9, -21, -20, 8, -9, -11, -20, -20, -29, -30, 6, 6, 13, -22, -7, -38, 11, -14, -20, -25, -29, -32, -15, -5, -8, -33, -13, 6, 3, -18}
}
, {{60, -127, -48, 10, 51, -7, -96, -49, -66, -8, -25, -23, 69, -33, -32, -235, 26, -45, -1, -86, 8, -122, -92, -44, -35, -154, -83, 2, -80, -118, 55, -45}
, {67, -10, -107, -11, -32, -13, 43, 4, -57, 5, -52, -35, -47, 12, 20, -192, 19, -1, -30, -20, -14, -67, 0, -59, -52, -18, -66, -53, 59, -96, -21, 31}
}
, {{-71, -38, -25, 55, 19, 21, 15, -20, 19, 14, 37, 25, -84, -105, -5, 18, 11, 8, 60, 27, -60, -26, 114, 5, 7, 83, 57, 4, 12, -56, 23, -13}
, {-49, -2, -60, 41, 8, -2, 6, -187, 32, 15, -43, 78, 81, -53, -51, -1, -26, -131, -80, 5, 31, -11, 14, 42, -60, -46, 3, 21, 3, 23, 1, -61}
}
, {{10, -65, 47, 102, -99, -23, -153, 16, -17, 5, -28, -13, 30, -66, 1, 90, -59, 52, -35, -128, -36, 27, -31, -172, -105, -103, -168, -43, -30, -72, -135, 30}
, {53, 107, 47, -47, 86, -8, -50, -136, -89, -13, -4, -21, -52, 33, -51, -158, -15, -42, -89, 25, -92, -86, 61, 77, -36, 51, -22, 8, -15, 14, -193, -195}
}
, {{12, -92, 88, 76, -39, -13, -139, -104, 82, 4, -68, -164, -3, 16, -51, -4, 26, -54, -42, 40, -156, 7, 68, -31, -62, -10, -73, -14, -92, 104, 61, -59}
, {-95, -49, 0, -10, -95, -20, -213, -11, -12, -59, -54, 25, -103, 71, 81, 81, -4, -26, -103, -15, -47, -24, 30, 8, -61, -11, 49, -31, -131, 92, -85, 2}
}
, {{-2, -102, -99, 19, -65, 6, -196, 23, -35, 10, 1, 7, -11, -88, -54, -7, 1, -4, -60, -64, -33, -37, 93, -18, -24, -169, -98, -116, 53, -72, -283, -45}
, {8, 5, -128, 21, -78, -66, -91, -10, -70, 21, 34, -18, 38, 22, -29, 19, 44, 29, 112, -72, -82, -77, 17, -30, -28, -63, -21, -202, -9, -80, 135, -87}
}
, {{-24, -8, -28, -29, -40, 0, -51, -20, -32, -3, 10, -25, -46, -28, -12, -6, -58, -58, -18, -12, -95, -21, -30, -13, -106, 13, -1, -21, -22, -79, -19, -47}
, {-48, 7, -6, -48, -12, -13, 21, -40, 9, -21, 10, -1, -63, -33, 10, -21, -34, -30, 0, -32, 10, -8, -13, -35, -60, -54, -37, -24, -63, 22, 11, -32}
}
, {{110, 45, -44, -61, -17, 18, 9, -115, -26, 4, -28, 22, 68, -84, 22, 23, -42, -28, 49, -163, -80, 22, 7, -31, -96, -87, 7, 37, -85, 38, 78, 51}
, {-30, 71, 11, -1, 12, 2, 6, -131, 31, -4, 31, -33, -66, 82, 204, 36, 17, -18, -29, -129, 3, -17, 110, 18, -68, -31, 45, 67, -65, -44, 2, -46}
}
, {{3, 8, -2, 9, -23, -15, 13, 10, -8, -30, 11, 1, 6, -7, 1, -29, -26, -23, 13, -29, -5, -6, -4, -18, -16, -9, -16, -2, -17, -18, -3, 7}
, {-36, -7, -12, 3, 8, 2, -4, 12, -20, 14, -31, 6, -7, -5, -24, 6, -1, -18, -5, -8, -21, -18, -30, -6, -11, 6, -24, 2, -3, -4, 4, -2}
}
, {{-125, -108, -10, -14, 4, -50, 63, -48, -32, 0, 18, -14, -22, 6, -17, -32, 29, 6, 76, -1, -77, 33, -79, 3, -18, -22, -19, 27, -58, -4, -10, -11}
, {-14, -112, 40, -116, -38, -53, -31, -24, -65, 7, 10, -102, -83, -66, -86, 24, 27, 34, -7, -78, -61, 41, -4, -60, -8, 80, 31, 20, -35, -83, -35, 10}
}
, {{-37, 32, -3, -29, -11, 6, -98, -36, 50, -8, -34, 78, 44, 50, -119, -32, -4, -85, -41, 36, -60, 36, -72, 16, -44, -67, 20, -136, 36, 15, -58, -169}
, {-22, -20, -45, -68, -27, -148, -97, 13, -21, -2, 5, -1, 32, -70, -110, 25, -15, -19, -18, 40, -108, 27, 44, 44, -75, 20, -8, -181, 34, 45, 30, -129}
}
, {{-145, -125, -35, 22, -78, 24, -1, -51, 4, -4, -29, -73, -74, -5, -3, -52, -131, -68, 74, 19, -15, -27, -67, -51, -86, -78, -61, -69, -65, 20, -82, -35}
, {-6, 48, -58, 90, 29, -2, -21, -27, -4, 2, 50, 114, -58, 120, 176, -51, -20, -53, 24, -44, 49, -5, 12, 43, 13, -41, -2, 13, 33, -18, 37, 105}
}
, {{32, -12, -96, -61, 48, -37, -8, -9, 32, 19, -7, -25, -48, -56, 42, -50, -9, -15, -86, -65, 20, 32, -98, 24, -16, -175, 45, -23, -43, -150, 2, 14}
, {21, -111, -54, -3, -43, -20, -144, 2, -38, -11, 58, -15, 54, 15, -20, 0, 8, 7, -47, 29, -49, -1, 26, -44, 9, -147, -31, 1, 42, -138, -31, 29}
}
, {{-39, 0, -34, -2, -17, 6, -11, -26, -4, 13, -2, -21, -3, 11, 6, -18, 2, -24, -3, -34, 0, -15, -14, 3, -29, -21, -5, 3, -6, -16, -23, 5}
, {-75, -12, -25, 3, -13, 6, -8, 1, -29, -16, -15, -3, 17, -18, 10, -2, -21, -4, -13, -32, -5, 2, -30, -17, -6, -6, -14, 6, -30, -35, 1, -32}
}
, {{-8, 5, 4, 11, -17, -8, 1, -25, -31, 5, 5, -26, -20, -11, -6, -24, -12, 0, -21, -9, -21, -23, -10, -32, -35, 8, 4, -17, -10, -3, -28, -9}
, {-3, 4, -7, -1, -27, -1, -13, -6, -17, -2, -24, -26, 2, -3, -26, -20, -20, -2, 4, -15, -9, 3, -8, -6, -16, 14, -14, 15, -11, -17, 4, -10}
}
, {{25, 74, 3, 5, 26, -8, -41, -75, -22, 10, -5, -17, 7, 0, 0, -68, -35, -103, -54, 33, -44, -49, 57, -56, -151, -111, 48, -1, -31, -22, -128, -126}
, {51, 29, -82, 40, 21, -5, -46, 44, -88, -2, 25, 3, -10, -54, 16, 39, 32, 61, 30, -15, 13, 60, -14, -50, 13, -60, 12, 35, -6, -11, -15, -64}
}
, {{-3, 8, -23, 4, -29, -26, -6, -2, -11, -18, 11, 13, 6, -3, -10, -18, -7, -1, 2, -20, -3, 5, -25, -17, -26, -2, -21, -6, -15, -19, -27, -6}
, {-6, 8, -7, -15, -11, -11, -14, 6, -23, -6, -31, 5, -5, -31, 7, -20, -19, 4, -23, 0, -21, -12, 3, -20, -26, 13, -17, -30, -11, -26, 2, 11}
}
, {{-92, -57, -99, -10, -30, -15, 4, 9, -38, -6, -10, 2, -71, -15, 10, -12, -23, -48, -15, -81, -37, 3, -57, 5, 84, -100, 18, -42, -36, 81, -37, -29}
, {-19, 9, -14, -8, -37, 6, 20, -81, 6, 7, -18, 9, -3, -5, -27, -111, -88, -38, -24, -58, -52, -27, 13, -12, -129, -16, -27, 27, 6, -28, -4, -68}
}
, {{-12, -44, -3, 8, -33, -4, -4, -15, 5, -10, -31, 13, -3, -26, -1, -39, -57, -16, -25, -21, 1, -6, -23, -23, -61, -9, -35, -11, 1, -38, -7, -19}
, {31, 9, -88, 6, -59, -17, -11, -25, -21, -18, 14, -6, -21, 8, 6, -80, 6, -51, 5, -53, -87, -38, -12, -23, -38, -40, -11, -7, -35, -64, -14, -24}
}
, {{-126, 56, -19, -71, -122, 8, 28, -19, -13, 25, 33, -42, -45, -100, 11, -11, 12, 34, 22, -12, -69, -29, 10, -9, -22, 46, 16, -5, -22, 37, -66, 6}
, {-21, -50, 15, -141, -18, -23, -79, -25, -73, 1, 27, -14, 22, 36, 3, 69, -27, -20, -3, -64, -71, 98, 53, -43, -31, -163, 25, -60, -43, -30, -120, -164}
}
, {{-70, -52, -120, 5, -48, -7, -109, 7, -56, -20, 20, 2, -16, -8, -26, -43, -70, -74, 6, -42, -98, -41, -62, -100, -82, 72, -13, -110, -34, -126, 14, 4}
, {-60, 1, 86, -64, -30, -29, -98, -96, 63, 7, -16, 13, -40, -67, -16, -35, -26, -12, 6, 54, 36, -76, 105, 2, -130, -22, 80, 9, -18, 31, 71, 10}
}
, {{23, 81, 10, 13, -88, -16, 49, -10, -57, -13, -61, -71, -10, -20, -5, -73, -58, -50, -26, 13, -70, -58, -57, 27, -153, 33, -131, 35, -7, -96, 11, 58}
, {-13, 134, 24, 20, 44, -6, 52, -34, 43, -5, -48, -35, -26, -25, 1, -35, -37, -68, -50, -33, -42, 15, 19, 10, -44, 22, -18, -6, -25, 88, 41, -101}
}
, {{55, 28, 12, -96, 36, -22, -26, -8, -7, 22, 24, 95, -12, -65, 5, -48, -45, -19, -37, -49, -32, -22, -171, 83, -38, -62, -19, -20, 22, -49, -3, -8}
, {105, -42, -111, -120, -11, 15, -2, 5, -36, 17, 26, -26, -80, -39, -57, -13, -42, 5, 85, -44, 33, -7, 75, -15, -96, -9, -47, -66, -21, 83, 5, 20}
}
, {{-49, -6, -17, -30, -37, 9, -147, 47, -27, -1, -141, 10, -21, 1, 65, -4, -102, -120, -56, -102, 13, -48, -26, -138, -54, -46, -69, -22, -116, 47, 3, -85}
, {-45, 5, 74, -41, 44, 91, 38, -12, -46, 15, 98, -87, -16, 90, 110, -12, 12, 8, 172, -141, -1, 39, 27, -46, -20, 50, 18, -107, -97, 78, 27, 42}
}
, {{-92, 34, 106, -96, -122, -8, -83, -86, -65, 10, -94, 10, -31, -9, 41, 11, 4, -25, -54, -35, -57, -88, -40, -79, -80, -1, -138, 55, -107, -99, 44, -52}
, {-21, -79, -123, 16, 31, 21, -156, 75, -6, 1, -39, 8, 18, -22, 78, 26, -200, -9, -3, -5, -7, 51, -71, -65, -85, -86, -73, 20, -74, 53, 59, 62}
}
, {{13, -40, -29, -102, -40, -33, -57, 24, -59, -32, -7, -75, 8, -83, -24, -30, -7, 36, 13, 17, -151, 59, -87, 19, 41, -50, -56, -52, 11, 32, -59, 3}
, {23, 0, -47, -55, -36, 14, -37, -3, -19, -28, 42, -76, -27, -3, -5, -6, 22, 33, 31, -104, -51, -18, 59, 0, 11, -3, -13, -58, -30, 49, -43, -27}
}
, {{-248, -4, -116, -31, -20, -40, 11, 21, -9, -10, -127, -80, 4, 25, 11, -4, -108, -98, -2, 13, -47, -45, 6, -20, -114, -59, 55, -39, -156, 43, -57, -7}
, {24, 76, -67, 6, -17, 15, -66, 17, 11, 34, 51, 21, -8, -45, -7, 3, -31, 20, 3, -6, -19, 3, 23, 21, -31, -2, 55, -46, 79, 71, 21, 53}
}
, {{-35, -8, -25, -31, -4, 5, 88, -13, 53, 12, -21, 23, -19, -14, 10, 59, -3, -85, 12, -57, -12, -8, 79, -31, -161, 67, -62, -85, -49, -13, 62, -29}
, {-48, -78, -17, 32, -13, -7, -27, -30, 5, 10, 46, 26, -98, 34, -43, -7, -22, -37, -16, -16, 49, -109, 0, -98, -79, 23, 2, -169, -22, 43, -23, -135}
}
, {{-34, -19, -13, -15, -50, 8, 11, -22, 6, -9, -9, -2, -19, -23, 27, -17, -9, 5, -14, -27, -18, -17, -5, -17, -37, -23, -22, -11, -12, -16, -14, 4}
, {-13, 7, -35, -16, -41, -6, -3, -23, -18, -6, -16, 3, -1, -21, 26, -6, -16, -20, -4, -18, -15, -12, -17, -18, -40, 6, -21, -9, -30, -14, -29, -38}
}
, {{-27, -1, -39, -49, 19, -82, 16, -14, -36, 34, 12, -21, -72, -101, 7, 6, -2, 8, -40, 28, 23, -43, 32, -3, 23, -37, -94, -19, 37, -26, 64, -62}
, {-15, -57, 49, 0, -16, 4, 38, 8, -11, 18, 5, -34, -58, 11, -33, 10, 8, -22, -2, 51, -38, -26, 2, 33, 23, 63, 33, 0, 5, -14, -13, 18}
}
, {{36, -8, -125, -52, 38, 7, -72, -16, 43, -9, -17, -13, -11, 27, 33, -60, 42, 23, -9, -88, -109, -57, 19, 68, -43, -33, -20, -11, -1, 12, 49, 40}
, {-29, -33, -94, -13, -57, 6, -64, 35, -56, -2, 18, -71, -45, -72, -105, -21, -22, 18, 41, -16, -65, 43, 115, 9, -58, 19, -54, -11, -17, -77, 25, -24}
}
, {{-52, -31, -58, -1, -26, 7, -3, -33, -19, -12, -8, -9, -13, 1, -2, -11, -16, 14, 5, 13, -2, -31, -12, -8, -19, -47, -26, 9, -23, 12, -12, -20}
, {-23, 1, -21, 7, 12, -18, -12, 6, -5, 13, -7, -6, -22, -22, -1, -12, -39, -19, -10, 15, -73, 0, -8, -20, -34, -20, 6, 3, -26, 18, 8, -1}
}
, {{24, -67, 75, -36, 19, -16, -149, -101, 39, 28, 60, 2, -70, -52, -89, -70, -30, -62, 56, 20, 67, 31, -103, -25, 12, -81, -27, -94, 33, 102, -31, 15}
, {-22, 68, 19, 5, 4, -20, 8, -11, -34, 7, -14, -86, -135, 33, -93, -38, -72, -58, 27, -79, -40, -138, 27, -66, -144, -13, -91, -34, -44, -93, 36, -27}
}
, {{137, -23, -9, -12, 10, -2, -49, -46, 21, -7, 17, 20, -10, 25, 0, -53, -96, -206, 30, -36, 25, -47, 50, -87, -9, -1, -69, 18, -10, 46, 31, -8}
, {-2, -20, -40, 3, 1, -10, -11, -21, 25, 2, -58, -60, 11, -43, -12, -116, -174, -134, -25, 36, 19, -102, 27, -157, -59, -48, -9, -50, -40, 36, 64, 45}
}
, {{3, -70, -49, -49, -26, -26, 30, -5, -40, 37, -17, 12, 18, 12, 71, -45, 15, 18, -80, -78, -143, -65, -60, 9, -11, 12, 37, -62, 0, 15, -17, 38}
, {-3, 49, -57, -48, 65, 25, 1, 10, 40, 40, 13, 7, -36, -64, 45, -28, 1, -30, 45, -21, -1, 10, 17, 9, 3, -13, 80, 47, -7, -52, -27, 70}
}
, {{-23, -15, 2, -24, -23, -17, -35, -8, -20, -13, 4, 2, 7, -25, -15, 6, -25, -20, 10, -5, -11, -26, 8, -7, -10, -4, -21, -26, 1, -13, 2, -9}
, {-23, -7, -11, -25, -29, -20, 1, -18, -8, 8, -23, 6, 7, 12, -25, -22, 3, -14, 4, 3, -11, -31, -29, -19, -26, 6, -14, -11, -1, 9, -27, -7}
}
, {{-87, -7, 16, -119, -64, -35, 38, 34, 41, 17, 29, -110, 28, -10, -31, 3, -23, -18, -136, 30, -63, -8, 0, 20, -127, 23, -51, 10, 29, -113, -18, -6}
, {7, 75, 36, -77, 24, -46, -13, -61, -19, 14, -22, -7, -49, -136, -112, -36, 9, -109, -111, 34, 18, -26, 51, 26, -121, 95, -65, -92, 1, -106, -24, -129}
}
, {{-57, -125, -57, -28, -36, -56, -29, -3, -7, -13, -35, -10, -92, 36, 42, -36, -2, 22, 30, 40, -128, -36, 9, 9, -96, 20, -18, 24, 34, 32, -13, 52}
, {19, -122, 26, 41, 23, -32, 35, -44, -26, 10, -40, 4, -53, -59, 64, -25, -22, 12, 28, 29, -59, -14, -45, 43, -107, 35, -34, -60, 33, 54, 28, -22}
}
, {{15, -55, 20, -32, -26, -11, -7, -20, -11, -7, -18, -9, 36, 6, -45, -67, -92, -59, -33, -86, -36, -7, 4, 72, 56, -111, -117, 99, -53, -50, -51, 2}
, {-64, -25, -78, -37, -69, -1, -23, -46, -28, -28, -7, -16, 55, 18, 11, 46, -61, -86, -1, -5, -12, -56, -65, 34, -70, 23, -22, -37, -101, -40, -88, -32}
}
, {{-15, -13, -38, -10, 2, 12, -11, -16, -16, -25, -5, -28, -5, -8, 8, -14, -8, -13, 7, -6, -1, -30, -4, -26, -26, -28, 8, -26, 3, 2, -14, -1}
, {7, -6, 3, 13, 0, 6, 3, -24, -5, -3, -14, -28, -10, -9, -6, -22, -2, -24, -15, -13, -18, -17, -23, -2, -8, -27, 7, -14, -27, 4, -9, 6}
}
, {{44, 5, 7, 21, -57, 7, -3, -61, 14, -16, -53, 17, 7, -79, 53, -86, -105, -166, 14, 4, 48, -64, 79, -38, -105, 59, 37, -47, -91, 90, -80, -59}
, {85, -27, -35, -90, -27, 26, -154, -25, -82, -9, -39, -9, 36, -37, -121, 32, -63, -88, -33, -42, -12, 72, -218, -24, -171, -77, -57, 5, -52, -5, -93, 35}
}
, {{74, -117, -54, 17, -4, 10, -10, -81, 52, 12, 15, 8, -101, -23, -63, -81, -9, -21, -44, -25, 22, -28, 79, -82, -79, -47, -39, -4, -58, -13, 116, 73}
, {-163, -111, -44, 9, 25, 2, 8, 119, 41, 2, -58, 0, 50, -11, 16, -38, -107, 20, 22, -181, 20, -167, 135, -127, -163, -4, 36, 10, -14, -75, -73, -42}
}
, {{-65, -41, -14, -23, -27, -83, 7, 33, -82, 14, 40, 31, -1, 4, -29, 59, 37, 21, -44, -15, 11, -27, -41, -31, 61, -38, -19, 0, -6, -20, -96, -64}
, {-4, -44, 32, -52, -82, -31, 22, -52, -28, 20, -17, 11, -145, 8, 1, 6, -26, -39, -14, -40, -117, -162, 26, -122, -36, -14, -21, -37, -60, -9, -35, 2}
}
, {{25, 86, -17, 16, 25, -33, -32, -233, 8, 21, -47, -14, -187, -50, -7, -156, -130, -96, -26, -16, 44, -6, 67, -39, 15, -49, -42, 98, -4, -1, 6, -12}
, {-62, -5, -43, -15, 0, -2, 0, 69, 5, -14, -21, -18, -64, -4, -7, -29, -5, 4, -41, -70, -27, 76, -28, -85, -136, -39, -98, -23, -81, 18, 3, -44}
}
, {{-68, -36, 63, -11, 33, 10, -15, -36, -11, 6, -17, 80, 22, -26, -61, -63, 60, 15, -8, 39, -95, -140, -35, -13, 11, -90, -116, 42, 65, -75, 35, 61}
, {-90, 7, -37, -45, -92, -24, -54, -4, -44, -4, -94, 116, -29, -29, 11, 24, -69, -7, 79, -45, -59, -85, -151, -44, -136, -19, -103, 68, -77, -77, 13, 38}
}
, {{17, -83, -102, -65, 16, -3, -97, 9, 29, 7, -17, -27, -5, 13, -23, -4, -11, 28, -20, -2, -74, -73, -18, -67, -30, -125, 17, 135, -97, -91, 91, 21}
, {8, -7, -61, -36, -26, 17, -206, -33, 5, 15, 16, -47, 4, -73, -4, -7, -46, -38, 7, 0, -66, 19, -67, -26, -105, -101, -41, 26, 25, 53, -76, -80}
}
, {{-76, 44, 1, -23, 23, 44, -68, -132, 5, 15, 8, 33, -7, 18, -69, -34, -2, -89, 61, 60, 3, 43, 31, 39, -18, -36, 46, 44, -8, -4, 17, -55}
, {-80, 27, -59, -86, -7, 8, -64, -81, 46, -8, 76, -17, 12, -59, -32, 41, 14, -55, -95, -5, -53, 21, -1, 14, -68, -137, 46, -5, 6, 12, -20, -96}
}
, {{-76, 19, -47, 4, -19, -13, -71, 8, -25, -2, -13, -18, -51, -4, 9, -42, -14, 10, -21, -28, -19, -16, 3, 4, 13, -4, -21, -57, -64, -41, 5, -18}
, {-89, -61, -62, -63, -31, -1, -5, -22, -16, -28, -5, -5, 17, 12, 4, -58, -53, -23, -2, 12, -33, -55, -6, -53, -85, -3, -26, -18, -51, -1, 12, 5}
}
, {{-18, -3, 1, -6, -10, -26, -13, -21, 4, 12, 9, 2, -9, -4, -7, -6, -26, -1, 13, 0, -40, -19, 4, -11, -20, 11, 5, 6, -2, 6, 0, -3}
, {-22, -25, -28, -17, -24, -10, -25, -12, -32, 11, -6, 7, -20, 4, -24, -5, -20, -33, 7, -3, -13, -22, -15, 0, -25, 12, -22, -30, -4, -15, -25, -17}
}
, {{-36, 67, -96, 39, 29, -104, -30, -23, -30, -5, 22, 54, 5, -116, 23, 26, 24, 22, -27, -122, -17, -80, 60, 58, 4, -22, 71, -2, -26, -135, -8, -47}
, {-58, -9, -97, -9, 11, 112, -107, 29, 27, 58, 17, -87, -97, -53, -91, -58, 7, -60, -32, -82, -78, -102, 50, 33, -61, -78, 34, 9, -40, -7, -102, -126}
}
, {{-38, 12, -40, 23, -16, -7, -43, -46, 113, -1, -33, -27, -75, -5, 23, -150, -104, -79, -66, 31, 41, -28, -47, -48, -56, -1, -104, -43, -50, 59, 51, 53}
, {140, -11, 7, -82, -53, 7, -27, -28, -60, 5, 48, -16, -101, 11, -26, -82, -93, -127, -78, -64, -13, -37, -69, -50, -88, -29, 0, 71, -63, 61, -22, 97}
}
, {{-70, -64, 20, -44, -13, -16, 32, -71, -14, 5, 16, 13, -76, -10, -10, -40, 6, -92, -16, -79, -15, 4, -70, -20, -187, 40, -57, -106, -78, -56, -52, -112}
, {42, 18, -6, -42, -64, -28, 5, -49, -80, -15, -30, -15, -32, -41, -17, 117, -70, -73, -30, -24, -4, -10, -38, -42, -79, 5, 43, -1, -106, -73, -94, -43}
}
, {{-18, -19, -15, -13, -20, -1, -19, 15, -1, 6, -27, -2, -17, 4, 10, -6, -48, -17, 5, 4, -9, -28, 0, 10, -3, -2, -15, -6, -58, -18, 4, -17}
, {-29, -15, -3, -21, -28, 10, -21, -5, 3, -22, -28, 12, 7, 23, 6, 11, -13, -17, -3, -22, -42, -30, -6, -16, -17, -23, 24, -14, -39, -8, -4, 10}
}
, {{11, -52, -59, 4, -14, -8, -30, 11, -63, -28, 1, -7, -16, -61, 2, -100, -64, -131, -21, -60, 63, -107, -39, -58, 15, -94, -71, -13, 10, -43, 38, -4}
, {34, -41, -100, 16, -3, 14, 15, -153, 10, -11, -42, -3, -32, 11, -12, -131, -83, -91, -45, 58, 14, -62, -59, -77, -19, -85, -28, -55, -6, -19, -8, 52}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_GROUPS
/**
  ******************************************************************************
  * @file    averagepool1d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    21 april 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _AVERAGE_POOLING1D_9_H_
#define _AVERAGE_POOLING1D_9_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS  64
#define INPUT_SAMPLES   23
#define POOL_SIZE       4
#define POOL_STRIDE     4
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

typedef int16_t average_pooling1d_9_output_type[POOL_LENGTH][INPUT_CHANNELS];

#if 0
void average_pooling1d_9(
  const number_t input[INPUT_SAMPLES][INPUT_CHANNELS], 	    // IN
  number_t output[POOL_LENGTH][INPUT_CHANNELS]); 	// OUT
#endif

#undef INPUT_CHANNELS  
#undef INPUT_SAMPLES
#undef POOL_SIZE
#undef POOL_STRIDE
#undef POOL_PAD
#undef POOL_LENGTH

#endif//_AVERAGE_POOLING1D_9_H_
/**
  ******************************************************************************
  * @file    averagepool.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "average_pooling1d_9.h"
#include "number.h"
#endif

#define INPUT_CHANNELS  64
#define INPUT_SAMPLES   23
#define POOL_SIZE       4
#define POOL_STRIDE     4
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

#define ACTIVATION_LINEAR

// For fixed point quantization
#define INPUT_SCALE_FACTOR 7
#define OUTPUT_SCALE_FACTOR 7
#define OUTPUT_ROUND_MODE ROUND_MODE_FLOOR
#define NUMBER_T int16_t
#define LONG_NUMBER_T int32_t


static inline void average_pooling1d_9(
  const NUMBER_T input[INPUT_SAMPLES][INPUT_CHANNELS], 	    // IN
  NUMBER_T output[POOL_LENGTH][INPUT_CHANNELS]) {	// OUT

  unsigned short pos_x, k; 	// loop indexes for output volume
  unsigned int x;
  LONG_NUMBER_T avg, tmp;

  for (k = 0; k < INPUT_CHANNELS; k++) 
    for (pos_x = 0; pos_x < POOL_LENGTH; pos_x++) {
      tmp = 0;
      for (x = 0; x < POOL_SIZE; x++) {
        tmp += input[(pos_x*POOL_STRIDE)+x][k];
      }
#ifdef ACTIVATION_RELU
      if (tmp < 0) {
        tmp = 0;
      }
#elif !defined(ACTIVATION_LINEAR)
#error "Unsupported activation function"
#endif
      avg = tmp / POOL_SIZE;

      output[pos_x][k] = scale_and_clamp_to(NUMBER_T, avg, INPUT_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
    }
}

#undef INPUT_CHANNELS  
#undef INPUT_SAMPLES
#undef POOL_SIZE
#undef POOL_STRIDE
#undef POOL_PAD
#undef POOL_LENGTH
#undef ACTIVATION_LINEAR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
/**
  ******************************************************************************
  * @file    flatten.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 january 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _FLATTEN_9_H_
#define _FLATTEN_9_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define OUTPUT_DIM 320

typedef int16_t flatten_9_output_type[OUTPUT_DIM];

#if 0
void flatten_9(
  const number_t input[5][64], 			      // IN
	number_t output[OUTPUT_DIM]); 			                // OUT
#endif

#undef OUTPUT_DIM

#endif//_FLATTEN_9_H_
/**
  ******************************************************************************
  * @file    flatten.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 2.0.0
  * @date    26 november 2021
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "flatten_9.h"
#include "number.h"
#endif

#define OUTPUT_DIM 320

#define NUMBER_T int16_t
#define LONG_NUMBER_T int32_t

static inline void flatten_9(
  const NUMBER_T input[5][64], 			      // IN
	NUMBER_T output[OUTPUT_DIM]) {			                // OUT

  NUMBER_T *input_flat = (NUMBER_T *)input;

  // Copy data from input to output only if input and output don't point to the same memory address already
  if (input_flat != output) {
    for (size_t i = 0; i < OUTPUT_DIM; i++) {
      output[i] = input_flat[i];
    }
  }
}

#undef OUTPUT_DIM
#undef NUMBER_T
#undef LONG_NUMBER_T
/**
  ******************************************************************************
  * @file    fc.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 january 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _DENSE_27_H_
#define _DENSE_27_H_

#ifndef SINGLE_FILE
#include "number.h"
#include <stdint.h>
#endif

#define INPUT_SAMPLES 320
#define FC_UNITS 128

typedef int16_t dense_27_output_type[FC_UNITS];

#if 0
void dense_27(
  const number_t input[INPUT_SAMPLES], 			      // IN
	const number_t kernel[FC_UNITS][INPUT_SAMPLES],  // IN

	const number_t bias[FC_UNITS],			              // IN

	number_t output[FC_UNITS]); 			                // OUT
#endif

#undef INPUT_SAMPLES
#undef FC_UNITS

#endif//_DENSE_27_H_
/**
  ******************************************************************************
  * @file    fc.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "dense_27.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#elif defined(WITH_NMSIS_NN)
#include "riscv_nnfunctions.h"
#endif

#define INPUT_SAMPLES 320
#define FC_UNITS 128
#define ACTIVATION_RELU

// For fixed point quantization
#define WEIGHTS_SCALE_FACTOR 7
#define BIASES_SCALE_FACTOR 7
#define TMP_SCALE_FACTOR 7
#define INPUT_SCALE_FACTOR 7
#define OUTPUT_SCALE_FACTOR 7
#define OUTPUT_ROUND_MODE ROUND_MODE_FLOOR
#define NUMBER_T int16_t
#define LONG_NUMBER_T int32_t


static inline void dense_27(
  const NUMBER_T input[INPUT_SAMPLES], 			      // IN
	const NUMBER_T kernel[FC_UNITS][INPUT_SAMPLES],  // IN

	const NUMBER_T bias[FC_UNITS],			              // IN

	NUMBER_T output[FC_UNITS]) {			                // OUT

#if !defined(WITH_CMSIS_NN) && !defined(WITH_NMSIS_NN)
  unsigned short k, z; 
  LONG_NUMBER_T output_acc;

  for (k = 0; k < FC_UNITS; k++) { 
    output_acc = 0;
    for (z = 0; z < INPUT_SAMPLES; z++) 
      output_acc = output_acc + ((LONG_NUMBER_T)kernel[k][z] * (LONG_NUMBER_T)input[z]);

    output_acc = scale(NUMBER_T, output_acc, WEIGHTS_SCALE_FACTOR - TMP_SCALE_FACTOR, OUTPUT_ROUND_MODE);

    output_acc += scale(NUMBER_T, (LONG_NUMBER_T)bias[k], BIASES_SCALE_FACTOR - TMP_SCALE_FACTOR - INPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);


    // Activation function
#ifdef ACTIVATION_LINEAR
    // Linear (MEANS NONE)
    output[k] = scale_and_clamp_to(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
#elif defined(ACTIVATION_RELU) || defined(ACTIVATION_RELU6)
    // ReLU
    if (output_acc < 0) {
      output[k] = 0;
    } else {
#if defined(ACTIVATION_RELU6)
      if (output_acc > scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE)) {
        output_acc = scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE);
      }
#endif
      output[k] = scale_and_clamp_to(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
    }
#else
#error "Unsupported activation function"
#endif
  }
#else

#if BIASES_SCALE_FACTOR > WEIGHTS_SCALE_FACTOR
#error "CMSIS-NN does not support BIASES_SCALE_FACTOR larger than WEIGHTS_SCALE_FACTOR"
#endif

  static q15_t bufferA[INPUT_SAMPLES];
#ifdef WITH_CMSIS_NN
  arm_fully_connected_q15(
#elif defined(WITH_NMSIS_NN)
  riscv_fully_connected_q15(
#endif
                             (q15_t*)input,
                             (q15_t*)kernel,
                             INPUT_SAMPLES,
                             FC_UNITS,
                             INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - BIASES_SCALE_FACTOR,
                             INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR,
                             (q15_t*)bias,
                             (q15_t*)output,
                             (q15_t*)bufferA);
#ifdef ACTIVATION_RELU
#ifdef WITH_CMSIS_NN
  arm_relu_q15((q15_t*)output, FC_UNITS);
#elif defined(WITH_NMSIS_NN)
  riscv_relu_q15((q15_t*)output, FC_UNITS);
#endif
#elif !defined(ACTIVATION_LINEAR)
#error "Unsupported activation with CMSIS-NN"
#endif


#endif
}

#undef INPUT_SAMPLES
#undef FC_UNITS
#undef ACTIVATION_RELU
#undef WEIGHTS_SCALE_FACTOR
#undef BIASES_SCALE_FACTOR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
#undef NUMBER_T
#undef LONG_NUMBER_T
/**
  ******************************************************************************
  * @file    weights/fc.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

#define INPUT_SAMPLES 320
#define FC_UNITS 128


const int16_t dense_27_bias[FC_UNITS] = {-26, -39, 34, -45, 0, -17, -19, -94, 74, -54, -18, -15, -28, 15, 83, -8, -62, -19, 18, 54, -8, -8, 59, 23, 2, -46, 36, -9, -28, 46, -23, 40, -25, 125, 40, -11, 41, -67, -15, -41, 26, -16, -1, -9, 58, 123, -7, -52, -40, 30, -9, 64, 19, -8, -10, 10, -7, -8, 38, 119, -8, -30, -68, -8, -15, -10, 18, -24, -10, -5, 108, -58, 39, -9, 58, -6, -15, -15, -7, 79, 15, -8, 39, 19, 52, -24, -19, -9, -8, 1, -9, 12, -57, -3, -13, -7, -8, -41, -16, -59, -2, -20, -59, -25, -20, -11, 46, -13, 0, 18, -66, -32, -8, -2, -25, -8, 57, -18, -69, -3, -8, 32, -27, 79, 18, -9, -7, 53}
;

const int16_t dense_27_kernel[FC_UNITS][INPUT_SAMPLES] = {{0, -15, -35, -35, -54, -9, -4, -9, -15, -3, -18, -24, -54, -5, -47, -28, -17, -17, 0, -25, -17, -33, -11, -17, 0, -5, -6, -17, -14, 0, -29, -8, -1, -46, -21, -12, 6, -14, -13, -84, -19, 2, -27, -57, -39, -6, -36, -15, -9, 6, -48, -27, -8, -28, 0, -19, -19, -21, -22, -20, -24, 5, -17, -54, -11, -42, -45, -69, -58, -13, 1, -55, -18, -11, -23, -30, -36, -7, -45, -32, -12, -13, -9, -93, -13, -39, -19, -23, -3, -1, 3, -7, 2, -29, -24, -20, -25, 28, -15, -58, -21, -31, -17, -26, -13, -3, -4, -49, -25, -2, -3, -14, -12, -1, -32, -44, -27, -71, -50, -2, -22, -17, -1, -66, -11, -2, -19, -53, 2, -36, -39, -86, -80, 4, -14, -58, -4, 5, 7, -55, -23, -13, -53, -14, -2, -16, -13, -40, -48, -43, -17, -5, 4, -8, 0, -13, 5, -16, 2, -20, -28, -28, 9, -50, -8, -14, 0, -48, -18, -12, 1, -46, 2, -17, -41, -19, -13, -8, -59, -5, -4, -32, -1, -6, -25, -20, -4, -49, -21, -15, 3, -35, -14, -9, -3, -34, -78, -4, -18, -1, -25, -16, -16, -9, -30, -21, -55, -11, -1, -8, -19, -34, 2, -22, -9, -14, -10, -20, -2, -17, 5, 1, -12, -18, -10, -4, -22, -71, -1, -9, -2, -63, 6, -2, 20, -6, 2, -16, -53, -21, 2, -2, -52, -2, -1, -36, -57, 3, -21, -16, -19, 3, -11, -18, -1, -10, -18, -4, -6, -28, -2, -17, 5, -1, -32, -9, 6, -4, -46, -21, -26, 3, -5, -17, 5, -31, -40, -52, -19, -19, -19, -23, -12, -23, -22, -16, -32, -4, 1, -19, 21, 15, -16, -9, -4, -54, -11, -12, 17, -31, 0, -14, -115, -19, 5, 3, 29, -8, -25, -23, -32, -8, -5, 3, -8, -41, -11, -7, -18, -53}
, {1, -23, -5, -19, 8, -14, -5, -16, -43, -14, -9, -71, -61, 1, -24, -40, -11, -59, -17, -32, -56, 5, -20, -16, -2, -57, -23, 3, -16, -22, -2, -12, -25, -15, -23, -14, 2, -18, -22, -57, -43, -12, -17, -10, -85, -10, -18, -19, -8, -14, -6, -26, -43, -46, -11, -58, -24, -9, -8, -51, -29, -7, -22, -52, 4, -19, -71, -20, -8, -3, -13, -33, -12, -10, 10, -32, 6, -17, 33, -40, -12, -21, -10, -60, -41, -16, -34, -21, 7, -6, -13, -5, -9, -46, -8, -12, -44, -11, -33, -37, -4, -23, -7, -51, -36, -15, 7, -53, -59, -10, -37, -14, -8, -9, -26, -12, -32, -58, -18, 5, -5, -14, -23, -15, -15, -13, -7, -60, -20, 34, -63, -71, -50, -1, -13, -16, -19, 6, 5, -10, -63, -12, -23, -29, -17, -54, -6, -2, -10, -26, -6, -15, -2, -12, -19, -3, 3, 2, -1, -10, -37, -7, -29, -39, -4, -36, -17, -41, -39, 5, -48, -49, -35, -2, -37, -25, -10, -14, -61, -40, -21, -62, -13, -27, -48, 1, 3, -29, -5, -21, -6, -30, -13, 2, 3, -2, -50, -16, -10, 13, -50, -20, 4, -67, -68, -32, -18, 4, -11, -47, -17, -20, 21, -37, 3, 4, -13, -46, -17, 6, 0, -17, -2, -23, -16, -39, -40, 8, -2, -38, -6, -63, -49, -18, -36, -49, 4, -5, -62, -17, 3, 0, -36, -43, -10, -63, -77, 0, -82, -14, -10, 22, -27, -6, 2, -44, -21, -20, -12, -95, -17, 2, -1, 0, -14, -19, -20, -47, -31, -9, -27, 3, -23, -22, 3, -5, -20, -58, -16, 1, -21, 2, -22, -14, -2, -22, -9, -21, -9, -33, -6, -35, -1, -21, -4, -65, -36, -5, -26, -61, -46, 4, 3, 8, -8, -12, -54, -45, 2, -68, 24, -26, -45, -17, -1, -23, -10, -9, -5, -76}
, {12, 62, -66, -72, -84, -18, 0, 72, 110, 7, -9, -28, 61, 30, -4, -52, 84, 59, 6, -70, -84, 37, -52, -18, 0, -42, -11, 53, -33, 38, 29, 40, 27, 85, -61, -103, -16, -8, -25, -76, -25, -28, 50, 39, -14, -23, 42, -28, -7, -3, -171, -1, -88, 66, 38, -83, -11, -30, -12, -93, -110, -75, -9, 43, 3, -27, -23, -12, -8, -7, 86, 1, 59, -12, -17, -150, -3, 50, 22, -93, 16, -8, -18, -9, 13, 37, -141, 7, 1, -88, -3, -57, -19, 18, -26, -13, 51, 56, -144, -33, 17, 38, 20, 29, 0, 14, -45, -37, 28, 5, 28, 62, 14, -7, -174, -50, 18, -35, 19, 18, -2, 39, 1, -17, -22, -23, 9, 23, -20, -9, -126, -50, -18, 1, -16, 9, 2, -1, -13, 35, 1, -14, -17, -71, -36, 22, -14, 22, -57, 114, 2, -10, 4, -24, -5, -4, -31, -31, 46, 29, -3, 135, 48, -69, 15, -7, -5, 4, -18, -45, 55, -20, 38, -7, 33, 11, 20, -1, -95, 56, -64, -99, 114, 48, -5, 54, -12, 12, -87, 36, 0, -8, 6, -41, -176, 7, 179, 15, -13, -2, 33, -2, 2, -52, 6, 93, 71, -111, -44, 28, 12, 31, 22, -9, -39, 10, -6, 77, -2, 20, -4, -14, 92, 57, 8, 53, -27, -71, 11, 1, 9, -40, 29, -6, 44, 34, 2, -8, 32, 46, -59, -2, 19, -43, 17, -37, 43, 7, 12, 55, 30, 31, 47, -64, -10, 53, -20, -85, -147, 44, 25, 7, -4, -20, 90, 15, -15, 22, 73, 10, 8, -161, -25, 40, -3, 48, -39, -23, -118, -3, 19, -119, 8, 103, -2, -24, 42, 9, -6, -30, 28, -66, -52, 28, 2, -18, -32, -43, -48, -7, 31, 7, -13, 8, 86, 6, -164, -24, 69, -97, 48, -48, 62, 53, 14, 26, -159, 19, 24, 19}
, {-15, -17, -27, 4, -18, -10, -7, -3, -63, -3, 4, -141, -38, 135, -82, -21, -47, -22, 3, -59, 59, 15, -19, -2, -10, -17, -6, -21, -17, 49, -63, 88, -12, -86, -137, -55, -56, -17, -18, -35, -91, -3, -96, -37, -17, 6, 159, -108, -29, -13, 11, -83, -57, -7, -50, -130, -5, -19, -10, 41, -123, 75, 3, -51, -12, 40, -56, -71, -28, 6, 59, 22, 42, -15, -6, -78, -98, -13, -102, -15, -34, -23, -19, -53, -60, 23, -19, -19, -7, 9, -23, 1, -21, -73, -49, -10, -28, 8, -11, -58, 21, 7, -14, -72, -67, 3, -165, 97, -100, -5, -9, -30, -33, 11, 217, 18, -64, 87, -28, -69, -46, -7, -21, -55, -12, -2, -9, -73, -4, 14, -161, 31, 4, -23, -2, -117, -44, 5, -17, -90, -65, -32, -25, -86, -4, -38, -17, -65, -122, -16, -68, 2, 5, -60, 3, -7, 3, -27, -58, -56, -40, 60, -13, -35, -31, 23, -7, -142, -11, 5, -130, -86, -53, -13, -40, -113, -20, 7, -73, -49, -16, 29, -1, -12, -87, -1, -13, -78, -24, -43, -6, -66, -20, -82, -84, -50, -2, -22, -16, -109, 67, -2, -8, -18, -85, -31, -106, -82, -23, -74, -8, -169, -156, 25, -66, -18, 4, -68, -19, 5, -9, -89, -20, 0, 11, 118, 68, -128, -41, 99, -10, -195, -117, 3, -49, 5, -74, -17, -46, -117, 87, 15, 207, 13, -100, 147, 74, -13, -147, -18, -15, -86, -31, -54, -8, -27, -3, -161, -65, 20, -8, 1, 58, -34, 16, -14, -11, -21, -119, -29, -103, 3, 25, -7, -1, -48, -150, -62, -107, 3, -16, 42, -17, -5, -9, 108, 23, -21, -31, -27, -25, -49, -71, 64, 2, -128, -70, 5, -36, -33, -66, -6, 35, -126, -39, -11, -12, -76, -20, 51, -140, -33, -48, 4, 2, 13, -61, -36, -3, -37}
, {-11, 59, -33, 11, 27, 19, -57, 37, 7, -6, 12, 31, -68, 16, -23, -38, 14, -31, 20, 3, -11, 35, 35, -2, 12, 13, -2, -20, 26, 58, 62, 27, 28, -132, -24, 37, -16, -51, 7, 9, 48, -18, -56, -109, 9, 13, -9, 30, -61, 4, -16, 32, 28, 50, -5, 91, 44, -11, 21, 51, 59, -13, 2, -46, -8, -82, -59, 17, 167, 4, -76, 51, -32, 14, 14, -55, -62, 61, 40, -8, -16, 72, -4, 30, -45, -82, -69, -4, 15, -22, 2, 54, 13, -30, 35, 0, -25, 24, 56, -6, -35, 45, 10, -54, 23, 38, -46, -5, -27, 12, 18, 34, 27, -6, 5, 94, 83, 82, 68, -7, 65, 21, 12, 44, -137, 23, -8, -59, -6, 62, 19, -2, 14, 0, -7, -2, 37, -5, 22, 38, -60, 71, 85, -25, 31, 44, 7, -26, -9, -111, 22, -11, 18, 35, -5, -13, 22, -56, -34, 96, 105, -10, 72, 11, 20, -112, -8, 2, 29, 55, -83, -96, 8, -7, 7, 37, -11, -7, -27, -64, -15, 22, 52, -11, -4, -36, 5, -13, -4, 40, -27, -202, 4, -48, -1, 19, -54, -7, 122, -17, 71, -6, 8, 24, -52, -24, -66, -113, 25, -26, -7, -11, -37, 41, -50, 17, 2, 4, -2, 35, 10, 89, 43, -54, -2, -11, 129, -30, 16, 88, 12, -81, -1, 7, 110, -55, -20, -13, 21, -53, 96, 17, 61, 95, 21, -74, 27, 4, -25, 11, -5, -31, -93, -35, -7, -159, -1, 42, 75, -30, -79, 9, -121, -57, 51, 18, 30, 25, -32, -73, 94, 10, 5, 11, 18, -16, 69, -140, 38, 8, 4, 69, -24, -42, 17, 0, 108, 6, -11, 5, 7, -10, -126, -35, -20, 45, 18, 12, -162, -52, -61, -10, 11, 11, 9, 16, 49, -90, -14, -47, -33, -57, -18, -57, -18, 130, 43, 184, -18, -54}
, {-4, -68, 21, -79, -54, -5, -47, 31, -61, 1, -8, -13, -78, -17, -35, 14, -13, -17, 5, -74, -45, -72, -51, -16, -20, -55, -22, 5, -9, -32, -8, -35, -23, -27, -28, -18, 1, -8, -15, -75, 2, -19, 7, -31, -81, -4, -64, -40, -13, -4, -9, -32, -117, 39, -71, -21, -54, -9, -9, -54, 5, -46, 5, -43, -21, 77, 12, -38, -52, -2, 13, 31, -22, -10, -8, 26, -40, -1, -57, 74, -5, -55, -7, -63, -55, -71, -17, -13, -9, -26, 6, -10, -11, -36, 20, -40, 5, -76, -129, 55, -14, -62, -10, -48, 35, -12, 2, -10, -70, 4, -38, -22, -1, 4, 9, -11, -60, -21, -68, -29, -46, 5, 4, -15, -20, -35, -5, -112, -13, 32, -27, 32, -80, -20, 20, -52, -67, 9, -13, -10, -26, 1, -30, -4, -1, -38, 5, 17, -33, 6, -74, 0, -19, -76, 7, -2, -1, 37, -12, 8, -4, -137, -41, 37, -2, -72, -4, -65, -33, -9, 41, -31, -34, -7, -31, -43, -18, 14, -102, -39, -29, 22, -47, -15, -24, -19, 1, -7, -38, -6, -22, -27, 1, -17, -85, -13, -113, -10, -61, -59, -40, -16, 10, -80, -65, -4, -39, -40, -10, -7, -1, -23, -42, -3, -60, -9, 6, -10, -11, -4, -17, -31, -56, 7, -21, -121, -38, -65, 1, -33, -17, -58, -32, -9, 5, -34, -41, -4, -27, -2, 6, -1, -81, -25, -11, -151, -17, -8, -50, -19, -7, 15, -43, -29, 6, -22, -2, -27, -54, -14, -27, -4, -18, -2, -51, -22, 4, 92, -16, -2, -55, -87, 6, -38, 7, -20, -31, 12, -13, -9, -8, 3, 5, -4, 7, 21, -43, -23, -19, -103, -33, 13, -17, 79, -9, -20, -35, -22, 62, 85, -44, 14, -12, -54, 44, -2, 41, 34, -34, 71, -38, -9, -42, -11, -14, -33, 44, 22, -10, 86}
, {0, -43, -28, -13, 1, -19, -5, -34, -23, -12, 11, -20, -23, -16, 3, -33, -22, -2, -3, -40, -23, -14, 5, -21, 1, -19, 5, -21, -16, -10, 6, -22, 4, 15, -3, -42, 0, 5, -15, -23, -18, 0, -8, -19, -37, -19, -19, -15, 0, -7, -17, 5, -10, 0, -23, -9, -18, -19, -5, -18, -18, 2, -21, -16, -17, -17, -28, -8, -28, -18, 0, -16, -24, -20, 2, -27, -24, -14, -1, -25, -10, -3, -7, -23, -27, -20, -19, -9, -9, -11, 3, -9, -20, -21, 4, 2, -21, -8, 0, -13, 0, 6, 1, 6, -20, -1, -37, -15, -25, -1, -27, -22, 3, -15, 0, 4, -16, -20, -29, -14, -23, -8, 2, -11, -19, -11, -15, -6, -18, -18, -18, -9, -9, -6, -13, -16, -3, -21, -9, -11, -32, -22, -18, -13, -13, -18, -7, -3, -17, 2, -27, -2, -1, -20, -21, -13, -10, 2, -13, -3, 0, 13, -3, -12, 1, -15, -10, -40, -21, -14, 7, 6, -30, -11, -16, -6, -10, 5, -27, -10, -23, -13, 1, -19, -22, -22, -3, 11, -3, -13, -4, -4, 2, 12, -26, -6, 7, -13, 9, 5, -19, -7, -14, -18, -24, -12, -17, -27, -10, -5, -11, -16, -28, 2, -2, -3, -2, -19, -19, -18, -1, -15, -9, -5, -8, 19, -4, -25, 6, -15, -10, -26, -9, -15, -21, -17, -16, 7, -5, -2, 4, -2, 13, -18, -16, 4, -23, -1, -31, -7, -9, -26, -3, -2, -11, -12, -2, -45, -16, -22, 6, -4, -18, 3, -27, -12, 10, -14, -14, -22, 6, -9, -3, -4, -14, -25, 8, 6, -45, 0, -9, -22, -19, -7, -21, -37, 5, -15, -24, 3, 2, 12, -19, -21, 5, 4, -22, -2, 1, -5, -1, -15, 9, 1, 5, -19, -10, -5, 0, -17, 10, -15, -7, 6, -8, -23, -19, -12, -5, -17}
, {15, -120, -60, 4, 75, 2, 7, -42, -45, -8, 4, -69, -1, -39, 101, -66, -38, 18, 20, 47, -165, 9, -101, -13, -1, -25, 7, -6, -15, -34, -41, -62, -29, -90, -3, -46, 34, -64, -19, -79, 0, -24, -113, 5, -68, -7, 62, -19, -95, 5, -225, -29, 34, -120, 77, -6, -17, 24, 0, -53, -6, 23, -6, 8, 1, -64, 63, 68, -190, 0, 60, -90, 3, 13, 11, 111, -29, -73, -6, -15, -35, -10, -4, 15, 107, -97, 44, 9, -11, 21, 20, -23, -36, -66, 119, 2, 31, 8, 57, -18, 25, 129, 23, -33, 25, 43, -40, -87, 41, -6, -53, 34, -27, -3, 48, 32, -14, 2, 43, 55, 82, 5, 2, 23, -50, -51, 7, -34, 8, -47, 24, -27, 86, -23, -36, -63, 6, -13, -1, -27, -31, -18, 78, -77, -4, 35, -18, 5, -24, 45, -32, -4, 3, 8, -6, -54, -22, -11, -42, 15, 38, 53, 50, -102, -15, 22, -12, -52, -11, -75, 35, -73, -39, -18, -16, 7, 80, -5, 13, 33, 9, 30, 4, -90, -68, -10, -5, -52, 21, -44, 12, -78, 16, 26, 21, 42, -61, -2, -93, 28, 11, 2, 0, 76, 22, 61, 24, -18, -49, -32, -3, -16, -29, -12, 72, 4, 16, 7, 16, -41, -14, -30, -8, -49, 27, -39, 5, -23, 5, 41, -23, 50, -13, -29, 71, 64, -15, -9, -16, 16, -49, -8, 69, 4, -2, -44, 33, -33, -22, 24, 4, -38, -14, 24, 6, 90, 13, 41, 2, 23, 72, 5, -153, -36, 30, -4, 0, 66, -1, -21, -8, 19, 7, -56, -5, -82, 84, 55, 19, 0, -19, 12, 19, 7, -12, -20, 75, 25, 25, -53, 161, 7, 28, -76, -2, 36, 11, 14, -102, 38, -8, -6, 39, 30, 36, 17, 149, -51, -92, 113, 69, -20, -18, -17, -2, 28, -15, -22, 8, 49}
, {-6, -13, -91, -129, 9, -12, 28, -84, 32, -6, -12, 92, 27, 21, -144, -48, -12, -100, -13, -118, -57, -139, -2, -4, -5, 14, -16, -122, 47, -50, -125, 10, -39, 107, -28, 22, 38, 50, 4, -11, -103, 2, 83, 17, -116, -24, 19, -20, -59, 6, 88, -13, -61, 85, 76, -36, -22, -2, -14, -120, 9, -12, 2, 13, -13, 118, -88, -234, -15, 15, -11, -85, 23, 8, -21, 12, -103, 67, -32, -33, 26, -2, 11, 41, -11, 14, 37, 23, 14, 64, -5, -1, 20, -28, -38, 33, -43, 39, 2, -28, 100, -94, -5, 10, -36, 15, -28, -19, -5, 1, -39, -44, 35, -10, 57, -75, -152, 98, 22, -27, -14, -27, -6, -186, 55, -13, 22, -25, -6, -15, -25, -180, -22, -8, -18, -29, 31, 6, 4, 47, -97, 11, 22, -20, 62, -37, 6, -126, -10, 3, -36, 2, -3, 31, -19, 10, 24, 17, 149, 99, 9, 49, 9, -23, 94, -17, 9, -18, -14, 14, -76, -33, 26, -11, 19, 9, -85, -18, 59, -59, -75, -83, 69, -58, -35, -73, -1, -78, -31, -114, -11, -14, -17, -29, -87, -110, 102, -3, -60, -2, 27, -3, 3, -21, -16, 101, -300, -56, -66, 79, 5, -34, -19, 101, -47, -7, 12, 56, -17, 1, 38, -56, -126, 107, 19, -15, -9, -55, 76, -33, -4, -28, 23, -11, 11, 59, -37, 6, -20, 41, -64, -16, 29, 106, 1, -42, 46, -42, -2, -21, 21, 16, 80, -46, 0, 52, -38, -30, -18, 44, -19, -14, -19, 39, -53, -22, 6, -70, -16, 18, -12, 28, 62, 40, 13, -92, -32, 17, 0, 4, -22, 21, -2, 140, 38, 1, -18, -3, 23, -36, -35, 15, 7, 120, -18, 37, -51, -1, 90, -92, 71, -4, -93, -40, -56, 5, 69, -37, 31, 55, -6, -26, -18, -40, 18, 51, 25, 86, 6, 21}
, {-17, -70, -14, -18, -11, -12, 33, -79, -110, 5, 1, -10, -56, 23, 29, -25, -25, -40, -20, 38, -88, -134, -22, -18, -3, 47, 13, -73, -58, -33, 22, -16, 35, 44, -133, -83, -225, -19, -7, -19, -60, -2, 30, -138, -86, -6, 46, 0, 15, 12, -130, 0, 26, -68, -2, -136, -151, -24, 37, -96, -60, 69, -4, -70, 30, -81, 36, 3, -40, 11, 20, -139, 14, -1, 6, -64, -21, -38, 88, -27, 25, -17, -7, 28, 23, 27, -31, -2, -4, 6, 8, -27, -42, -39, 109, -52, -169, 52, -42, -12, -121, -3, 13, -6, 6, 6, 75, 4, -25, 4, 80, -23, 3, 7, -143, -44, 33, -257, -34, -9, -62, -22, 46, 14, -183, -95, -8, 35, -6, 42, 22, 74, -170, 2, 5, 105, -74, -5, 7, 32, -13, -22, -57, 105, -30, -26, 10, 14, 48, 5, -28, 8, 1, -29, 10, -54, 76, -11, -64, 33, 34, -53, -5, 76, 41, 104, -9, -19, 57, -9, -4, -35, -73, -4, -63, 29, -1, 2, -15, -52, 55, -32, 3, -21, -60, 83, 82, -33, -135, -37, -15, -67, 28, -152, -126, -88, 81, 0, 101, -110, -148, 0, 13, 73, -30, 33, -8, 10, 37, -84, -10, 2, -38, -22, 33, -13, 20, -110, 0, -12, 32, -79, 156, 53, -48, 11, 6, -85, 115, 115, 10, -118, -100, 7, 98, 103, -139, -14, 45, -23, 33, 10, 48, 34, 17, 46, -87, -17, -53, 15, -16, -98, 159, 56, -38, 110, 98, -3, 12, 128, -26, -4, 110, -122, -36, -4, -2, 38, 4, 22, -87, -26, -36, -56, -4, -62, -4, 21, 33, -3, 5, -68, 8, -18, 25, -128, -89, 103, 57, -22, 77, -83, -78, 42, -46, -113, -16, -12, -61, 58, 11, -12, -82, 16, 38, -14, 23, 81, -40, 70, 69, 75, 1, -15, 41, -5, 86, -6, -11, 21}
, {0, -14, -14, 13, -11, -11, -4, -7, -10, -2, -5, -25, -47, 12, -9, 5, -19, -10, -8, -39, -14, -20, -18, 1, 14, 3, 0, 8, -8, -9, 7, -18, -2, -18, 6, -12, -1, -17, -10, -34, -4, -9, -6, -5, -30, 2, -20, 2, 2, -24, -8, 7, -21, -2, -45, -15, -4, -39, -17, -10, -1, 0, -2, 11, 1, 15, -19, -41, -9, 16, 1, 27, 2, 1, -3, -3, -35, 16, -15, 6, -4, -14, 9, -34, -11, -10, -5, 10, -1, 4, -4, -7, 16, -7, 9, 7, -18, -31, 9, -7, -10, 4, -6, -23, -17, 6, -19, -20, -50, 0, -19, -8, 12, 7, 7, 0, 16, -13, -4, -15, -27, -24, 14, 15, -3, -3, 4, 6, 19, -53, -14, -10, -9, 8, 18, -27, 1, 15, 7, 8, -9, 6, 26, -1, 3, -5, 13, -1, -15, -4, -32, 7, 18, -13, 13, 7, -7, -4, 3, 5, -3, -29, 21, -11, 1, 0, -20, -31, -9, 8, -15, -11, -3, -6, -7, -17, 1, -12, -4, -9, 2, -17, -21, -16, -6, -10, 6, 46, -15, 6, -6, -4, 18, -32, -24, -2, -16, -7, -13, -41, -20, -9, -11, -29, -18, 4, 2, -17, 0, 7, 5, 25, -30, -8, -18, -8, -2, -6, -16, 3, -4, -11, -10, 0, -16, -12, 0, -23, 6, -14, -17, -7, -14, 12, -7, -17, -11, -9, -23, -2, 15, -10, 18, -20, -41, -2, -34, 8, -17, -32, -2, 0, -10, -15, -8, -5, 18, -16, -18, -5, 0, 2, -4, -14, 6, -13, -9, -14, -33, -1, 14, 5, 4, -14, -2, -7, 2, -13, 4, 3, 2, 10, -19, 13, 8, -10, 11, -3, -10, -12, -5, -6, -9, 3, -2, -4, -16, -7, -11, -38, -20, -13, -13, -2, -7, -14, -2, 9, -29, -16, -26, 8, -2, 6, 12, 17, -5, -10, 6, -11}
, {-1, -76, 2, 7, -1, 6, 11, -56, -87, 3, 10, 37, -155, 80, -22, -58, 0, -53, -19, -9, -23, -41, -6, -5, 6, -17, -7, 53, -21, -85, 75, 2, -16, 53, 8, -24, 92, -21, -23, 4, 45, 8, 55, 21, 4, 17, 31, -78, -11, -22, -8, 155, -75, 2, -79, 33, -81, -46, -9, 13, 62, -30, 1, 27, -5, 59, 26, 36, 133, 6, 23, 17, 20, -21, 21, 7, -42, -134, 14, 86, -19, 75, 4, 68, 28, -15, 68, -21, 4, 37, -6, 47, -16, 51, -71, 14, 77, 83, 2, 57, -82, -126, -13, 68, 33, 4, 27, -114, 28, 12, -62, 49, 22, 2, -1, 77, 72, 53, -56, 23, 50, 16, -7, 36, 39, -30, -2, -120, -7, -68, 54, -113, -73, -22, -15, 37, -77, -8, 10, 52, -15, -51, -32, -12, 31, -52, -10, -40, -89, 75, -89, -6, -12, -67, 1, 7, -16, 7, 113, 56, 11, -71, 34, -1, -92, 137, 5, -43, -26, -1, 9, -31, -48, -10, 4, 9, -11, 1, -2, -97, 42, -22, -1, 39, -145, 42, -19, -32, 58, -70, -9, -35, -19, -6, -64, 3, -96, -3, -20, -63, 25, 5, 10, 111, -43, -70, 22, -29, 19, -45, -20, -50, -22, -53, 5, -8, -15, 16, -3, 58, -3, -31, -40, -6, -90, -43, 119, -66, -42, -62, 1, -24, -57, -14, 19, -72, -102, -2, 64, -8, 26, -21, 58, -34, -49, -101, -9, -37, 12, 14, -18, -171, -22, -74, 7, -47, 5, 65, 29, 44, -191, -6, -37, -47, 16, -13, 22, -10, -7, 16, 83, -25, -50, 45, -5, 83, 40, 61, 6, -10, 0, 47, 15, -5, -58, -42, -26, 25, -23, -65, 17, 6, 30, -63, -11, 11, -1, -16, -81, 47, -16, -9, 18, 17, 46, 2, 64, -102, 17, -15, 23, -3, -39, 22, -16, 22, -57, -7, -16, 83}
, {-21, -54, -32, 1, 4, -2, -2, -29, -47, -15, 1, -29, -59, -1, -12, -34, -3, -8, -6, -41, -59, -5, -58, -4, -8, -20, -19, 6, -9, -52, -8, 2, -26, -60, -11, 11, 1, -5, -3, -83, -39, 4, 0, -15, -55, -14, -29, -17, -21, 6, 2, -11, -32, -20, -5, -34, -17, 2, -3, 1, -17, -19, -11, 5, 0, 44, 14, 13, -4, -6, 6, -52, -19, -5, -4, -27, -43, -13, 6, -19, -21, -45, 0, -45, 4, -18, 26, 0, -6, 18, 0, -22, -7, 8, -13, 10, -13, -32, 2, -21, 5, -10, -2, -53, -38, -13, -34, -34, -33, -21, -78, -8, -6, -16, 25, -15, -51, -40, -21, -25, -18, -12, -11, -60, -4, -14, -21, -37, 3, -27, 1, -17, -20, -7, 1, 30, -28, -2, 0, -34, -30, 2, -10, 1, -5, -54, -6, -29, -48, -36, -18, -13, 0, -34, -18, -9, 0, -13, 7, -6, -4, -25, -5, -11, -20, -47, 6, -19, -12, -22, -25, -38, -18, 3, -46, -14, -3, 1, -9, -22, -14, -37, 24, -39, -42, -8, -16, -2, -25, -21, -13, -35, -9, 12, 21, -47, -48, -9, -35, -39, -36, -14, 3, -18, -32, 4, -24, -34, -13, -34, -8, -68, 8, -55, -37, -1, -14, -12, -4, -21, -2, -2, -25, -36, -16, -85, -20, -33, -14, -41, 4, -38, -5, 5, 16, -55, -42, 7, -31, -2, -17, -16, -3, -20, 3, -52, -50, -25, -38, -12, -23, 1, -17, -6, -19, -43, -4, -28, -36, -49, -51, -17, -29, -16, -26, -13, -21, 11, -39, -23, 9, -13, -22, -27, 6, -50, -25, -4, -42, 6, -4, -7, -7, 0, -15, -29, -1, -40, 3, -79, -22, -36, -6, -12, -21, -51, -39, -17, 5, -39, -54, -4, -22, -4, -15, -3, -16, -1, -12, -39, 38, -8, -45, -21, -21, -56, -5, -14, -20, 0}
, {1, 9, -38, -140, 1, 0, -25, -43, -55, 1, -7, -137, -24, -34, -78, -68, 83, 0, 11, -67, -56, -17, -199, -4, -2, 13, 1, 89, -44, 122, -93, -40, 50, -18, -21, -116, 127, -56, 10, -16, -158, -7, 28, 4, 4, 7, 12, -6, -3, 5, 40, 21, 1, 19, -305, 65, -45, -1, -23, 52, 50, 9, 8, 6, -20, -38, -89, -55, -42, 1, -70, -197, -27, -14, -18, 50, -86, -84, -47, -3, 75, 66, -9, -219, 66, -31, -35, -21, -8, 75, 2, 43, 31, 10, 11, 28, 3, -28, 31, -110, -33, 45, -16, -35, -158, -17, 0, 11, -279, 14, 76, 121, -39, 4, 50, 51, -29, -31, -149, 98, 1, -38, -15, -118, 30, -36, 8, -12, -5, -2, -79, 101, 86, -8, -100, 9, -22, -14, 4, 86, 85, 20, -36, -35, 4, -6, 1, 26, 56, 51, -20, -17, -7, -19, 20, -14, -19, -44, -32, 69, -53, -19, -37, -62, 71, -3, 7, 25, -77, 18, -63, 3, -26, -2, 13, 90, 76, -1, -60, 69, 1, -55, 98, -3, -78, 24, -12, -80, 79, 22, -1, 13, -2, -200, -24, 70, 29, 0, 61, -10, 58, 14, 1, 6, 38, 125, 120, -15, -1, 79, -5, 56, -86, -32, -113, -16, -6, -46, -4, 4, 7, 50, 203, -131, -109, 47, 97, -117, 56, 29, 16, -71, -100, 35, -30, 26, -49, 13, -59, -220, -67, 1, -53, -3, -9, 56, 11, -83, 100, -48, -1, 66, -30, -9, 1, -6, -15, -130, -18, -41, 101, -15, 12, 117, 38, 17, 14, -10, 8, -179, 29, 98, -4, -27, 5, 15, -103, -67, 20, -14, -17, -92, 6, 24, 17, 16, -8, -39, -25, 82, 46, -60, -38, 9, 2, -60, -81, -5, 35, 33, -152, 4, 1, -26, 88, 9, -24, 79, -2, 44, 111, -97, 39, -26, -11, -113, 71, -11, 32, -17}
, {13, 0, -116, -34, 188, 22, 45, -13, 3, 0, 2, -66, -46, 53, -43, -11, -18, 80, 18, -47, -16, -9, -34, 22, 1, 83, 13, 2, -4, -28, -55, 35, -14, 32, 68, -13, -5, 79, 17, -113, -44, 3, 60, 64, -62, 6, 100, 9, -30, -7, 94, 28, -38, 65, -53, -26, -37, 11, 11, -24, 40, 27, -6, 31, 11, 1, -5, -29, -76, 11, -28, 18, 92, 22, -7, 8, -93, 13, -81, 22, 10, 31, -6, -76, 11, -7, 40, 9, 10, -36, 12, -25, 1, -6, -53, -5, -66, -107, 67, 16, -25, -20, 21, -60, 41, 20, -137, -164, -60, 15, -28, -90, 1, 16, 45, -44, 66, -111, -44, 7, -19, 32, -7, 84, -11, 13, 5, -121, 20, -59, 22, 108, -82, 15, -56, -14, 23, 3, 8, -21, -144, -14, -12, -32, 8, -20, 19, -8, -37, -85, -84, 7, 0, 50, 4, 20, 4, 16, -49, 43, 23, -198, 25, -32, 82, -12, 3, -96, -7, -7, 27, -5, 20, -3, -11, 21, 18, 6, 25, 42, -17, 14, -53, 25, 26, 7, 2, -6, 38, 11, 20, -34, 9, -9, 12, -80, 47, -2, -19, 60, 63, 12, 6, 55, -73, -2, 53, -26, 7, 17, 8, -30, 49, -69, 7, -6, 21, 47, 4, -26, -2, 94, 29, -2, -5, -29, 89, 23, 47, -2, -6, -77, -5, -3, 17, 8, -69, 10, 84, -24, 4, 7, 8, 12, -10, 46, -61, -33, 13, 14, 9, -53, 20, 63, 12, -46, 14, -18, 72, -11, 103, -3, -59, 30, 61, 0, 20, 24, -133, 77, -4, -15, 27, 99, 21, 4, 39, 41, -23, -5, 10, 66, 3, 4, 3, 39, 90, 19, 5, 59, 23, -17, 97, 30, 13, -25, 12, 7, 11, 1, -39, 3, 13, 69, 26, -5, -102, 52, -112, 23, -10, 14, 58, 13, 4, -141, 19, 5, 12, -34}
, {-16, -16, -6, -19, -16, -18, 1, -16, -2, -8, -20, 2, -14, -9, -12, -16, -14, -22, -13, -6, -24, -7, -10, -13, -18, -1, -3, -18, 3, 4, -5, -6, -4, -12, 0, -12, 2, 0, -11, -19, -20, -20, 4, -16, -14, -9, -24, -8, 4, -9, 3, -19, -4, 3, -12, 6, -10, 0, 0, -4, 7, -2, -12, -8, -17, -1, -16, -9, -8, -5, -7, 16, 0, -10, -1, 4, 6, -8, -4, -5, 1, -17, -7, -18, 2, -2, -12, -13, 3, -15, -4, -13, -11, -2, -5, -8, -11, -12, -23, -9, 3, -16, -16, -14, -20, -7, -20, -8, -4, -3, -6, -3, 2, -15, -12, -2, 5, -21, -6, -13, -9, -2, -16, 6, -19, -18, -3, -21, -6, 4, -16, -20, -11, -20, 7, -16, -9, -20, 8, 5, 0, -1, -19, -14, -23, 4, -7, -20, -5, -4, -17, -13, -2, -16, -2, 2, 0, -5, -16, -4, -16, 11, 6, 1, -17, -1, -8, 5, -18, 6, -6, 2, 6, -4, -20, -1, -10, 0, -5, -1, -11, -8, -10, -21, -2, -2, -1, -12, -10, -21, 4, 3, 3, 1, -4, -18, -17, -4, -19, -13, -1, -19, 4, -2, -22, -7, -21, -21, -6, -16, -23, -14, 4, -22, -1, 1, 0, -13, -13, 0, -16, -20, -8, -13, -16, 2, 5, -18, 6, -5, 2, -22, 2, -16, -4, 0, 1, 4, 0, -12, -10, -21, -10, -5, -1, -12, -12, -3, -8, -2, -16, -16, -19, -14, 4, -11, -2, -9, -14, 4, -5, -13, 4, 0, -7, -16, -24, -1, -5, 4, -10, 2, 4, -22, 1, -23, -24, -13, 4, -23, -12, 6, 6, -6, -10, -4, -11, -11, 4, 2, -10, 4, 6, -17, -13, -1, 1, 0, 1, -13, -12, -16, -7, 2, -11, -15, -7, 4, -22, -13, -7, -12, -17, -23, -9, -7, -10, -8, -9, 6}
, {-15, -74, -15, -121, -23, 6, -61, -57, 51, -7, -16, -53, -42, 96, -22, 5, 8, -55, -4, 49, 10, -58, -74, -6, -7, -19, -21, -21, 32, 36, 88, 25, -39, 24, -123, 5, 3, 59, -3, -6, -3, -42, 20, 25, -33, 9, -50, 10, -37, 6, -52, -14, 15, 78, 12, 16, -55, 41, 11, -97, 37, 160, -3, 1, -16, 2, -103, -138, -45, -21, 147, -29, 18, -5, -2, -119, -18, -8, -5, -47, 29, -23, 3, -6, -36, 16, -98, -8, 10, 33, -19, -41, -23, 20, -179, 25, 1, 69, 27, -66, 84, -105, -2, -46, -21, 28, 58, -15, -94, -11, 40, -6, -53, 4, 3, -13, 50, -32, 67, -30, 4, -14, -5, -29, 19, 27, -1, -40, 1, 55, 7, -79, 68, -5, -32, -56, 37, -15, -15, -29, 53, 64, 96, -113, -35, 7, -15, 21, 21, 90, -1, -7, -23, -8, -19, -15, 10, -15, -74, 53, -6, 2, -26, -51, 94, 57, -4, -44, -28, 35, 29, -57, -21, 5, 90, -69, -44, -6, -30, -98, -94, 20, -34, -29, 62, -73, -19, -86, -111, -11, 5, -73, -12, 36, 15, 76, -10, 2, -70, -2, -47, -10, 14, 32, 20, -4, -20, 32, 33, 71, -8, -17, 12, -10, 21, -21, 4, 31, 4, 32, 5, -21, -66, 39, 40, 16, -87, 43, 25, 24, 7, 75, 28, 46, 34, -7, 80, 1, -43, 56, 15, -22, -42, -5, -7, -33, -24, 81, 47, -43, 9, 60, 60, -196, -18, -12, -14, -43, 25, 71, -69, -6, -29, 62, -63, -13, -18, -18, 41, 26, -112, 32, -12, 59, 5, -11, -20, -90, -11, 15, 6, -24, -17, 89, -20, 62, 116, 41, -29, 45, 2, 11, -48, 35, -41, 16, 0, 25, 96, 58, 2, -10, -82, -25, -5, 7, 20, 37, 26, -37, 3, -5, 23, 15, 5, 90, 40, -115, -12, 2}
, {17, -16, -11, -78, -23, 12, 7, -74, -23, 7, -7, -7, -76, -16, -21, 0, -17, -19, -19, -29, -13, 31, -35, -5, -13, -22, -3, -6, 2, -14, -19, -4, 3, 0, -2, -10, -9, -4, 21, -52, -6, 19, -8, 7, -29, 1, 1, -14, 116, 1, -2, -14, 17, -44, 43, -23, 1, 0, -14, 8, -4, 2, -21, -13, -11, -3, -26, 9, -29, -12, -11, -20, -6, 5, -11, -21, -14, 1, -13, -21, 4, -2, -18, -26, -21, 0, 0, -7, 6, -23, -23, -21, -6, -13, -14, -14, -11, -18, 7, -22, -8, -13, 13, -22, -1, 6, -9, 16, -16, -3, -36, -6, 6, 19, 4, -8, -9, -15, -38, -17, -3, -17, -1, -4, -11, 3, -16, -20, -15, -18, -10, -47, -42, -7, -9, 32, -5, 1, -11, -15, -29, -9, -17, -7, -2, -8, -16, -34, -15, -20, 5, 5, -11, -13, -1, -5, -5, -21, -5, -23, 1, 7, -4, -14, -14, -2, 5, -19, -24, -17, -21, -29, -7, -24, -34, -16, -1, -3, -9, 1, -22, -7, -19, 2, -4, -3, -4, 3, 3, 6, 1, -9, -15, -21, -5, -39, -42, -21, 10, 9, -6, 11, -4, -17, -14, -16, -58, -14, 2, -16, -17, -9, -11, 7, -3, -5, -9, 4, -19, -16, 4, -19, -21, -26, -25, 0, -20, -9, -24, -23, 13, -13, -8, -1, -4, 10, -24, 0, -42, -17, -5, 18, -7, 1, 5, -19, -42, -7, -4, -19, -1, -14, -13, -6, -19, -15, -18, -12, -14, -60, -47, -21, -2, 15, -6, -7, 1, -22, -47, -7, -10, -1, -19, -16, -7, -53, -17, -17, -6, 2, -8, -16, 2, 2, -9, -1, -1, -11, -15, 24, -21, -15, -17, -22, -10, -11, -21, -10, -20, -14, 1, -12, -30, -3, -16, 22, -13, -4, -17, -18, -33, -21, -9, -22, -16, -39, -15, -14, -9, 0}
, {-10, 55, -7, -132, -6, -21, 74, 22, -16, 14, -3, 83, -62, 159, -14, 19, -29, -97, 17, -86, 37, -10, 104, -11, -18, 11, 1, 76, -88, -41, 140, -11, -5, -10, 20, -23, 44, -38, -21, 11, -33, 11, 7, -26, 39, 6, -78, 10, 3, -12, -36, 34, -75, 27, -6, -47, -31, 4, 14, -17, -109, 78, 8, 10, -7, -24, 6, 16, 42, 1, -24, 72, 45, -4, -7, -127, 11, 27, 135, 4, -75, 113, -9, 15, -60, -61, -72, -2, -3, 5, 6, 109, -34, -22, 51, -58, -68, -49, -59, -61, -25, -8, 16, -21, -60, 37, -115, 22, -35, -6, -8, 31, 107, 10, -55, -27, 51, 12, 24, 43, 33, -15, -9, 12, 43, 75, -5, -10, -6, -13, 44, -51, -42, -24, -73, -56, -1, 4, -8, 79, 16, 36, 36, 70, 36, -47, -12, -22, 26, -98, 28, -12, -27, -9, -9, 68, 30, -48, 13, -27, -20, 7, 69, -72, 21, -80, -25, 19, -56, -7, 69, 29, -24, 13, -37, 14, -36, -8, 50, -39, -66, 101, 31, -49, -26, 6, -20, 6, 68, 51, -24, 63, -15, -3, -22, -156, -52, -7, -44, 86, 44, -2, -28, -15, -26, 83, -3, -43, -46, -77, 2, -33, -10, 12, -12, 22, 7, 29, -3, -25, 22, -30, -7, 6, -4, -30, 40, 17, 11, -50, -5, -29, 27, 23, -78, -26, 8, 4, -16, -19, 26, 3, 64, -20, 67, -23, 20, -53, 72, 33, 8, 26, -10, 99, -4, -30, -25, 0, -38, -53, 29, 13, 84, -47, -16, 1, 4, -23, 22, -42, 99, 10, 42, 60, 18, 71, -18, 29, 19, -4, 1, -39, 19, -24, 0, -47, 55, -38, -45, 1, -173, -17, -42, 72, 0, 4, 17, 4, 60, 71, -37, 2, -61, 37, -50, 2, -122, 57, 64, 102, 6, -39, -23, -2, 2, -48, 52, 46, -2, 44}
, {1, 3, 19, 44, -120, 1, -24, 96, 22, 4, -2, 11, -30, -6, -259, 22, -12, 58, 17, -48, -41, -27, 54, 20, 12, -128, -8, -18, 18, -99, 42, -92, -53, -91, 11, -3, -97, -13, 13, -24, 35, 19, -107, -137, -7, 5, -13, 11, 81, -10, -28, -58, 26, -6, -43, -10, -53, 1, 8, -41, 12, 27, -1, -146, 5, -41, -29, -48, -22, 3, -49, -38, 0, 7, 14, -46, 16, 13, -132, 1, 9, 7, 3, 13, -72, 14, -40, 0, 3, 33, -7, -11, -2, -34, 12, -66, 8, 52, -1, -14, -61, -25, 1, -58, 12, 11, -10, 15, -61, -2, 73, -53, 7, 0, 2, -21, -62, -46, 37, 59, -20, 23, 20, -128, 6, -59, -7, -35, 9, 9, -22, -72, -79, 15, 39, -11, 6, 18, 5, -67, -26, 31, -106, 60, 5, 84, 20, -66, -56, -65, 24, -2, 0, -31, 2, -2, 15, -120, 86, 28, 37, -86, -19, 11, -24, 3, 13, -13, 37, 36, 0, 37, 55, 3, -126, -18, 22, -1, -47, -94, 69, 15, 52, 23, 22, 5, 14, -32, 10, 17, -5, -44, 4, 50, 25, -2, -29, 4, -151, 33, -17, 8, 2, 71, -76, -43, -36, 72, 17, -19, 16, -26, -29, 42, 48, 21, 6, 18, -7, 8, 23, 2, -61, -24, 8, -10, -94, 3, -133, -61, 10, 5, 31, 4, -48, -24, 9, 16, -27, -28, -48, -20, -29, 31, 26, -95, -48, 25, -13, 24, 15, -40, -77, 9, 7, -5, -2, -8, -3, 34, -13, 4, -85, -59, -22, 21, -6, 28, -109, -73, -27, 40, 5, -84, 8, -62, -70, 76, 12, 20, -2, -62, 18, -11, 6, -139, -187, -114, -6, 44, 17, 5, -16, -39, 4, 29, 78, 3, 29, -137, 29, 18, -51, 21, 51, -13, -108, 18, 24, 19, 25, -83, -68, 14, 0, -139, -55, -36, 7, -66}
, {-3, 3, -10, -18, -8, 5, -21, -4, -2, -22, -4, 6, 1, -8, -22, -20, -10, -10, 1, -17, 2, -5, 1, -14, -13, -7, -6, 7, 6, -19, -12, -16, -12, 10, -9, -14, -21, 0, 2, -2, 2, 4, -15, -9, -12, 5, -12, -13, 3, -6, -3, -14, -10, -1, -30, -6, -6, -17, -1, -5, -4, -3, 1, 4, 5, -8, -15, -2, -1, -12, -5, -4, -7, 1, -7, -20, 5, -12, -16, -14, -14, -4, 1, -19, -3, -3, -3, -13, 6, -1, -10, -8, 5, -2, 1, -9, -13, -15, 5, 6, 1, -7, -11, -20, -18, -4, 2, -3, -5, 1, -21, -1, -14, -7, -3, -20, 5, -22, -14, 2, -13, -17, -19, -3, -19, 6, -18, 2, 5, -14, -21, -6, 5, -16, -5, 3, 6, -20, -8, -12, -14, -8, 1, -13, -16, -8, 5, -14, -1, 0, -1, -8, -19, 4, 1, -1, -10, -13, -2, -11, -21, -10, -16, -1, 4, -15, 1, -19, -21, 1, -19, -16, -9, 4, 3, -18, -22, -14, 5, -16, -3, -5, -13, 6, 4, -21, 2, 6, 4, -1, -7, 4, -20, -4, -3, -8, 1, -17, 2, 0, -22, 1, 1, -7, 5, -7, 4, -14, -9, -17, -6, -17, -11, -14, -11, 6, -10, 3, -12, -8, -3, 6, -5, -17, 2, -14, -14, -19, -8, 1, -19, 4, -20, -4, 3, -2, -19, -7, -10, -13, 4, -14, 5, -10, -2, -10, 0, -19, -11, 1, -9, -5, -4, -1, -3, -20, -9, -15, -16, -18, 0, -23, -13, 5, 0, 0, -7, 5, -11, -8, 0, -17, -20, 5, -19, -1, -7, -17, 5, -3, -2, 2, -2, -1, -6, -4, 5, 5, -20, 9, -13, -21, -10, -8, -20, -4, -4, 0, -22, -15, -6, -14, -18, -4, 1, -13, -5, -10, -13, -12, 7, 6, 0, 3, -11, -22, -15, 2, -11, -7}
, {6, 1, -11, -6, -8, 6, -4, -13, 2, 3, -1, -16, -4, -7, -4, -8, -10, -20, 1, -5, -9, -19, -20, -1, 4, -13, -8, 6, 5, -2, -19, -9, -5, -9, -23, -14, -18, -15, 6, -20, -1, -9, 1, -18, -14, 5, -19, -8, -19, 5, -16, -6, -16, -3, -13, -12, -12, -9, -1, -22, -5, -2, -13, -8, 5, -13, -8, 5, 5, -14, 6, -21, 3, 0, -7, -10, -22, 0, -15, 5, -20, 0, -20, -5, -18, -15, -8, -18, -9, -9, -11, -16, 1, 1, -20, -13, -14, 7, -4, -6, 6, -21, -10, -19, -20, 3, 2, -8, -20, -13, 4, -10, -12, 4, -8, 1, 0, -1, -5, 2, -14, 4, -16, 12, -13, -7, -11, -6, 6, -14, 1, -6, 1, -3, 10, 15, -21, -17, -15, 2, 4, -11, -14, -15, -7, -14, -15, -7, -21, -5, -18, -3, -13, -6, -21, 1, -17, -11, -22, -19, -3, -5, -2, -8, -21, -13, -15, 1, -13, 4, -21, 2, -17, -11, -11, -5, -11, 5, -20, -16, 6, 2, 13, -20, -9, -1, -10, -17, -12, -2, -18, 2, -11, -19, 4, -18, -19, -11, -17, 6, 5, -13, 1, -17, -14, -11, -14, -21, -18, -10, -13, -4, -16, -6, -21, 1, -15, -8, -12, -11, -15, -9, -13, -20, 0, 2, -5, -10, 4, -5, -14, -18, -21, 1, -11, -18, 4, -1, 0, -14, 5, -15, -15, 6, -13, -5, -2, 3, -12, -16, -2, -4, -22, 2, -13, -2, -22, 0, -17, -20, -20, -15, 8, -5, -17, -3, -14, 3, -1, -17, -5, -16, -13, -18, -3, -22, -19, -19, 5, -12, -23, 1, 1, 4, -20, -5, 3, -19, -7, -6, -11, 8, -20, -15, -18, -7, -4, -2, 6, -3, -4, 0, 3, 2, 0, -13, -19, -10, 1, -2, 7, -18, -9, -15, 5, -7, -18, -17, -21, -2}
, {-17, 20, -124, -41, -28, 15, 7, 0, 101, -4, 0, -74, 46, -67, -265, -27, -3, 39, 13, -1, -3, 11, 66, -17, 15, -35, 16, 53, -2, -10, -151, 16, 101, -105, 68, 6, -44, 83, 15, -33, -24, -7, -58, 28, 31, -1, 13, 21, -28, -3, 77, -8, -48, 32, -168, -42, 52, -3, 21, -136, 20, 24, -4, 76, -27, -22, -57, -84, -82, -2, -78, -72, -115, -6, -3, -40, 21, 60, -93, 4, -20, -10, -9, -58, -28, 48, -6, -6, 2, -119, 8, -1, 24, -89, -73, -48, 79, -229, -44, 5, -52, 64, -19, 5, 43, -20, -64, -49, 23, 3, -15, 40, 9, -21, -26, -159, -146, 34, -10, -74, -61, 3, -8, -27, 16, 63, -12, 32, -8, 36, -151, -155, 0, -1, -39, -75, -80, 16, -17, -40, 0, -1, -72, -48, -23, -62, 2, -45, -42, -22, 35, -15, -9, -44, -6, -62, 63, -31, -9, 29, 38, 123, -111, -5, -35, 7, 15, -6, 47, -17, -131, -15, 17, -8, 37, 48, -71, 17, -85, 48, -7, 10, -20, 32, -26, 20, 4, -130, 23, -45, -14, -89, -2, 23, -107, -161, 156, -10, -27, -74, -43, 14, 0, 20, 0, 166, -69, 21, 24, -60, 4, -22, -61, 5, 35, -4, -20, -74, 20, -26, -7, -29, 54, 150, -2, -80, 53, -2, -92, -83, 10, 19, 2, 26, -156, -33, 34, 2, 33, 43, -78, 20, -3, -16, -138, 103, -17, -22, -53, 64, 10, -56, -62, -12, -10, -65, 16, 14, -165, -171, -76, 2, -59, -159, -76, -18, 7, 40, -10, 86, -86, -82, -99, -24, -5, -73, -50, 24, -6, -19, 2, -104, -19, -6, -40, -35, 17, 4, 46, 40, -2, -14, 70, 59, 8, 15, 37, -29, 115, 46, 36, -10, 41, 56, 117, 17, -68, 0, -202, 113, -45, 7, 8, -36, 0, -128, 18, 24, 5, 1}
, {-5, 25, -43, -61, 67, -10, -14, -105, 68, -7, 5, -20, -65, 6, -67, 11, -31, 45, 4, -66, -58, -11, -129, -6, 7, 70, 20, 0, -8, -61, 50, 41, -30, 72, -49, -90, 42, 6, 4, -106, 21, 6, -20, -10, -35, 9, 42, 6, 37, -4, -15, -24, -149, -41, 9, -14, 50, -47, -2, -4, 33, -20, 3, -73, 3, -22, -129, 76, 21, 17, -58, -79, -16, 7, 9, -22, -45, -61, -55, -4, 3, -14, 4, 28, 3, -27, 7, 2, 13, -83, 5, 14, 1, -142, 33, -39, -21, 22, 130, -17, -10, 105, 1, -21, 3, -4, 48, 123, 21, 0, 1, -29, 31, 5, -66, 25, 10, -41, 79, -39, 42, 8, 5, -94, 52, -33, -1, 86, 1, 54, -47, 92, -36, 19, 25, -73, -3, 6, -1, 93, -77, 33, 91, -55, 17, -3, 14, -87, -20, -11, 39, 3, 0, 86, -8, -14, 8, -85, 140, -42, 6, -28, -9, -11, 70, 18, 18, -15, 14, 0, 48, -47, 13, -7, -36, -50, 65, -12, -55, -42, -41, -15, -74, 31, -9, -17, 9, 20, 42, 35, 14, -104, -4, 17, -13, -20, -18, -6, -68, -5, -26, 12, 1, -18, 4, 4, 25, 5, 12, -27, -8, -54, -12, -36, 41, -7, 0, 20, 16, 1, -2, 53, -53, 44, 32, -12, 17, -13, 12, -43, 14, -17, -13, 25, 21, -21, 7, 12, -49, -25, 16, 16, -67, 57, -30, -86, -68, -26, -11, 22, 6, -70, 4, 0, 11, -50, -6, 32, 1, -136, 36, 0, 32, 15, -39, 11, 1, 98, -25, -15, -33, 22, 25, -113, 1, -27, 46, -82, 7, 11, 19, 101, 19, 12, 5, 76, 66, -41, 39, 6, -38, 38, -44, 22, 11, 30, -13, 28, -44, 4, -4, -1, 10, 39, 36, -16, 29, -69, -26, 3, 37, -69, -37, 16, 19, -62, 62, -103, 3, 58}
, {16, 26, 3, 15, -30, 5, 104, 45, -58, 1, 4, 89, -51, 69, -6, 44, 50, 31, -4, -87, 39, 128, 41, 21, 13, -38, 0, 88, 42, -50, 72, 15, 0, -199, 64, 25, 21, -44, 12, 12, 52, 0, -55, 46, 8, -4, -110, 99, -22, -5, 31, 40, -36, 57, -37, -6, -99, 43, -3, -27, 120, 37, 13, 90, 1, 10, -21, -9, 25, 20, -53, 51, -25, 8, -8, -20, -88, 41, 78, -112, 5, -89, -5, -186, 33, -50, 0, -1, 8, 17, -6, -50, -1, 13, 148, 21, 20, 52, 39, -135, 8, -95, -6, -43, 6, 1, -10, -43, -2, -2, 46, 17, -14, -4, 112, 36, -116, -91, 24, -59, -57, -31, 6, -126, 64, -68, 20, -60, 14, 11, 13, -4, 68, 8, 126, -50, 65, 14, 8, 93, -69, -25, -152, 38, 40, 16, 11, -130, -2, 23, 106, -3, 14, 83, 3, -15, -2, -46, 56, -58, 1, 9, 39, -53, 10, -44, 11, 5, -18, 20, -81, -131, -45, 2, 70, -1, 70, 6, -9, -4, -114, -138, -11, -21, 50, -1, 13, -103, -62, -79, 24, -85, 8, -11, 63, -18, 92, 14, 215, 72, 93, 4, 5, -22, -33, -85, -69, -80, 22, 21, 17, -129, 46, -114, -12, 14, 5, -86, 7, 31, 7, 24, -4, -25, -46, 0, 5, 26, -107, 15, 0, 16, -46, 5, -36, -24, -24, -14, -39, 15, -2, 4, -79, -74, -10, 42, 42, -18, -38, 10, 8, -118, 11, 41, 3, 9, 14, -5, -2, 87, 116, 4, 113, -52, 35, 17, 2, -14, -23, 45, 79, -17, 61, 6, 1, -120, 75, -35, -35, 19, 19, -43, 10, 22, 11, -60, -121, -50, -9, 64, 8, -84, 32, 33, 4, -49, 46, 1, 12, 53, -54, -2, 94, 29, 61, 2, 60, -29, -92, 55, 96, 16, 45, 22, 5, -112, 12, 8, 17, 52}
, {-12, -67, -19, -56, 1, -18, -10, -37, -29, -13, 4, -52, -77, 4, 29, -42, -17, -1, -6, -43, -4, -35, -29, -2, -11, -49, 4, 6, -15, -9, -14, 12, -18, -88, -33, -52, -2, -6, -22, -84, -11, -17, -74, 11, -63, 1, -69, -13, -14, -19, -15, -5, -3, -57, -37, -8, -11, -6, -22, -33, -18, 2, -16, -28, -8, -6, -22, -42, 2, -6, 1, -16, -30, -6, 0, -31, -22, -13, 8, -9, -4, -13, -3, -114, 33, 6, -34, -18, -21, -38, -20, 2, 5, -10, 3, 5, -4, -61, -13, -135, -9, -55, -14, -62, -28, -19, -64, -60, -42, -13, -6, -1, -22, -21, -19, -17, -31, -20, -63, 3, -28, -20, -6, -40, -5, -28, -19, -73, -7, -57, -9, 43, -15, -15, -20, -35, -25, -18, 5, -9, -32, -11, 73, -24, -10, -28, -9, -110, 17, -33, -32, -7, 2, -12, -16, 6, 6, -26, -21, -9, -17, -72, -11, -87, -3, -21, 0, -131, -23, -18, 14, -52, 17, -14, -27, -20, 3, -17, -17, -9, -31, -36, 42, -34, -4, -13, -22, 27, -12, -21, -3, -17, -13, -17, -50, -73, -64, -8, -12, -16, -22, -22, -2, -36, -74, -10, 64, -40, -8, 17, -2, 1, -49, -30, -17, -15, 0, -23, -8, -1, -17, -46, -4, -13, -13, 7, -26, -51, 5, -39, -4, -40, -12, 7, -91, -76, 31, -13, -34, 3, -15, 3, -37, -20, -36, -53, 14, -16, -5, 5, -14, -73, 3, -23, -21, -49, -8, -3, -23, -23, -34, -17, -22, -61, -47, 19, 5, -39, -51, -21, -59, -27, -13, -26, -4, -101, -41, -25, -43, -9, 0, -41, -6, 1, 1, 4, 0, -41, 1, -67, -15, -63, -17, -44, -4, -85, -41, 1, -16, -70, -23, -14, -46, -2, -13, 2, -17, -1, -42, -44, -56, -22, -31, 0, 3, -66, -6, 4, -18, -58}
, {8, -108, -3, 23, 1, -4, 67, 81, 33, 20, -7, -7, -49, -3, 26, 4, -1, -12, 9, 43, -155, -163, -89, 15, -7, -59, 6, -35, -1, 21, 8, -76, -46, -23, 3, 29, -20, 39, 14, -56, -36, -12, -42, -103, -20, 0, -35, 14, -10, 6, -66, -31, 80, 22, -85, 33, 60, 12, -6, 65, -35, 47, 2, -113, 16, -65, -49, 48, 137, 20, 53, -132, 21, 8, 4, -61, -46, -28, 89, -13, 20, 71, -8, 18, -49, -129, -33, 15, 4, 62, 14, 26, 15, 5, -86, 49, -64, -9, 96, -23, 13, 21, 5, -93, -20, -3, 16, -206, -80, -4, 52, -112, -14, 7, 128, -64, 7, -31, -20, 114, 45, -9, 7, 11, -26, -16, 1, -187, 2, -7, 27, -22, 97, 9, 82, 72, -21, 2, -1, -30, -13, 59, 63, -47, 16, -20, 19, -95, -54, -38, -17, 3, 17, -14, 19, -6, 24, -65, -10, 16, 11, 58, -31, -61, 21, 96, 1, -71, 46, 2, 33, -31, -20, 0, 19, -63, -32, 6, -50, -30, -79, 59, 123, -8, 60, -28, 7, -64, 25, 69, 20, -165, 7, 40, 12, -66, -5, -5, -63, 43, 56, 5, -6, -13, -8, 48, -11, -57, 1, -51, 16, -55, 9, -41, 85, 14, -4, 2, -7, -3, 5, 14, -17, 37, 46, -37, 78, -11, 21, 15, -5, 10, 12, 19, -97, -3, 33, -1, 61, 67, -12, 17, 63, 36, -16, -78, 38, 11, 113, -29, 3, -29, 45, 34, -6, -165, 15, 30, 29, 44, -4, 16, -27, -39, 26, 10, -4, 66, -102, -4, 68, 9, 21, 11, -3, 51, -35, -28, 22, 20, -5, 71, 9, 11, -2, -36, -27, -2, 0, -115, 39, -62, 20, 5, 2, 20, 37, 10, -64, -5, 6, 0, 2, -29, -21, 20, 19, 71, -53, -69, 78, 5, 13, -27, 1, 8, 32, -19, 7, -101}
, {3, -6, -18, -21, -21, -9, -18, -21, -8, -19, -17, -11, -22, -16, 1, -24, 2, -21, -10, -29, -5, 0, -20, -11, -2, -17, -6, -2, -21, 3, -14, -16, -10, -20, -13, -16, -16, 0, -11, -27, -16, -6, 3, -18, -13, -21, -19, -15, -7, -5, -21, -14, -11, -15, 3, -16, -12, -13, -1, 3, -15, -20, -20, -17, 2, 2, -15, 4, -18, -6, -10, -43, -17, -6, -16, -15, -20, -21, -14, 5, 1, -4, 6, -12, -5, -18, -17, -7, -10, 5, -14, -6, -4, -13, -2, 0, -17, -27, -1, -27, -15, -17, -2, -6, 6, 3, -15, -9, -9, -14, -21, -19, -14, -12, -21, -16, -1, -15, -31, -10, -21, -3, -17, -9, -19, -9, -11, -22, 6, 6, -3, -4, -15, -10, -5, 2, -22, -15, -5, 6, 0, 2, -3, -3, -21, -17, 2, -2, -21, 3, -4, -7, 5, 6, 4, -3, -19, -23, 0, -9, -16, -32, 4, -9, -13, -22, -1, 3, -5, 5, -13, -7, -10, 6, -1, -8, -16, 6, -3, -5, -17, -13, -34, -8, 2, 6, 4, 1, 1, -13, -8, -2, -14, -18, -1, -4, 3, -13, 0, -12, -3, -5, -3, -12, -2, -4, -14, 1, -6, -6, 4, -22, -3, -1, 5, 6, -23, -12, -8, -15, -6, -7, 2, 0, -22, -40, -16, -22, -19, -17, -2, -1, -2, -6, -12, 3, 4, -14, -20, 6, -5, -23, 5, -12, 2, -8, -16, -11, -15, 7, -6, -18, -22, -17, 0, -22, 4, -10, -14, -17, -19, 7, -7, 4, -3, -10, -12, 0, 5, -5, 3, -20, -18, -23, 0, -1, 3, -8, -16, 0, -22, 6, -19, -12, 3, -16, -15, 3, 4, -28, 3, -16, 5, -21, -10, -16, -17, 6, -8, -13, -12, -10, 1, 5, -2, 3, -5, -2, -3, 0, -3, -15, -18, 6, 7, -21, -11, 6, -9, -4}
, {7, -144, -101, -43, 7, 17, 72, -77, -93, -4, -6, 21, -7, -43, -88, -124, -15, -109, 6, -64, 17, -14, -34, 6, -3, -54, -11, -31, 15, -118, -27, -26, -34, 126, -70, -178, -32, 59, -3, -105, 20, 5, 102, 38, -180, -5, -13, -99, -10, -3, -160, 72, -154, 32, -64, -25, -89, 57, 9, 21, 132, -75, -2, -5, 11, -212, 17, 14, 35, -3, 72, 102, 44, 9, 3, -74, 8, 107, 55, -58, 6, 8, 7, -11, 14, -65, 84, 9, 7, 13, 6, -8, -9, 110, 101, -99, -33, 131, -151, -4, -69, 33, -1, -25, 5, -8, -91, 7, -38, -25, 34, 8, 18, 11, -101, 47, 61, -151, -50, 41, -17, -32, 12, 24, 17, 3, 11, -49, 14, -19, -76, 78, 91, 7, 24, -119, -60, -2, -2, 101, 48, -117, -19, -73, -10, -9, -4, -18, 14, 107, 75, -8, -9, -91, -3, -3, 1, 12, -120, -54, -17, -3, -157, -120, 33, 101, 20, -11, -33, 24, -40, 21, -94, 0, -78, 82, 17, -8, -119, 16, -75, -4, -112, 6, -40, 16, -5, -45, 29, -57, 3, 5, -4, -108, -85, 67, 37, 3, 72, -98, -17, 7, -14, -28, 44, 94, 59, -109, -2, -35, -1, -12, -4, 20, 85, -2, -2, 23, -3, 4, 5, 5, 69, -58, 0, 29, -51, -134, 148, 59, 8, -13, -13, 4, 31, 62, -80, 6, -77, -108, 50, -3, -136, 18, -39, 26, -52, -13, -7, 14, 7, 44, 11, -87, 3, 75, 14, -64, 11, 10, -58, 1, -45, 140, 46, 8, -24, -22, -49, -20, 29, -57, 10, -54, -3, 13, -13, -168, 144, 5, 10, -113, 15, 16, -3, 48, -44, 0, -46, -98, 44, -122, -131, -115, -7, -95, -30, 12, -33, -174, -116, 7, 35, 11, -33, 10, -173, -77, -30, -160, 23, -17, -1, -8, 2, 49, -1, -27, 10, -168}
, {9, 27, 3, -63, -2, 16, -62, 46, -1, 20, -13, -10, -3, 136, -102, -26, 1, 15, -1, -21, 49, 68, 23, 13, 6, -51, 18, -110, -19, -7, -14, 58, 26, 14, -101, 46, 59, 40, 7, -1, 25, 18, 27, -118, -41, 5, 115, 19, 0, -6, -149, -26, 65, -60, 37, -7, -82, 46, 2, -75, 47, 92, -4, -84, 8, -73, -44, 106, 119, -6, 61, -32, 14, 21, 16, -48, -23, -5, -48, -74, 6, 62, -4, 14, -48, -8, 2, 0, -6, -53, 19, -67, 9, -52, 62, -25, -46, 1, -31, -91, 22, 71, 14, -101, -43, -1, 133, -13, 11, 2, 76, -54, -2, 17, -85, 12, -97, 109, -38, -64, 59, -28, -19, -55, -65, 31, -2, -124, 4, 64, 34, 40, 25, 14, 122, 9, 9, -7, 19, 65, -70, 1, 75, 14, -32, -2, 16, 66, -57, 115, 16, 7, -6, -124, -1, -25, 0, 31, 59, -7, 0, 12, 96, 44, -53, -30, 1, 7, 49, -12, -19, 59, -13, -9, 59, -24, 4, 4, 20, 87, -11, 61, -127, 62, -87, -27, -9, 15, 8, 0, -18, -61, -22, 3, 15, -169, 16, 15, 82, 63, -88, -1, 7, 15, -43, -118, -50, 11, 52, 11, -6, -12, -14, -165, 27, -7, 12, 75, 5, 57, 6, 4, -97, 24, 12, -29, -46, 19, -23, -47, 15, -12, -28, 9, 100, -32, -61, -18, 41, -20, 83, 1, 51, -47, 8, -26, -74, -34, 5, -11, -13, 16, 22, -89, -19, -187, 26, -67, 62, -151, 15, 4, -23, -39, 21, -9, 13, -88, -122, -15, 21, 22, 11, -46, -11, -55, -55, -132, -71, -14, -4, 62, 12, 34, -24, 77, 1, 0, -8, 54, 7, 0, -77, -75, -3, -75, -38, -13, 148, 4, -57, 1, -38, -62, -61, 2, 64, -101, -116, -8, -57, -40, -180, -16, -4, 53, 35, 9, -21, -64}
, {-16, -2, -10, -16, 13, -4, -4, -21, 7, -5, -11, 15, -11, -5, -21, -29, 10, -3, -11, -36, -10, -15, -8, -11, -19, 6, -2, 18, -19, -11, 3, -3, -18, 11, -15, -28, -8, -18, 7, -32, 3, -22, -14, -4, -5, -1, 14, -5, -8, 2, -20, -25, -7, -4, -14, 13, -6, -6, -8, -20, -12, 7, -4, -19, -11, -20, -26, -1, 1, -5, 13, -23, -28, 2, -17, 4, -43, 3, -5, -37, -2, 0, -3, -7, -27, -14, -24, -5, 3, -21, -4, -20, -19, -26, 7, -2, -21, 10, -20, -30, -2, -8, -2, -49, -36, -13, -22, -23, -27, -10, -7, -24, -1, 2, -7, -23, -9, -16, -13, 4, -11, -9, 7, -17, -13, 6, -2, -27, -8, -1, -17, -29, -1, -10, 14, 4, -2, -6, -1, -18, -37, -1, -21, -1, -5, 5, -13, -6, -7, -43, -9, -20, 4, -8, -3, -10, -2, 3, 1, -5, -23, 5, 4, -30, -25, -35, -8, -27, -13, 4, -4, -22, -23, -12, -1, -21, -14, -17, -10, -3, 3, -35, -13, 0, 5, -18, -12, 7, -12, -16, -19, -31, -16, 3, 0, 0, 0, -18, 3, 1, -7, -9, -3, 3, -25, -15, -1, -11, -18, -4, -22, -11, -11, -18, -12, -11, -18, -8, -15, -12, -2, 1, -2, 0, -3, -18, -22, 0, 2, -21, -5, -26, -18, -22, -7, -16, -3, 5, -5, -23, 7, 4, -14, -22, -16, -25, 9, -12, -22, -22, -5, 13, -3, 3, -6, 6, 3, -6, -13, -8, 14, -13, -4, 2, -3, -15, 7, -15, -3, -9, 3, -12, 3, -8, 4, 17, -6, 5, -10, 6, 0, 5, -10, -9, -8, 4, 2, -5, 3, 8, -1, -2, -16, -38, -5, -35, -12, -13, -7, -10, -18, 3, -21, -14, -19, -9, -12, 0, -11, -16, -6, -14, -14, -2, -9, 2, -17, -2, 13, -10}
, {8, -37, -103, -121, 56, 0, -3, -39, 12, 1, 9, -8, 8, 31, -43, -66, 42, 63, 15, 9, -74, 40, -39, -2, -8, 45, 15, -17, 4, 3, -41, 61, 28, 97, 45, -69, 22, 33, 5, -84, 24, 12, -69, 25, 1, -9, 71, 1, -20, 19, 33, 3, -88, -25, 69, -25, 79, 1, 8, -75, -13, -57, 7, -5, 2, 51, -46, -112, -84, 2, -8, 52, 43, -2, -7, 37, -73, 55, -30, -1, -16, 81, 11, 19, -6, 62, 65, 6, -3, 10, 6, -34, -4, -33, -36, 33, -6, 18, -68, -102, 132, -31, 23, -44, 14, 0, 26, -53, -47, 2, -81, -81, 4, 5, -147, -49, -96, 108, 33, 41, 59, 27, 20, -46, 29, 32, 9, 45, 8, -1, 42, -30, -55, 18, -120, -15, 71, 5, 12, -21, -28, -38, 44, 105, 7, 50, 8, 6, 6, 84, -64, 17, 14, 22, -3, 7, 13, 48, 123, 3, 10, -74, 41, -10, -21, 74, 21, -79, 4, 16, -9, 51, 33, 1, -81, -73, 4, -6, -13, -62, -116, -57, -27, -16, 32, -16, 6, 100, -61, -9, -7, 79, -1, 31, 32, -101, -38, -3, -69, 71, 97, 0, -3, -79, 21, -74, 8, 8, 24, 22, 0, -6, 23, -48, 109, 16, 1, 58, -6, 5, -8, 32, -73, -51, -32, 1, -24, -58, 54, 68, 5, 4, -46, 20, 49, 18, -16, 4, -11, 45, -29, -7, 0, 144, -12, 49, -51, -18, -33, -4, 12, -46, 34, 67, 20, -12, 13, 2, -2, -7, -58, 9, 31, 54, -43, 12, 0, -66, 9, -10, -7, 74, 25, 38, 8, -33, -26, -44, 16, 10, 17, 25, 9, 17, -1, 33, -45, 68, 40, -37, -28, 63, -64, 44, 15, 29, 59, 15, -170, -134, 15, 7, -13, 32, -17, 1, 36, 27, -99, -39, 39, 32, -45, -1, 17, 36, 12, 13, 17, -83}
, {16, 85, -23, -80, -41, 13, 60, 44, 26, 6, -10, -90, -15, 66, -60, -11, -28, -54, 1, -38, 55, -29, -1, 1, -6, 31, 17, -19, 29, 13, 32, -8, -6, -86, 80, -56, -30, -18, 7, -21, -93, 14, -66, 14, -39, 8, -53, -31, 38, 13, 177, 2, 22, 24, -216, -26, -2, -18, 11, 10, -67, 9, 6, 8, 1, -307, -138, 22, 7, -3, -31, -97, 21, 8, -9, -19, -13, 99, 99, -7, 20, 44, 7, -174, -121, 46, 18, 14, 3, -32, 17, 15, -13, -53, -8, 94, 66, -72, 91, -215, 36, -18, 0, -52, -102, 13, -74, 39, -42, -24, 78, -14, 62, -5, -9, -30, -59, 72, -257, 58, 97, 1, -1, 29, 59, -23, 16, 88, 4, 39, -3, -64, 42, -4, 92, -74, 19, 18, -10, 48, -69, 83, 62, 58, -6, 18, 0, -20, 34, -4, 93, 9, 0, -3, -21, -38, -12, -113, 122, 49, -40, -53, 122, 37, 165, -10, 8, 1, -47, -10, -69, -10, -91, -20, 25, 19, 8, 13, -54, -18, 59, 49, -159, -40, 21, 52, -5, 9, 42, -142, 7, -16, -5, -245, -151, -90, 15, -6, 16, -67, 31, 15, -14, -23, 52, -13, 126, -239, -31, -59, 24, -125, -22, 39, -99, 19, -10, -156, 0, -16, 18, -30, -167, -72, -98, -58, -40, -130, 34, 42, 1, -17, -67, 7, 9, 22, -25, -7, -40, -36, 80, 11, 31, 9, -44, 46, -93, -74, 84, 19, 11, 84, 29, 9, 1, 17, 9, 118, -44, 56, -29, 12, -14, -38, 89, 11, -15, -138, 27, 66, 62, -28, -37, -59, -1, -110, -31, -6, 31, 1, -19, 62, 2, 28, 39, -55, -176, -12, -3, -19, 34, 16, -62, 31, 14, -6, -161, 0, 87, 48, -16, -13, -42, -23, 104, -6, 71, 26, -82, 60, 60, -134, 80, 15, 8, -22, 79, 33, 6, 13}
, {21, 61, -9, -92, 70, 3, 39, -64, 22, 21, -16, 64, -24, -25, -58, 2, 13, -49, -8, -26, -7, 9, -15, 11, 14, -57, 13, 16, -5, -60, -121, -44, 50, 46, 49, -24, 57, 25, 1, -31, -45, 4, -64, -125, -21, -10, 96, -2, -9, -1, 150, -20, -192, -105, -105, 83, -35, -47, 4, -147, -92, 65, 17, -219, 16, -6, 0, -18, 60, -8, 98, -34, -33, 3, 0, 9, -41, 114, 65, -81, 5, -62, -2, 11, -59, 8, -38, -5, 0, -48, 21, -40, 32, -22, -25, -31, -16, -28, 27, -35, 29, 52, 21, 41, -24, 15, 103, -63, -64, -6, 112, 51, 23, 5, -76, -24, -40, 27, 3, 68, -70, -9, 15, -139, -48, 41, 10, 64, 10, 21, -51, 61, -30, 15, 46, -188, -12, -3, -3, -3, -32, -62, 70, -87, 6, 5, 19, 46, 2, 1, 7, 19, 14, -20, -3, 3, 12, 24, 77, -66, -31, -109, -5, -32, -50, -23, -4, 11, 43, 2, 63, -205, 9, 17, 73, 56, -15, 3, 7, -27, -90, -150, 13, 65, 36, -7, -3, -20, -176, -92, -7, -137, 3, 6, -77, 12, -49, 0, 26, -57, -4, 1, -12, -29, 15, -20, -82, -12, -18, 9, 1, 29, -7, -10, 14, 12, -8, 1, 14, -7, 11, -24, -122, -11, 3, -60, -45, -3, 38, -30, 10, 38, -33, -11, 16, -32, 20, -8, 43, 37, -50, 0, -48, -136, -27, -133, 9, 71, -38, 27, -4, 23, -84, -37, 11, 58, 11, 4, -186, 44, -67, 2, -50, -115, 38, 8, 2, -8, 15, 90, 38, -68, 19, 62, 3, 53, -14, 19, 21, 0, 9, -30, 16, 26, 6, -57, -33, -23, -23, -41, -92, -35, 74, -37, -7, 14, 37, -35, -70, -107, 45, 6, 24, -2, -53, 19, -63, 18, -62, -58, 22, 77, 34, 13, 0, 30, 53, 4, 1, 0}
, {0, 64, -64, -11, -22, 5, 12, -50, 106, -2, 7, -27, -12, 1, 7, 7, 0, 3, -6, 33, 6, 18, 2, 2, -18, 24, -16, 83, -19, -156, -26, -66, -30, 24, 135, 30, -23, 36, -16, 48, -43, -13, -62, 36, 57, -20, 71, 98, 23, 4, -139, -91, -23, 114, 111, 30, -59, 26, -21, -51, -75, 9, 1, 19, -4, -111, -158, -204, 79, -15, -23, -13, 32, 3, -5, -34, -148, 1, 41, -22, -1, -129, 6, -90, -10, -105, -61, -10, -6, -85, -11, -17, -16, -4, 38, -83, 26, 91, -45, -118, -102, -88, -6, -35, -7, -11, 95, 10, -108, -6, 31, 42, -7, -21, 43, -66, -48, -116, -65, -123, -52, -32, -22, -258, 5, -14, 4, -134, -4, 26, -129, -183, 30, -6, -59, 32, 21, 5, -11, -27, -69, -6, 8, -9, -14, -50, -9, -114, -78, -204, 39, -7, -16, -15, 5, -22, 4, -6, -82, -102, 0, 44, -70, -117, -73, -62, -13, 7, -20, -3, 44, -58, -183, -19, 50, 41, -24, 4, 54, -53, -24, -56, -186, -126, 7, -13, 6, -65, 36, -41, -22, -16, -20, -182, -142, -62, 15, -14, -76, -73, -14, 6, -16, -73, -17, 29, -63, -18, 41, 11, 4, -32, -173, 125, -133, -3, -17, -30, 5, -16, -5, -67, -71, 45, -65, -6, 61, -219, 12, 38, -9, -102, -64, -12, 91, 32, -121, -8, -26, -105, -34, 7, 7, 46, 56, -1, -241, 41, 52, 3, 0, -8, -34, -36, -9, -19, -7, -174, -185, -39, -31, 3, 0, 31, -17, -2, -10, -59, 100, 13, 37, -101, -11, 71, -5, -59, -304, -186, -69, 2, 2, -191, -2, -19, -16, -73, 55, 102, -53, 63, -94, -90, -131, 52, -3, -6, -10, -14, 48, -121, -199, 7, 13, -43, -25, -11, -159, -41, 48, -7, -56, 11, 62, -19, -15, -9, -55, -37, -7, -11}
, {-2, -18, -23, -8, -14, -16, 10, -2, -23, -4, 16, -12, -9, -6, -11, -3, -17, -13, 6, 0, -15, -13, -3, -2, -10, -20, -12, -3, -5, -20, -14, -18, -18, 0, -19, 3, -3, -22, -9, -18, 1, -19, -22, 1, -15, 3, 1, -20, -22, -6, -7, -6, -8, -21, -20, -22, -12, 3, -18, 3, 4, -6, -6, -18, 3, -7, -20, -5, -1, 0, 3, 3, -9, -1, -13, 3, -9, -22, -3, -21, -4, -1, -8, 19, -15, 6, -15, -12, -21, 2, -9, 3, -5, -7, -23, -2, 0, -6, -12, -21, -18, -13, -21, -12, -22, -4, -1, -20, 5, -11, -7, -4, -2, -1, -6, 3, 4, 20, 18, 1, -15, -20, -22, -9, -14, -10, -13, -3, -8, -17, -12, -8, -11, -3, -3, 0, -19, 7, 3, 2, -4, -22, -5, 4, -14, 0, -8, -28, -15, -21, -7, -21, -22, -11, 5, -10, 5, 5, -15, 3, -14, -26, -4, -17, -15, 1, -9, -2, -1, -14, 3, 2, -11, -6, -10, -15, -5, -20, 13, 2, -19, 5, -10, -2, -15, -12, 3, -11, -18, 9, -10, -18, -8, -12, -24, 11, -16, -5, -8, -22, 3, 0, 17, -10, -8, -1, 0, -13, -4, -22, 3, -26, 9, -23, -1, 5, -13, -16, 6, 5, -20, 2, -13, -9, -2, -27, -12, -19, -7, -6, -12, -14, -20, 6, 4, -13, -18, -4, -34, -22, -2, -18, -19, -20, 5, -13, -4, -18, 4, -19, -6, 2, -5, -15, -12, 6, 6, 9, 8, -6, -22, -13, -11, -2, -16, -3, 2, 1, -18, -15, -20, -18, -3, -9, -15, -22, -18, -16, -1, -19, -12, 4, -20, -17, 2, 5, -11, 6, 2, -18, 20, -18, -11, -16, 3, -24, -7, 0, -8, -13, -12, 4, 4, -5, -3, -10, 3, -6, -14, -15, -4, -19, -19, -19, -10, -11, -22, -13, -23, -10}
, {2, -33, -23, -41, 76, 2, 0, -1, 100, 2, 22, 46, -15, -8, -4, -30, -3, 14, -9, 13, -24, -14, -6, -12, 5, 21, 6, -6, 6, 49, 2, -11, -10, -6, 177, 55, -154, 63, -4, -29, 6, 3, 99, -97, -45, -2, 107, -45, 22, 3, -153, 121, -9, -165, -40, 17, 54, -11, -4, -29, -65, 119, 23, -17, -3, -71, -34, -70, 25, -11, 31, -97, -3, -5, 0, -108, -3, -206, -3, -132, 14, -63, -6, 24, -33, -31, -174, 61, 21, -59, 8, -67, 60, 93, 67, 32, 73, -21, -10, -27, -53, -21, 19, -82, -40, -11, 80, 55, -44, 2, 9, 46, -107, -2, 32, 37, -66, -33, -194, -44, 7, -35, 17, 21, 194, 110, 1, 36, 0, -15, 30, -97, 44, 10, -145, -144, -54, 9, 21, -36, -42, -8, 68, -34, 23, 51, -11, 27, 34, -29, -26, 21, 2, -97, 1, 37, -54, 49, 65, 16, 44, 103, -65, 58, -77, 25, 17, -46, -25, -25, 14, -52, -55, 5, 72, 48, -17, -21, -70, 32, -45, -89, -61, 1, 77, -47, -9, 41, -39, 37, 0, -41, -7, 9, -7, -195, 52, 5, -35, -192, -26, -11, 26, 58, -53, 57, 9, 11, 30, 27, 9, 77, -38, -47, -50, 13, -5, 71, 12, -9, 16, 6, -162, 75, 45, -75, 92, -7, -27, -56, 15, 29, 75, 65, 23, -4, -23, -13, 73, 116, 70, -4, 47, 18, -56, 22, -49, 14, -17, -21, 3, -8, -5, 10, 26, 0, 1, -39, 6, -33, -84, -2, -13, -103, -9, 8, -7, 71, 57, 27, -127, -117, -65, 45, 19, 0, -82, 14, -10, 2, -4, -76, 6, -3, 1, 35, 142, 16, 40, 6, -67, -32, -69, -35, 2, -10, 47, -27, -16, 22, -4, 5, -46, 42, 171, 3, -89, 68, -87, 4, -34, 5, 1, 44, 1, -48, -89, -86, 22, 35}
, {0, 10, -33, 47, 7, 4, 48, -7, 10, -6, 2, 44, -17, -18, 11, 52, 26, -13, 17, 3, -4, 18, -30, 15, 3, 117, 12, 28, 46, 9, 36, -50, -27, -51, 5, -7, -22, 66, 7, -59, -8, 31, 58, -83, -11, 14, 10, 44, -3, -11, -63, -116, -56, -58, -58, -61, 6, 29, 20, 3, -77, -23, 16, -63, 3, -2, 27, -8, -51, 10, -72, -21, -67, -23, 2, -118, 1, -168, -48, 10, 26, -58, -3, 82, -1, -44, -45, -13, 5, 32, 4, 5, -14, 95, -74, 2, 14, -114, 66, 52, -14, -74, -30, -28, 17, -21, -127, -54, 37, 4, -45, -35, -47, 4, 14, 49, 8, -188, 15, -7, -48, -23, -7, -40, 33, -89, 3, -112, -5, 44, -5, 3, -52, 15, -29, 12, -29, 5, 18, -134, 28, -33, -20, -65, 8, -22, 19, 35, 10, -25, 25, 9, 27, -12, 2, 12, 8, 49, 23, -13, -11, -19, -45, -1, 10, -39, 17, -14, -50, 20, 59, 8, 20, 26, -15, -36, 89, 3, 27, 35, 51, -16, -115, -59, 4, 23, -2, -11, 8, 12, 28, 22, 18, 54, -15, 17, -18, 16, -48, -5, 3, -7, 21, -152, 34, 38, -62, 37, -49, 4, 16, 41, -18, 8, -20, 14, -1, 19, -7, 43, 37, -17, -102, 44, 45, -129, -33, 0, -51, -62, 16, -3, 42, 30, 26, -59, 63, 11, 0, -38, 36, 5, -84, -17, 0, 34, -21, -15, 20, 13, 16, 61, 26, -31, 9, -70, 25, 22, 54, -71, -30, 6, -44, -23, 25, 16, 5, -31, 28, 17, -183, -23, 39, -77, 14, -1, 24, 89, 48, -4, 8, 107, 11, -18, 9, 37, -8, -18, 39, -107, 24, 8, 105, -71, 2, 15, 1, 34, -13, 83, -55, 0, -45, 21, -2, 9, -74, 24, -6, -71, -51, 24, 21, 1, -4, -24, 49, -29, 14, 61}
, {-4, 5, 8, -35, -9, -16, -12, -39, -18, -16, -15, -13, -20, -15, -15, -16, -8, -11, -11, -24, -14, 1, 1, -3, 7, 5, -17, 1, -20, -14, -17, 9, -1, -23, -18, -14, 4, -4, -12, -23, 6, -1, 4, -4, 3, -14, 15, -3, -13, -22, -7, -7, -10, -14, -52, -15, -11, -7, 3, -33, -14, -18, -5, -3, -16, 1, 1, -14, -12, -7, 14, 25, 10, -14, -3, -16, -16, -2, -16, -7, -9, -20, -2, -16, 2, -1, -14, -3, 5, 2, 3, 4, 0, -19, -12, -8, -7, -9, -25, -5, -12, -15, -6, -15, -11, 5, -20, -19, -14, -11, -5, -15, -10, 0, -25, 10, -12, 9, -7, -3, 7, 4, -14, -15, 8, -17, 0, 4, -4, -9, 16, -42, -25, 5, 8, -27, 13, 5, 5, -24, -13, -7, 3, 3, 1, 1, -9, -13, -21, 14, -1, -6, 9, 2, 10, 13, 15, 11, 8, 13, 10, -38, -12, -20, -7, 0, 3, -11, 14, -12, -6, -14, -13, 10, -2, -1, -20, -4, 0, -7, -15, 13, -20, 2, 2, 4, 9, -19, 8, 2, 5, -15, -5, -44, 0, -27, -17, -13, 1, 0, -8, -15, 16, -2, 1, -2, -4, -11, 22, 2, 0, -30, -20, 4, -6, -10, 0, -8, 7, 6, -1, 0, -2, 7, -1, -7, -8, -1, 6, -15, -20, -11, 12, -16, 4, -27, -23, 12, -7, -2, 4, -10, -2, 13, -23, -5, -18, 9, -6, 13, -7, -6, -1, -3, 0, -16, 18, -9, -2, -1, -6, 7, -17, -12, 8, -14, 4, 0, -7, 20, 1, 1, 15, 1, -6, 13, -16, -15, 13, 15, 11, -2, 8, 7, 7, -3, -6, -14, -7, -14, -3, -2, -3, -11, -21, -10, -6, -11, -8, -14, 6, 18, -3, -12, 5, -2, 6, -6, -24, 16, 17, 8, 0, -3, -1, -13, 23, -15, 20, -22}
, {6, -41, -59, 11, -48, -23, -118, -17, 16, -4, -4, -32, -87, -18, -67, -26, -2, -49, -11, -9, 10, -32, -87, -9, 1, -4, -10, 7, 4, 7, -15, -29, -18, -38, 0, 0, -7, -27, -14, -96, -20, -10, -122, -59, -17, -4, -133, -34, -25, 6, 30, -34, -19, -84, -19, -17, -37, -16, -6, 9, -11, 0, 3, -59, -20, 48, -25, -162, 20, 4, 28, -79, 1, -22, 3, -26, -22, -23, -64, -21, -3, -54, 4, -29, -25, 5, -23, -9, -17, 32, -17, -14, -3, -38, 0, -26, -3, -17, -5, 35, -7, 4, -8, -81, -3, -2, -38, 61, -65, 2, -81, -8, -1, -5, 74, 8, -45, 28, -108, -25, -56, -17, -8, 30, 17, 17, -17, -7, -7, 45, -17, -44, 25, -1, 40, -6, 31, -21, 1, -6, 12, 2, -39, -19, -11, -69, -5, -11, 69, -35, 25, -7, -21, 21, -11, -21, -12, 41, -24, -15, -6, -49, 34, 22, 5, -1, -6, -66, 28, -6, 27, 35, 28, -20, -145, -8, -12, -15, 64, -2, -3, 21, 10, -7, -6, -10, -12, 127, 6, 87, -16, -51, -9, -43, 55, -63, 22, -1, -42, 84, -6, -19, -5, -59, -44, -6, -6, 38, -11, -15, 3, -19, -25, -47, -94, -4, 6, -37, 1, 7, -1, 9, 21, -7, 4, 2, -39, -1, 13, 7, 4, -29, 25, -4, -74, 19, -18, -10, -159, 1, -14, 5, -35, -6, -10, 14, 19, -26, -64, -10, -20, -25, -15, -2, 1, -28, -4, -60, -33, -55, 88, -4, 44, -34, -39, 6, -14, -56, 29, -2, 16, -42, -3, -58, -23, -19, -4, -16, -71, -13, 0, 21, 5, -5, -6, -20, -17, -40, -14, -18, -7, -59, 21, 34, 4, -82, -5, 4, 1, 19, -72, -21, -26, -4, -19, -9, 30, 23, -22, 51, 28, -14, -23, 4, -17, -16, 33, 4, -23, -19}
, {-5, 41, -46, 9, -85, -5, 77, -45, 15, -2, -5, -28, 23, 72, -15, -21, 43, 50, 3, -21, -87, -43, 30, 2, 2, -41, -4, -66, 12, -24, 20, 59, -20, 67, 34, -165, -13, 3, 26, -34, 36, 12, 43, -10, -1, 0, 24, -56, 17, -1, -83, 70, -156, 40, 53, -24, -13, 19, -10, -48, 42, -50, 14, 19, -4, -107, 8, -3, -93, 9, 107, -47, 29, -8, -16, -127, 27, 29, 1, -156, -45, 48, 21, -19, 13, -4, -66, 19, 8, 91, -12, -66, 1, 31, 25, 30, 58, 4, -197, -123, 34, -17, -3, -4, 63, -4, -31, -33, 20, -7, -45, -47, -23, 1, -173, -2, -100, -30, 22, -13, 34, -49, 8, 13, 97, 26, 1, -125, 19, -82, -64, -14, 37, 19, 78, -35, 1, 20, -23, -37, 48, 13, -28, -102, -35, 49, 4, -17, -30, 62, -8, -1, -7, -64, -19, -3, 17, -107, -2, -12, -14, -5, 56, -124, -17, 64, 20, -34, 3, 14, 90, -38, -2, -16, -1, -7, -23, -4, 43, 28, -25, 30, 75, -84, 16, 35, 4, 25, 54, 16, 2, -23, 11, -59, -64, 17, 142, 14, -52, -79, 45, 8, -18, 8, 40, 102, -62, -173, -22, 37, -5, -35, 10, 21, -2, 7, 9, -28, 6, 2, 15, -47, 109, 81, -26, 58, -51, -159, 62, 85, 1, -51, 30, 11, -20, 1, 10, 7, -10, -34, 15, 1, 79, 61, -44, -55, -154, 112, 27, 35, 10, 26, 84, -58, 3, -50, 10, -67, -118, -3, 54, 5, 32, -51, 57, 19, 1, 15, 33, 85, 138, -177, -20, 48, 19, -53, -31, -11, -149, 2, 1, -121, 19, 14, 0, -59, -23, 68, -13, -105, 26, -167, 7, -52, 3, -20, -29, 20, -126, -168, 8, -9, 57, -52, -35, 15, 2, -53, 14, 28, 3, -6, 23, 12, 18, 37, -53, 8, -3, -107}
, {-11, -1, -19, 6, -18, 14, 3, -7, -14, -3, 0, -9, -15, -19, 14, -21, -24, -14, 0, -7, -26, -6, -12, -9, -5, 3, -14, -18, 3, -21, -24, 1, -2, -18, -4, -15, -18, -1, 11, -43, -21, 6, 3, -7, -9, -9, -21, 0, 11, 20, -1, -1, -15, -30, -21, -1, -20, -18, -10, -21, -3, -3, -16, -24, -8, -4, 3, -27, 5, -3, 10, 5, -7, -3, -15, -11, -26, -8, -2, -16, 3, -7, -22, -32, -11, 8, 6, 0, -8, -3, 6, -22, 3, -19, -9, 3, -20, -7, -17, -22, -5, 5, -4, -9, -22, 1, -7, -9, -28, -19, -35, 0, -2, 5, 4, -24, 5, 0, -8, -16, -4, 3, 5, 2, -6, -22, -22, -6, 11, -11, -17, -18, -16, 13, -6, 21, -18, 8, -4, -10, -17, 4, 17, -5, -21, -10, 7, 3, -25, -2, -18, 18, 0, -18, -13, -6, 8, -8, 2, -9, -4, -33, -14, -2, 2, -6, 21, -11, -3, 0, -13, -5, -9, -20, -31, -15, 4, 7, -1, -20, 1, -29, -15, -16, 0, -18, 15, 18, 20, -16, 11, -6, 14, -9, -9, 0, -3, -3, -23, 0, -15, -1, 6, -12, -2, 5, -2, -20, 17, -8, -8, -14, -9, 10, -22, -4, 7, -23, -19, -11, -17, 1, 16, 4, -14, -26, -15, 1, 10, -10, 4, -13, -8, 5, -9, -11, -22, 0, -16, -19, 7, 12, -12, -24, -8, -3, -1, -7, -21, -8, -10, 12, -9, -23, 0, -19, 4, -12, -17, -3, -20, -2, -2, -11, -2, 3, 2, -3, -9, 5, 15, -13, -3, 13, 4, -30, 2, 17, 0, -22, 17, -3, -13, 5, -5, 5, 6, 4, -24, -23, -5, -3, -21, -1, 21, -10, -8, -7, -22, -6, -15, -11, -25, -9, 0, 13, 0, -24, -20, -2, -2, -13, -7, -14, -4, 2, 5, -1, -1, -17}
, {-4, -60, -27, 17, 37, -4, -72, 120, 31, -1, -14, 114, -63, -11, 44, -100, -20, -21, -3, 37, -193, -152, -11, -6, 6, -90, 1, 60, -22, -10, -70, -89, -105, 59, 37, -2, -77, -57, -21, -10, -43, -16, 14, -19, -36, 2, 104, -226, 33, 15, -88, -28, 65, 37, -13, -59, -75, 52, -10, -27, -31, -55, -25, -52, 2, -130, 84, -114, -52, -2, -157, -30, -55, -17, -3, -31, -89, -24, 13, 109, -15, -97, 0, 11, -248, 3, -191, 6, -4, -61, -13, -6, -16, -17, 80, -196, 18, 10, -44, -36, -57, -7, -7, 13, 63, -19, 113, -2, -74, -18, -13, 20, -105, -5, -59, 33, -29, 10, -47, 41, -141, -61, -20, -40, -4, -33, -12, -11, -18, -84, 33, 111, -66, -20, -303, -62, -109, -21, 0, 107, 80, -20, -67, -39, -17, -85, -4, 10, -289, 23, -93, -10, -21, -193, -5, -19, -6, -83, -21, -136, -101, 56, -67, -20, -84, 26, 0, -19, -91, -22, 13, 54, -73, -4, -69, -17, -6, -2, -161, -38, -9, -10, -74, 23, -129, 20, -16, 27, -70, -84, -15, 138, 3, -111, -108, 10, 57, -8, -157, -41, 92, 3, -18, 36, 123, 18, -44, -19, -3, 79, -23, -31, -261, -79, -96, 3, -17, 77, -21, -20, -15, -26, 21, -24, 10, 66, -24, -47, -21, -86, -12, -57, -112, -4, 94, -73, -67, 0, 56, -10, -98, -21, -177, -31, -27, 68, 62, -73, 108, -13, 6, 42, -2, -33, 6, -19, -22, -56, 40, 48, 67, -17, 40, -69, -93, 0, -16, -104, -72, -42, -43, -21, -7, -24, 0, 45, -251, -66, -30, 0, -6, 49, -22, -11, -7, 109, 61, -66, -71, 30, -8, 63, -89, -120, -4, -54, 32, -14, -100, -166, 76, -21, -9, -112, -12, 21, -247, -45, 36, -248, 18, 15, -61, 4, -16, 13, -89, -18, -10, -131}
, {-19, -25, -37, -17, -14, -3, -3, -15, -6, -5, 0, -11, -9, -4, -13, -18, -12, 4, -10, -14, -23, -6, -18, 3, 5, -10, -12, -4, -8, -16, -23, -20, -15, -9, -7, -13, -16, 1, 5, -9, -13, 6, -23, 0, -24, -11, -20, -10, -21, -12, -1, -5, 1, -12, -7, -9, 6, -13, -4, 4, -18, 5, -12, -4, -13, -14, -13, -19, -12, -6, 15, -12, -8, 1, -5, 4, -16, -21, 6, -7, -19, 2, 6, 3, -5, -7, 6, -7, -3, -7, 5, -3, -3, -7, -10, -8, -9, -3, 3, -13, 2, -22, -3, -23, 4, -6, 11, -5, -10, -1, 5, -10, 5, -9, -5, 0, -17, -14, -1, -11, 1, -10, -3, 5, -16, -18, 6, -7, -19, -12, 5, -13, -11, -4, 2, 3, -22, -13, -21, 3, -9, -20, -22, -21, -20, -8, -21, -32, -19, -12, -8, -22, -4, -22, 0, 2, -20, -4, -2, -14, -12, -21, -10, -9, -14, -21, -10, -23, -9, 4, -4, -15, -2, -15, 0, 6, -6, -14, -5, -15, -7, -5, -5, -8, -4, -5, -5, -13, -2, -5, -12, -22, -14, -18, -1, 4, 1, -13, -12, -21, -13, -22, -19, -2, 2, -11, 15, -8, 0, -17, -19, -21, 2, -21, -9, -15, -13, -10, -10, -13, 2, -3, -13, 4, -23, -4, -6, -6, 5, -10, -16, -7, 5, -12, -2, -22, -20, 5, 3, 3, -1, -1, -14, -5, -10, 0, -3, -20, -18, -5, -8, 4, -7, -8, 5, -21, 2, -10, -12, -6, -4, 4, -12, 13, -22, -22, 7, -15, -29, -10, -12, -23, 4, -20, -10, 3, -7, 7, 4, 1, -18, -10, 2, 1, -19, -2, -7, -7, -10, -29, -6, -5, -5, -13, -1, -9, -8, 2, 4, -15, -20, -5, -20, 2, -16, 1, -20, -13, 5, -17, 14, 0, -16, -3, -10, 10, -13, -11, 5, -4}
, {1, 104, 49, -14, -102, 13, 67, -108, -34, 21, 0, 21, -33, 51, -81, -49, 8, 39, -3, -93, -14, -16, -7, 3, 10, -30, 7, 20, 8, -112, 18, -1, -10, -115, -47, -29, 9, 33, 9, -20, 26, 5, -6, -71, -55, 19, 73, 51, 1, -5, 52, 34, -142, -51, 9, 11, 37, 9, -5, -63, 46, 106, 5, -54, 6, 29, -22, 3, 32, 10, 21, -116, -9, -1, -8, 31, -23, 40, 34, -81, 3, -51, 21, 15, -23, -10, 15, 15, 4, 15, 16, 14, 24, -12, 13, -10, 23, 94, 4, -95, 18, 34, 15, -12, 33, -8, 34, -56, -17, 1, 76, 69, 9, -7, 89, 30, -50, -33, 47, 109, -59, 8, -7, -113, -11, -39, 11, -5, 7, 41, -45, 45, -21, -1, -10, -178, 15, -2, 10, -37, 25, 32, -17, -99, 1, -53, 20, 5, -31, 33, 39, 17, 16, 4, 11, 14, -6, -72, -18, -43, -51, -22, 67, -68, 15, -13, 7, 12, 23, -11, 55, -109, -15, 3, 21, 25, 6, 11, 9, 4, -91, -222, 28, 23, 1, 24, 5, 14, 22, -15, -1, 25, 2, 59, -111, -2, -19, 10, 41, -117, 47, 21, 5, 9, -37, 37, 4, -56, 17, 21, 10, -24, 41, 40, -13, 1, 14, -2, -4, -1, 0, -14, 68, -115, 18, -29, -10, -21, 58, 60, -1, -22, -30, -2, 26, 43, -24, 14, -19, 80, 1, 21, 115, -40, -27, 79, 60, 57, -15, 15, -5, -36, 22, -12, 2, -35, 18, 29, -75, 67, 93, 21, 20, -90, 75, 7, 17, 44, -20, 12, 138, -58, 12, -23, -7, 4, 1, -19, 7, 1, 8, 43, 4, 12, 21, -34, -64, -36, -48, -178, 38, -41, -27, -33, 5, -4, 45, -2, -80, -173, -33, 17, 97, -31, 46, 10, 55, -30, -63, 137, 77, 34, 12, 18, 12, -12, -10, 16, 6, -31}
, {-17, -36, -42, 16, 47, 20, -10, -118, 66, -4, 19, 77, 7, 31, -11, -122, 2, 57, 5, -125, -8, -16, -7, 6, -1, 24, -5, -53, 16, -159, 20, 62, 41, 76, -63, -89, 66, -54, 14, -88, 17, 19, 73, 59, -55, -2, 73, -30, 6, -5, -24, -16, -161, 35, -19, 24, 34, -48, -2, -31, 64, -67, -11, -5, 19, 43, -9, -40, -10, 4, 19, -10, -28, 3, 11, 15, -8, -18, -76, 22, -5, -83, 12, 77, 14, -34, 43, 3, 5, -3, -5, 67, -2, -11, 94, -62, 16, 41, -11, -9, 55, 5, 3, -21, 16, 8, 89, -127, -29, 6, -47, -67, -11, -3, -57, -71, 87, -25, 16, -53, 68, -24, 18, -99, -14, 28, -7, -196, 14, 30, 70, -14, -7, 14, 28, -75, 2, 14, 11, -47, -35, 15, 34, 35, 26, 45, 14, -12, 85, 14, 24, -1, -2, -37, -3, 26, -8, 51, 43, -4, 30, 34, -39, -9, 0, 35, 7, 2, -30, 34, 6, 16, -46, 1, 23, -11, 7, -18, -37, 50, -41, 80, -70, -60, 23, -53, 2, -55, 16, -6, 8, 11, -21, 38, -44, -1, -63, 4, -91, -28, -4, 2, 10, 16, -7, 11, -288, -21, 6, 15, 5, -77, 36, -44, -3, 11, 14, -48, 1, 1, 12, 17, 20, 51, 22, -4, -72, -32, -82, 54, -7, 8, -75, 16, 45, -68, 20, 5, 3, -58, 58, -19, 34, 47, -18, -46, -70, 1, -2, -16, -3, 0, 33, -36, 18, -67, -1, 42, -20, -5, 16, 21, -10, -1, -3, -2, -11, 17, 75, -36, -20, 23, 10, 17, 16, -87, 12, -26, -14, 12, 6, -27, 0, 13, 13, 26, 62, 17, 11, -31, -10, 34, -65, -17, -11, 8, -12, 12, -46, -135, -21, 14, -5, -42, -22, 1, 142, 40, -73, 12, 0, 73, -72, -41, -4, 45, 81, 125, 20, -89}
, {20, 15, -16, -25, -5, -4, -4, -13, 15, -7, 13, 0, -27, 20, -14, -6, -10, -17, 12, -36, 6, -1, -19, 6, 2, -4, 7, 7, 9, -6, -9, 7, 1, -18, -7, -19, 2, 5, -9, -27, 2, 9, -19, -11, -39, 11, -15, 1, -8, 2, 20, 3, 1, -1, -24, -8, 4, 1, 3, 6, 3, -5, 20, 1, 8, -16, -20, -18, 4, 10, 5, -12, -18, -14, -4, -5, -10, -14, 21, -6, -11, -20, 12, -32, 5, 4, -26, -10, -1, -18, 21, -10, -6, 5, -11, -23, 7, -6, -10, -28, 2, 5, -9, -22, -7, -8, -18, -12, -18, 3, -26, -10, 15, -20, -11, 7, -19, 6, -31, -10, 0, -8, 7, -30, -9, -8, -6, -9, 3, -18, -4, -29, 4, 8, 7, -32, 1, -5, -8, -24, 5, -12, -14, -11, -19, 0, -6, -35, -13, 4, -16, 3, -2, -13, 1, -6, -3, -9, 1, -16, 4, -24, 2, -14, -14, -12, -1, -12, -8, -5, -13, -26, -23, 1, -17, -6, -5, -9, 14, -22, -14, -15, -5, -12, -3, 8, 8, -20, -11, -1, 19, 0, 3, -10, -2, 10, -10, -26, -9, -53, -15, -10, -23, -8, -2, -1, 10, 3, -7, -2, 1, 2, -19, -10, -10, 4, 1, -1, 11, -12, -9, -19, 9, -1, -5, -30, 16, -15, 2, 6, 5, -20, 0, -9, -22, -37, -15, 1, -6, -16, -12, 6, 14, -1, 3, -19, 22, -10, -18, -16, -10, 30, -14, -15, -7, -25, -9, -28, 6, -12, -10, -21, 7, -53, -18, -19, -21, -18, -24, 4, 14, -22, -1, -8, -19, -24, 1, -24, -18, -18, -15, -20, 15, -20, -18, -9, -9, -1, -17, -26, -20, -19, -3, -19, 5, -23, -10, 4, 1, -28, -26, -9, -8, 0, 1, -13, -2, -8, -1, -4, -7, -13, -23, -21, -25, -41, 1, -6, -17, -13}
, {-12, -67, -22, -10, -32, 1, -8, -42, -51, -21, 6, -39, -44, -22, -45, -24, -2, -36, -9, -26, -13, -18, -43, 0, -19, -27, -14, -18, -19, -30, -31, 24, -26, -39, -5, -47, -18, 2, -18, -34, -44, -7, -24, -15, -21, 4, -19, -33, -5, -6, 23, -16, -2, 31, 39, -56, -37, 0, 4, -55, 2, -16, -3, -30, -12, -25, -9, -11, -19, -8, 4, 4, -51, 4, -13, -31, -66, -3, -53, -18, -9, -58, -13, -31, -40, -14, -29, -14, -8, -1, -1, -17, 2, -27, -15, -33, -17, -37, -4, -37, -17, -5, -9, -46, -51, 0, 0, -36, -33, -10, -46, -10, -3, -12, 5, -21, -45, -7, -36, -31, -49, -9, 1, -49, -5, -19, 3, 1, -21, -21, 18, 0, -12, -11, 1, -14, -59, -4, -7, -13, 34, 0, -39, 27, -14, -36, -22, -44, -14, -6, -26, -11, -13, -37, -21, -25, -19, -22, 16, -6, -8, 8, -66, -49, -6, -19, 1, 9, -9, -22, -6, -27, -11, -19, -30, -7, 1, -14, -40, -12, -44, -27, -74, -19, -30, -13, -8, 2, -22, -4, -10, -7, 6, -29, -41, 9, -7, -9, -3, -1, -42, -21, 14, -2, 20, -26, -56, -23, 7, -53, -10, -33, 19, -10, -21, -12, -17, -48, -11, 3, -21, -28, -20, -62, -13, -25, -47, -56, -24, -18, -2, -19, -11, -2, -53, -29, -26, -2, -24, -6, 0, -5, -1, -17, -12, -57, -19, -44, -16, 2, -20, 0, 2, -8, -4, -10, 6, 8, -50, 4, -19, -18, 0, 33, -29, 1, 12, 29, 1, -10, -18, -24, -10, -23, -17, -52, -7, -6, -14, -2, -10, -23, 3, -15, -9, -38, -37, 7, -8, -32, -42, -31, -20, 5, -14, 9, 9, 0, -19, -10, -1, -19, -33, -32, -23, -12, 16, 1, -42, -8, -13, -44, -24, -8, 3, -61, -24, -5, -12, -12}
, {-6, -13, -1, -36, -11, -18, -16, -13, -65, -18, 0, -33, -34, -11, 7, -55, -6, -23, -4, -56, -39, -2, -8, 1, -4, -31, -3, -12, -18, -35, 4, 4, -2, 3, -15, -40, -28, -4, -1, -30, -10, -13, 7, -22, -33, -12, -14, -11, 12, -11, -17, -15, -36, -12, -25, -18, -1, -10, 3, -47, -18, -2, 5, -10, 14, -27, -55, -72, 18, -3, -10, -22, -36, 5, -7, -59, 28, -13, 2, -49, -20, -31, 0, -43, -55, -7, -31, -18, -19, -53, -14, 0, -6, 18, -22, 2, -3, -40, -15, -51, -31, -15, 7, 3, -34, 4, -37, -16, 4, -11, -38, -47, 5, -6, -35, -12, 15, -35, -6, -17, -40, 0, -18, -36, -19, -11, 15, -8, 18, -45, -9, -30, 2, 5, -10, -28, -38, -8, -4, -32, -54, -3, -2, -18, -20, -25, -13, -39, -6, -10, -52, 1, -16, -9, -5, -20, 1, -7, -21, -18, -13, -35, -28, -49, -5, -17, -4, -31, -5, -1, -35, -18, -12, -15, -53, -17, -4, 3, -21, -7, -36, -23, -18, 0, -5, 11, -20, -38, -11, -12, 11, -20, -6, -48, -6, -46, 4, -5, 14, -46, -14, 9, 10, -43, -52, -4, 2, -21, -24, -2, -13, -16, 4, -2, -21, -1, -13, -10, -11, 5, -17, -9, -22, -15, -11, -29, -26, -21, 0, 1, -6, -14, -14, -19, -37, 12, 2, -12, -42, -3, -4, -22, -12, -6, -45, -17, -32, -15, 2, -22, -12, -52, -17, -16, -9, -25, -23, -13, -6, -43, 22, -12, 9, -20, -62, -25, -1, -47, 0, -1, -31, -2, -13, -15, -24, -39, -26, -6, -50, -18, -13, -35, 6, -16, -11, -25, -12, 1, -12, -46, -23, -19, -12, -5, 1, -26, -5, -14, -40, -1, -3, 0, -52, -21, -20, 2, -37, -17, -32, -23, -27, -8, -29, -11, 2, -13, -14, -1, 5, -23}
, {-4, -1, -71, -27, -78, -4, -201, 5, 29, 4, 16, 19, 64, -5, -228, -42, -20, -5, 2, -127, 7, -18, -3, 2, 9, 22, 3, 11, -4, -48, -7, 114, 51, -95, -41, -33, 1, -49, -25, -54, 0, 22, -31, -88, -17, -6, -3, 8, 3, -15, -56, -24, -166, 18, -56, -9, 22, -41, -3, 11, 54, -75, -4, -54, 2, 28, -43, 2, -97, -9, -45, -43, 0, -3, 16, 27, 32, -119, -115, -36, 27, -17, 0, -10, 49, -111, 31, -2, 8, 24, 16, 13, -10, -30, -78, -46, 49, -119, -11, -9, -39, 21, -1, -7, 13, -11, -82, -21, 5, 15, -84, -1, 71, -13, 63, -28, 38, 19, -4, -12, 34, 42, -7, -87, 19, 46, -7, -124, 9, -35, 22, 9, 1, -2, -29, 54, 5, 5, -3, -160, 8, -4, -181, 114, -28, 32, 19, -140, 33, 11, 7, 11, -8, -25, -6, -4, 24, -17, 109, 59, -19, -70, 17, -2, 15, -42, -6, 19, -27, -15, -41, 8, -5, 13, -105, -14, -8, -13, -1, 3, -54, 67, -25, 56, -19, -2, 6, -48, 92, 82, -7, -10, 0, 10, 36, -39, 106, -2, -85, 1, 37, -11, 9, -92, 30, 2, 119, -30, 32, -21, 16, -17, 57, 36, -10, 1, 18, 42, 10, 20, 8, 91, 101, 74, 48, 93, -12, -31, -4, 99, -10, 13, 31, 8, -54, -100, -6, 2, -34, -13, 53, -2, -31, 57, -75, 162, 52, 83, 59, 35, 10, -143, 24, 126, -2, -17, 15, -16, 26, -74, -51, 7, -96, 54, 12, 1, -9, 41, 53, -55, 31, 74, 1, -89, 19, -26, -40, 8, -20, 8, -3, 19, 22, 13, 10, 58, -106, -132, 18, -211, -28, 85, -27, 18, 1, 24, 17, 19, 14, -105, -9, 8, -121, -15, 94, -10, 48, 83, 4, -80, -37, -34, -16, 26, 4, -91, -6, 57, 22, -130}
, {-15, -8, -10, -15, -2, -22, -5, 0, 2, -3, -8, -5, -8, -23, 6, -8, 0, 2, 1, -19, -10, -10, -11, -19, -18, -3, -16, 3, -17, -1, -10, -19, -21, 0, -23, -10, -9, -9, -11, -38, 1, -18, -15, -20, 1, -21, 10, 1, -13, -3, -5, -3, -20, -19, -17, -18, 1, 2, -14, -20, -19, -3, -7, -21, 6, -5, -4, -23, -24, -19, -9, 17, -10, -13, -3, -8, -11, 2, -12, -14, -8, -1, 6, -1, -18, -18, -14, -9, -2, -9, -19, -4, -12, -22, -6, -20, -10, -20, 1, 9, -17, -15, -2, 0, -22, -4, -27, -16, -5, 4, -30, 3, -12, -5, -1, 4, -19, -37, -16, -9, -16, -20, -7, -10, -3, -10, -4, 0, -17, -22, -20, -23, -33, -14, 12, 5, -8, -1, -15, -15, -12, -11, -7, -9, 2, -13, -13, -22, -1, -20, -11, -20, 4, 1, 3, 4, -20, -7, -10, -10, -22, 0, -10, 2, -15, 2, 0, -6, -19, -5, -9, -1, 1, -12, -31, -17, -18, -14, 1, 0, 7, -17, 2, -7, -5, -15, 3, -21, 2, -5, 5, 7, 0, -11, -8, -40, -7, -19, 12, -13, -18, -19, 2, -20, -4, 0, 3, -7, -8, -5, -11, -38, 7, -19, -7, 4, -8, 6, -1, -12, -4, -7, 6, -17, 5, -15, -8, -37, 2, -4, -4, -7, -23, 1, -13, -16, -4, -4, -15, -22, 4, -3, 1, -14, 6, 1, -12, 1, 4, 3, -4, -6, -13, -1, 5, 2, 0, -3, -16, -27, -12, -2, 6, 20, -1, -4, 5, -13, -13, -2, -10, 1, -5, -1, -3, -37, -18, -10, -9, -9, 5, 4, -16, -20, -18, -21, -13, 3, 6, -22, -11, -2, 3, -2, -12, 3, 1, 6, -2, -16, -12, 0, -8, -6, -21, 5, 6, -21, -17, -7, -19, -18, -20, -20, -18, -12, 0, 6, -13, -6}
, {-33, 7, 39, -71, -28, 5, 70, -23, 3, 11, -24, -163, -119, 34, -95, 46, 10, -182, -8, -13, 14, -51, -61, -3, -23, 44, -11, 64, -3, -88, 127, 36, -27, -53, -49, 11, 154, -7, -10, 21, 53, 20, 7, -90, 14, -13, -41, 65, 101, 4, -228, -152, -14, 3, 60, 29, -73, -5, -11, 23, -140, 127, -9, -32, -36, -45, -31, -226, -71, 2, -49, -4, 15, -6, -21, 118, 14, 90, 139, 71, 10, -88, -3, 11, 37, 9, -21, -15, -23, -2, 2, -29, -33, -20, -1, -55, -40, -113, 2, -3, -83, 137, -15, 2, -43, 14, -132, 35, -25, -17, -33, 86, -76, 21, -102, 66, -1, 47, 35, -10, -50, -91, -2, -276, -74, 25, 3, 47, -6, -7, -39, -146, 177, -7, 145, 68, -33, 6, 13, -20, -37, 75, 11, 104, 10, -112, -9, 41, -28, 44, -34, -13, -13, -27, -15, 4, 21, 9, -24, -53, 62, 22, 11, -24, 45, 39, -13, -23, 18, -5, 127, 32, 62, 0, -12, 70, -41, 0, -95, 81, 6, 9, 45, -147, -10, 40, 13, 255, 49, 41, -22, -22, -10, -150, 25, -9, -14, 6, 38, 49, -11, -4, -6, -88, -56, 123, -73, 67, -43, 48, -5, -42, -64, -51, -165, -9, -10, -142, -18, -60, 7, 107, -29, -127, -17, -25, -82, -142, -243, -107, -9, -100, 69, -11, 54, -28, -67, 1, 75, 98, 118, -1, 83, -15, -164, 84, 50, -99, -5, -2, -17, 101, 23, 199, -21, -80, -21, -102, -190, -144, 3, -18, 107, -68, 53, 3, -6, -162, 5, -107, 25, -146, -18, 78, -12, -134, -128, -57, -48, -4, 6, -144, -22, -11, 1, 51, -23, -36, -8, 94, 97, -134, 19, 71, -4, -133, -245, 20, 44, 1, -208, -1, 81, 44, -144, 20, 145, -35, -175, -97, -70, 14, -8, -19, 5, 81, -46, -1, -35, -8}
, {11, -4, -56, 18, -30, 20, 38, 19, -42, 17, -13, 63, 38, -23, -126, 8, 39, 34, 5, -59, 53, 39, 88, 14, -1, 17, 17, -29, -24, -8, 60, 37, 52, 36, 63, -28, 55, 6, 1, 22, 107, -9, -39, -94, 42, 12, -133, -15, -48, 20, 85, 53, 74, -70, 24, 4, -12, 30, 21, 56, -16, -82, 0, -85, -16, -96, 19, -32, -12, 25, -25, 47, -20, 17, 9, 35, 11, -29, -16, 11, 29, -21, 22, -83, -69, -19, -90, 15, -12, -65, 4, -107, -37, -50, -17, 69, -34, -54, 45, -37, -73, -4, -32, -72, -46, 45, 111, 3, -77, -10, 17, -44, 74, 4, 25, -14, -41, 27, -15, 48, 30, -2, 7, 15, -55, -17, 41, 45, -6, -71, -48, -52, -134, -14, 16, -30, 55, 29, 8, -134, -71, 65, -62, -107, -2, 33, 21, -85, -32, -45, -82, 12, -12, -25, 9, -33, -41, -8, -15, 20, -70, -24, 9, -135, 14, -12, 37, -135, -59, 29, 24, -49, -225, 1, 142, -77, -123, 12, 26, -147, -100, -3, -161, 68, 33, -6, -17, -47, -67, -139, 6, 62, -16, 71, 30, -18, 10, 14, 58, -34, 54, 6, 16, 89, -36, -50, 95, -6, 27, -66, 13, 15, 23, -23, 63, -7, 3, 68, 5, 53, -11, -45, -20, -83, 79, 73, -28, 22, -71, 17, 34, 4, 11, 8, 41, -53, -5, -15, -49, -79, -11, -2, 37, -53, -25, -3, -57, -16, -51, 68, 11, -6, 71, 23, -9, -21, 19, 25, 24, 33, 42, 9, 17, -13, -35, 16, 6, 6, 22, -42, 111, 76, 4, 68, -9, -54, -25, -15, 40, 5, 2, 39, -11, 111, -17, -3, -57, 34, -42, 16, 99, 29, 69, 26, 15, 33, 34, 57, -26, 69, -63, -13, -4, 18, 32, 5, 38, 59, 6, 65, 27, -3, 22, -35, 5, 30, 45, 69, -10, -16}
, {5, 7, -7, -3, 0, -1, -10, -11, 1, -8, -13, -7, 0, -10, -13, 1, -4, -10, -5, -15, -20, -1, 6, 0, 4, -15, -16, -20, 2, -11, 1, -22, -22, -2, -8, -4, -4, -21, -10, -12, -22, -5, 0, -18, -12, -2, 4, -19, -3, -6, -6, -6, -18, 1, -18, -13, -11, -19, -3, 1, 3, -7, -8, 6, 2, -14, -13, 1, -2, -8, -19, -4, -22, -19, -8, 3, -19, 3, -12, -16, -2, -18, -17, -14, -2, -14, -2, -12, -15, -11, -11, 5, -8, 4, -23, -15, -8, -4, -11, -7, -4, 5, -3, -13, -22, -1, -14, -11, -5, 0, 5, -19, 7, -13, -17, 2, 6, -11, 1, -2, -13, -7, -5, -9, -22, -2, -22, -15, -11, -21, 6, -3, 6, 6, 10, -12, -1, 2, -1, -6, -6, 4, -5, -7, 6, -6, -1, -17, 0, 3, 1, -20, 5, -14, -19, -2, 4, 3, -20, -19, -5, -5, 1, 3, -10, -2, 0, -12, -1, -16, -18, -11, -2, -8, -12, -15, 0, -17, -22, -2, 2, -2, -6, -12, 5, -3, 0, -14, -14, 0, -12, -17, -13, 0, 1, -2, 6, -8, 0, -12, -11, 5, -20, -4, -21, -21, -9, -1, -8, -16, 3, -8, -7, -5, -15, -12, 0, -5, -17, -22, -15, -9, -16, -11, 4, 3, -9, -11, -18, 4, -21, -2, -21, -12, 5, -6, -15, -14, -16, 2, 6, -10, -7, -4, -13, -17, -1, -1, -20, -7, 0, -18, -10, 0, -13, -16, -7, -1, -18, -6, 5, -16, -5, 16, 3, -17, 1, -18, -20, -17, -1, -1, -20, -20, -4, 6, -8, -1, -13, -10, 5, -11, -21, 0, 1, -17, 1, -18, 0, -14, -22, 4, -16, -23, -15, 4, -17, -7, -15, -19, -13, -3, -3, -7, -2, -9, -13, -4, 3, -8, -2, -8, -8, -1, -13, -21, -13, -10, 4, -4}
, {-23, -9, -1, -20, 0, -15, 6, 11, -9, -14, -11, -14, -16, -4, -10, -19, -13, 5, -5, -13, -16, -1, -20, -16, -2, -4, 4, -22, -12, -6, -16, -6, -8, -4, -11, 10, -22, -21, -16, -20, 4, 3, -3, 1, -7, -20, -13, -6, 1, 4, 2, -12, -19, -14, -28, 2, -18, -20, -10, -22, -16, -4, 5, -11, 1, 3, -21, -13, -13, -2, 8, -14, -22, -14, -4, -21, 0, -21, -2, 5, -10, 4, -13, -18, -3, -15, -17, -11, 0, -1, -17, -3, -11, 3, -12, -5, -19, -8, -16, -19, 1, -2, -6, -21, -3, -12, -13, -18, -19, 6, -4, -12, 1, -12, 5, -7, -19, -21, -15, -6, 0, -2, -18, -4, 3, -15, 0, 0, 2, 6, -7, -15, -1, -14, -12, 1, -20, -7, 2, 2, -12, 5, -10, -18, 0, -21, -5, -12, 1, -16, 4, -7, -17, -13, -19, -10, 0, -9, -8, -7, -15, -2, -20, 2, -2, -15, 2, -26, -5, -9, -6, 6, 5, 5, -14, -17, -16, 3, -7, 1, -22, -22, -15, 0, 0, 2, -13, -9, -3, -18, 2, -11, -12, 2, -6, -19, -21, 6, 5, -20, -8, 4, -12, -24, -21, -12, -1, 1, -23, -10, 1, -11, -10, 4, 2, -11, 3, 0, -19, 2, 5, -21, -2, 1, -1, -10, 2, 4, -8, -17, -21, -3, -15, -20, -22, 10, -16, 5, -22, -10, 1, -16, -23, -11, -18, -19, -15, -23, -4, -18, -8, -14, -19, -7, -10, -4, -20, -6, -11, -2, -5, -9, -11, -15, -4, -20, -4, -19, -15, -11, -4, -18, -4, -17, -19, -11, -8, 2, -3, -6, -2, -3, -4, -7, -17, -13, 3, -17, 5, -10, -4, -4, -14, -22, -2, -32, -3, -21, -3, -18, -15, -8, 4, -15, -9, -10, -17, -5, -18, -17, -4, -11, -3, -2, -7, -10, -13, -20, -12, -12}
, {-13, 9, 19, 20, -49, 2, -63, -46, -14, 10, 18, 32, -14, 5, -96, 18, 12, -3, -7, 11, -4, -140, -49, 1, 16, 15, -9, 90, 27, -1, -68, -80, 21, -33, -130, 44, -100, 1, -7, 75, 49, 19, -67, -61, 21, -8, -127, 73, 57, 5, -61, -12, 61, -18, 68, -36, -55, 36, 0, -41, -77, 9, 17, -75, -31, -82, 41, 101, 126, 20, -13, 50, 38, 14, -19, 1, 40, -26, -158, 2, -8, -92, 23, -62, 2, -14, 27, -53, 8, 34, 22, -82, 65, -32, -58, -76, -37, -17, -69, -59, -193, 18, -1, 11, 7, -16, -52, -63, 15, 8, -21, 1, 7, 12, -18, -45, 31, -18, 24, -86, -45, 14, 2, -16, -108, -46, 6, 35, 9, -91, -90, -88, 31, 9, -4, 107, -6, -5, -11, -22, -4, -102, 41, 9, 64, -22, -13, -43, -97, 56, -107, -20, -2, -113, -20, -112, -26, 5, 92, -39, -78, 18, -12, -82, 100, -26, -17, 1, 29, -40, -20, 7, -42, -28, -8, -31, -35, 5, 108, 0, 4, 52, -14, -15, -132, 34, 11, -44, 73, -71, -1, -34, -10, -13, -13, 60, -73, -6, -68, 48, 57, 1, -5, -14, 11, 57, -6, 7, 67, 95, -20, -26, 13, -170, -72, -6, -17, 16, -8, -56, -19, 29, -311, -15, -8, 30, 64, -15, 65, -70, 7, -39, -152, -72, -65, 18, -9, -10, 34, -18, 13, -10, 7, 76, 23, 170, -57, -75, 101, 45, 20, -70, 62, 30, -2, -38, -21, -82, -107, -98, 49, -12, -20, 65, 43, 4, -20, -59, 4, -103, -30, 48, -36, 6, -15, -57, -182, 45, -111, -31, 1, 27, 7, -17, 3, 44, -137, 3, -7, -17, 46, -35, 32, -58, 18, -99, -97, 34, -83, 31, -77, -16, 88, -13, -191, -16, 24, -25, -35, 80, -72, -67, 20, 79, 14, -124, -15, -22, -5, 39}
, {-70, -85, -93, -28, 75, -6, -5, -35, -124, 0, -13, 11, 19, 89, -112, -83, -5, 63, -17, 9, -173, -59, -80, 0, -17, -102, 1, -28, 4, -104, 126, -3, -59, 78, -14, 11, -58, 41, -1, -83, -76, 6, 33, 3, -122, 16, 92, 58, 11, 7, -52, -21, 9, 11, -54, 41, -16, -33, -13, 14, 113, 65, -13, -6, -43, -14, 3, 20, -170, -19, -74, -14, 74, -5, -5, 100, 11, 118, -254, 116, -16, -66, 0, 35, -41, 5, -38, -23, -16, 167, -16, -84, -7, 14, -56, -146, -18, 70, 106, -15, 18, 14, -11, 18, -73, -23, -19, -52, -2, -2, -136, -89, -54, -15, 181, -44, -63, 45, -69, -26, -46, -9, -7, -94, 28, 15, -16, -58, 9, 79, 32, 38, 11, -19, -26, 13, -26, 5, 5, 67, 6, 23, -145, 94, 7, -74, -14, 15, -143, 35, 116, -24, 5, 135, -18, -70, 80, -99, -69, -34, -38, 17, 24, 31, -50, 110, -15, -45, 67, 28, -30, -67, -11, -21, -30, -60, 125, 1, -114, -105, -21, 138, -7, 127, -90, 38, -23, 68, -27, -75, 0, -124, -27, -11, 2, -23, -91, -18, -59, -49, -165, -21, 1, -57, -33, 107, -196, -79, 59, -60, -15, 36, -81, 73, 11, 0, -11, -114, -19, 44, -24, 95, 38, -169, -91, 22, -60, -73, -85, 119, -18, 0, 26, -8, 34, 21, -32, -2, -55, -89, 68, -9, 117, 88, 38, -11, 4, -47, -37, -15, 2, -28, 36, -27, 6, -60, 27, -111, 86, -112, -114, -11, 25, -10, -39, 1, 8, 65, -46, 105, -105, 41, -18, -92, -11, -50, -105, -50, -42, -8, -4, -84, -14, 58, 7, -10, -28, -110, -141, -9, -72, -61, -3, 44, -4, -110, -79, -11, 32, 39, -21, -7, -98, -155, 45, -9, 193, 21, -68, 131, -84, 12, -15, -12, -18, -76, 95, -39, 2, -2}
, {-17, -21, -4, -16, -3, -16, -6, 1, -8, 0, -13, -14, -3, -5, -19, 2, -21, -20, -8, -1, -5, 5, -17, -12, 3, -1, -3, -10, -2, -14, -7, -8, -17, -4, -6, -15, 4, 0, -20, -14, -20, -3, 1, -20, 5, -20, -15, -1, -7, 0, 6, -6, -8, -17, -2, -9, -3, -10, -5, 0, -11, 1, 5, 0, 3, -8, -7, -6, 6, -13, -4, -7, -5, 2, -10, -1, -17, -9, -7, 6, -5, -5, -22, -10, -3, -14, 5, -11, -12, -10, 5, 4, -12, -10, -23, -7, 0, 9, -10, 1, 5, 3, -3, -10, -18, -13, -4, -3, -15, -14, -18, -14, 3, -8, 2, -13, -12, -9, 9, 3, -7, -20, 2, -5, -10, -5, 4, -15, -2, 1, 6, -9, -19, 3, 12, -2, 4, -4, 0, -13, -8, -10, -9, -7, -10, 6, -16, 7, 1, 4, -10, 0, -14, -14, 3, -2, -22, -1, -8, -12, -20, 5, -21, -12, -6, -8, -12, -12, -5, -21, -12, 0, 5, 2, -21, 4, -22, 4, 3, -6, -17, -3, 6, 3, -8, 0, -12, 5, -3, -4, -5, -9, -8, -15, -14, -11, -5, -7, -10, -6, -5, 0, -12, 2, -3, -19, 4, 3, 5, -1, -13, -8, -5, -2, -22, -14, 5, -3, -12, -8, -19, 1, -10, -7, -6, 8, -19, -17, 0, -4, -6, -15, 5, -11, -16, -10, 4, -4, 4, -20, 1, 4, 5, -18, 4, -17, -12, -15, -22, 2, -18, 5, -2, -9, -12, -7, -18, -22, -16, -17, -16, -22, -11, -14, -13, -17, -9, -3, -11, -2, -20, -5, -21, -1, -11, -19, -20, -16, -8, -2, -17, -3, -7, -18, -17, -2, 1, 6, -15, 1, -10, -18, -10, 3, -16, 3, -18, -16, -12, -11, -3, 3, 5, -5, -7, 6, -16, -10, -19, -7, -15, -14, -2, -17, 2, -8, 6, -15, -7, -14}
, {8, 9, 23, -19, 5, -18, -7, -13, -45, 6, -2, 132, -5, -74, -10, 16, 45, -80, 21, 14, -53, 55, 30, -2, 4, -164, 20, 143, 31, 114, -107, -96, -15, -29, 103, -14, -90, -73, -4, 28, 4, -76, -89, 24, 8, 2, -27, 20, -2, 0, 108, -67, -30, 30, -5, -96, -36, -9, 15, 26, -24, 66, 22, 11, -16, -101, -16, 11, 90, -25, 7, 56, -15, 6, 14, 56, -10, 80, -56, 20, -71, 13, 16, -7, 67, -65, -67, -4, 8, -4, 5, 39, -41, -74, -10, 18, -36, 46, -87, -39, -34, 60, 19, -8, -31, -86, 22, 65, -18, -18, 44, 57, -29, 5, 61, 123, -19, 74, -82, 22, -57, -63, -6, 69, 54, 45, -5, 59, 3, -21, 10, 48, 21, -12, 49, -91, 92, -23, 9, 28, -43, -141, 85, -114, 23, -29, 14, 20, 9, -2, 20, -65, -3, 88, -1, 68, -28, 17, -153, -97, 3, 106, 10, -27, -91, 7, -33, 26, -77, -49, 121, -14, -50, 8, -44, 13, 62, -1, -1, 42, -26, 34, 37, -18, 0, -49, -19, -49, 8, -29, -30, -5, -11, -84, -20, -6, 62, 5, 79, 15, 43, -13, -8, 14, -2, -41, -33, -33, -91, 44, 15, 29, -87, 98, -62, -10, -10, -22, -20, 38, 8, 13, -11, -1, -60, 124, 97, 31, -36, -13, -43, -4, 49, -8, 106, -17, -67, 4, 65, -33, 10, 19, 56, 23, 24, -66, 62, 69, -48, 56, 14, -18, 38, -23, -1, -58, -3, -15, 42, 46, 57, -12, 62, -48, -14, -6, 15, -55, -5, -154, -27, -81, -65, 10, 5, 7, 66, 23, -94, -1, -2, -18, -11, 17, -27, 34, 6, -63, 30, 13, 26, -6, 151, 17, 11, -7, 16, 32, 17, 75, -85, -14, -3, -32, 56, -19, 15, 45, -3, 82, 50, 104, 17, 14, 5, 15, 110, -13, 24, 68}
, {2, -44, -7, 43, 50, 12, 87, 66, -42, -2, 8, -71, 9, 22, -72, -75, 12, 4, 8, -69, -11, -28, -9, 12, 2, -38, 9, 19, 7, -6, -90, 51, 13, -45, 59, -19, -3, -31, 0, -62, 35, 7, 4, -7, -30, 17, -44, 13, 35, 20, 18, -15, 35, 25, 63, 51, 6, 0, 18, 7, 23, 39, 10, -62, 9, -3, 72, -12, -15, -1, -57, -2, 10, 0, 19, 70, -92, 27, -41, 55, 1, 37, 14, 38, -14, -74, -3, 15, 7, 57, -6, 16, 15, -90, 50, -62, -44, 112, -23, -35, 19, 58, -7, -11, -26, -5, 35, 17, -104, -1, 22, -11, -13, 1, -39, 73, -28, 146, -51, -28, 21, -2, -6, -67, -34, -52, 11, 10, 11, -101, -56, -37, -28, -4, 161, -12, 46, 11, 15, -35, -23, 38, 52, 13, 6, 58, 4, -40, -25, -59, -36, 19, 17, 21, 20, 18, -7, -55, -109, 15, -1, 22, 70, -47, -10, -46, 18, -23, 50, 1, 16, -65, 30, 8, -34, -9, 35, 6, -29, -50, -13, 35, 76, -2, 38, 4, 20, 58, -16, -18, 19, -62, 8, 0, 10, 43, 37, -4, 88, 56, -23, 8, -2, 61, -77, -7, -61, 34, 8, 68, 3, -23, 36, -19, 43, 17, -3, -28, 18, 12, 7, -84, 60, -83, 52, 93, -87, -26, 27, 2, -5, -15, 11, -9, 39, 16, -8, 18, -5, -24, 39, 1, -10, 21, 48, 73, -27, 7, 19, 5, 19, -19, 46, 28, 14, 32, -7, 52, 38, -20, 112, 14, 133, 54, -32, 19, 18, -33, -80, -40, -85, 51, 9, -6, -3, -85, -11, 57, 7, 13, 21, 18, 21, 6, 4, -25, -118, -6, -14, 28, -5, -7, -51, -41, 3, 19, 56, 12, -45, 52, 9, 17, 52, 59, 25, -6, -18, -25, 46, 57, 7, 12, -71, 15, 2, -44, 9, -10, 15, 52}
, {-20, -15, -1, -21, -12, -22, -5, -6, -5, -1, -4, 2, -2, 6, -11, -12, -17, -5, 1, -5, -10, 1, -16, -8, -21, -17, 4, 0, 3, -8, -11, -1, -9, -11, -6, -17, -4, 1, 5, -14, -14, 2, 6, -9, -13, 6, -22, -20, 4, -19, 6, -22, -19, -6, -6, -15, -7, -19, -13, -19, -4, -7, 2, -3, -4, -19, -22, 6, -13, -8, -14, -7, -21, 4, -10, -18, -3, -20, -19, -18, 4, -14, 1, 1, 1, 3, -2, 0, -4, -4, 3, 3, 6, -2, -7, -6, -1, -14, -8, -8, -18, -10, -11, 0, -15, -3, -22, -18, -5, -7, 2, -4, 3, -14, -7, -5, 4, -21, -14, -15, -2, -17, 6, 4, -4, -20, -15, -16, -14, -5, -8, -18, -7, -2, -12, -1, -17, 0, -9, -16, -18, -17, 1, -9, 3, 4, -8, -4, -23, -9, -12, 6, -2, -13, -13, -1, -6, -4, -22, -12, 1, -14, -20, -8, -15, -7, 5, -8, -19, -15, 1, 4, -11, -5, -1, -18, -2, -16, -6, -12, -15, 0, -10, 4, -9, 3, -9, -2, -10, -20, -4, 1, 0, -1, -7, -17, -19, -21, -11, -18, 2, -21, -5, 6, -19, -17, -7, -10, -3, -2, -13, -20, -14, -20, -22, -11, 4, -12, -10, 0, -16, 6, -20, -3, -12, 14, -8, -9, -22, -13, -9, -4, -8, 0, -2, -1, -14, -5, 3, -5, 1, -5, 2, -1, -8, 2, -18, -18, -18, -10, -8, 0, -1, 6, -6, -15, 0, -9, -3, -20, -22, -15, -3, -15, -10, -20, -11, -5, -15, -16, -17, -3, 4, -8, -5, 1, -12, -20, -16, -10, 4, 5, 3, -18, -8, -16, -5, -3, -2, -9, -2, 8, -11, -9, 5, -7, -22, 0, -2, -19, -14, -21, -1, 1, -21, -3, -7, 5, -14, -21, -8, 1, 5, -3, -8, 6, -22, -22, 1, -22}
, {-19, -11, -9, -58, -19, -11, -2, 11, 1, -20, -16, -41, -63, -11, 7, -18, 1, -37, -11, -7, -22, -24, 10, 3, -14, -10, -5, -13, -19, -15, -20, -11, -6, 40, -8, -10, -19, -5, -6, -53, -33, 4, -28, -16, -41, -14, -45, -14, 2, -11, -17, -12, -15, -30, -2, -12, -36, -20, -5, -55, -6, 6, -10, -38, -1, -25, 0, 12, -22, -1, -8, 28, -43, -9, -4, -46, 4, -11, -3, 14, -8, -28, -1, -7, -30, -43, -44, -7, -4, -22, -10, -13, -18, -30, -20, -26, -23, -7, 2, 7, -6, -28, 2, -27, -29, -12, -26, -37, -35, 1, -51, -36, -15, -22, 17, -24, -5, -14, -20, -22, -9, -19, -23, 24, -15, 3, -22, -33, -21, -41, -2, 0, -11, -14, -14, -75, -22, -22, -17, -20, -41, 4, 17, -16, -1, -22, -10, 0, -21, -45, -13, -19, -13, 0, -19, -3, -11, -15, -10, -11, -5, 6, -9, -23, -8, -22, -6, -26, -9, -22, -43, -2, -30, -22, -30, -9, -20, -10, -27, -21, -10, -1, 12, 2, -25, -19, -7, 1, 2, -4, 5, 2, -14, -21, -27, 0, 4, -11, -5, -38, -17, -1, 19, -16, -51, 2, -30, -11, -19, -33, 1, -26, -38, -11, -17, 4, -21, -24, -1, -19, -21, 2, 3, -8, -1, 15, -18, -47, -10, 1, -14, -45, -20, -22, -12, -9, -44, -17, -51, -30, -12, 5, 15, -7, -24, -5, -52, -28, -35, -2, -11, 8, 0, 2, -19, -18, -9, -9, -11, -9, -16, -17, -6, 44, -21, -16, 12, -34, -41, -22, -26, -12, 4, -26, -4, -20, -37, -20, -4, -17, 0, 0, -19, -23, -23, -14, -8, -1, -8, -36, -16, 14, -17, 1, -16, -23, -30, -3, -37, -13, -17, -9, -39, -23, 4, -14, -20, -6, 1, -14, 56, -25, -16, -23, -14, -12, -1, -18, 6, 3}
, {19, 44, 39, 4, -3, 11, -104, 26, 86, 1, 8, -86, 12, 0, -101, 19, 3, -1, 6, 48, 44, -64, 4, 13, 6, 65, 5, 43, 11, 6, 34, -112, -17, -129, -5, 7, -97, 19, 3, -20, 12, 0, 4, -61, -48, 16, 43, 0, 2, 3, 8, 36, 30, -35, 21, 16, -59, 11, 4, -1, -6, 50, 12, -61, 12, -35, 9, 25, -5, 14, -91, 62, 7, 15, 29, -15, -16, 54, -6, 37, 24, -14, 6, 13, -32, -9, 38, 15, 4, -12, 0, -12, 17, 14, -51, 4, -11, 50, 45, -24, -12, -61, 5, 28, 23, 7, 44, -114, 5, 18, -116, 16, -7, 3, 24, -34, 52, 86, 91, 105, -68, 21, 19, -122, -19, 31, 3, -58, 4, -3, 43, 60, 19, 15, -179, -10, -26, 22, 13, 34, -10, -33, 68, -141, 2, -34, 22, 35, 36, -18, -21, 4, -1, -22, 15, -8, 6, 64, 31, -34, 29, -31, 103, 3, -22, -4, 7, 0, -44, 5, 143, -223, -25, 10, 12, 1, 27, 19, 9, 38, -37, -191, 15, -1, -41, 31, 9, -43, 32, -3, 0, -144, 14, -26, -10, 4, 50, 21, -48, -42, -8, 2, -3, 71, 46, 18, 97, -22, 38, 53, 8, 36, -15, 30, 37, 15, 4, -22, 3, 1, 2, -26, 69, -20, -1, 7, 16, 8, 93, 66, 9, 7, 64, -10, 31, 92, 23, 7, -52, 4, 6, 6, 9, -49, -8, 126, 38, 63, -1, -8, -4, -75, -55, -10, 16, 145, 17, -44, -93, 45, -21, -7, -80, 37, -33, -3, 2, 83, 34, 18, 69, -115, -8, -16, 14, -23, 76, -3, -12, 3, -2, -45, 17, 17, 2, 10, 71, 89, -16, -29, 14, -102, 17, -20, 19, -15, 5, -16, -39, -58, 25, 16, 38, 78, 6, 10, -3, -30, 44, -18, 65, -18, 6, 11, -4, 46, -22, 7, 7, -24}
, {7, -6, -4, -12, -14, 0, -11, -15, -5, 1, -14, 6, -11, -9, -23, 2, 0, 0, 3, 2, -3, -13, 2, -13, -7, -4, -15, -20, -3, -19, 1, -18, 5, -4, -9, 0, -7, -14, 0, 2, -22, 0, -1, -21, 0, -6, -9, -11, 5, -19, 5, -21, -22, 1, -10, -22, -15, -6, -15, -4, -3, 4, -7, -13, -13, -13, -16, 5, -20, 5, -8, 0, -10, 0, -15, -15, -5, 7, 1, 3, -20, 3, -2, -6, 1, -17, -12, -12, -5, -9, 4, -20, -2, 2, -8, -12, -15, -7, 2, -12, -7, -13, 5, -11, -19, 2, -18, -14, -4, -12, -13, -21, -20, -10, -17, -2, -2, -5, -22, -4, -10, 0, -4, -20, 2, -12, -21, -14, -3, -17, -8, -13, -18, 0, -8, 3, -4, -22, 4, -14, -4, -21, -19, 6, -11, -11, -13, -7, -8, -18, -6, 2, -5, -1, -4, 2, -23, -1, 1, -14, -11, -16, -7, 1, -21, -8, -3, -9, -7, -9, -18, 1, -14, -2, 1, -4, -23, -19, -22, 2, -21, -2, 5, -21, -16, -4, 3, -21, -17, -2, -20, -20, 2, -20, -14, 5, -18, -10, -18, -2, -2, -9, -12, -12, -17, -19, -17, -12, -20, -7, 5, -16, -10, -11, 3, 5, -19, 2, -11, -3, -19, -17, 3, -6, 6, -13, -20, 5, -12, -10, -13, -21, -4, 0, -2, 6, -11, -3, 2, 4, 6, -12, -3, -4, 4, 2, -15, 2, -19, -21, 5, -7, 7, -20, -6, -17, -22, -4, 0, -4, 3, 3, -1, 2, -9, -22, -12, -14, -1, -18, 5, -9, -18, -13, -20, 0, -11, 0, -5, 6, -6, -21, 6, 4, 0, 3, 4, -15, -13, 4, -21, -17, -5, -11, 5, -16, -4, -15, -17, 3, -9, -7, -22, -17, -14, -4, -4, -8, -20, -8, -1, -7, -3, -4, -5, -8, -2, -4, -1, -9}
, {-6, -75, 37, 15, 36, 12, -25, -84, -40, 16, -9, 26, 50, -106, -80, 79, 11, 18, 13, -121, 35, 20, -87, 17, 18, -54, -10, 32, 5, -7, -75, -7, -4, -27, 41, -9, 61, -48, 12, 45, 18, 14, 2, 6, 61, -4, 69, 39, 7, 4, 20, -11, -28, 41, -22, -40, 16, -29, 24, 18, -111, -54, 12, -55, -13, -28, -13, 13, 14, 7, 150, -127, -12, 2, 4, -16, 61, -4, -124, -19, -30, -37, 10, -26, 24, -30, -129, 23, -2, 29, -24, -63, 1, 0, -52, 149, 70, 33, 19, 4, 40, 4, -8, 5, 30, -14, -32, -42, -23, -1, 82, -5, -23, 14, 96, 2, -43, -77, 20, -42, 16, -16, 12, 18, 88, 27, -12, -33, 7, -59, -30, 41, 36, 11, 22, -89, -12, 21, 9, 50, 38, 40, -137, -43, -23, 3, 8, -84, 20, 7, -11, 13, -10, -34, -4, -41, 34, -24, -80, 37, -3, -42, 90, -56, 71, -48, 4, -1, 4, 17, 5, 38, -8, -16, 12, 14, 61, 17, 70, 27, -54, -34, 8, -46, 53, -38, 21, 23, -9, 108, 19, 21, 4, -60, 31, -4, -24, 9, 23, -125, -18, 21, -20, -17, 6, 8, -29, -14, -32, 29, -1, -29, -5, 4, -57, 14, -1, -49, -18, -30, 22, 10, -114, -7, -11, -79, 8, -31, 8, 5, 7, 30, -34, 21, 9, -23, -13, -19, -31, 53, -47, 9, 37, 29, -31, 14, -38, 19, 37, -21, 17, 7, -57, 41, 21, -45, 7, 1, 58, 8, -2, 17, 136, -164, -2, 19, -3, -56, 40, 92, -1, 53, -4, -21, 17, 16, 55, -64, -19, 27, 8, -42, -5, -23, 25, 29, 63, 19, -22, -13, -3, -30, -26, -10, 3, 16, 10, 5, -84, -36, -2, -13, 16, 20, -147, 7, 42, 45, -60, 16, -104, -36, 71, -42, 18, 9, -61, 17, 30, 12}
, {-16, -7, -9, -8, -10, 3, 14, -21, -8, -2, -23, -18, -21, -21, 4, -17, 5, -23, -8, -8, 3, -4, -9, -9, -10, -14, -15, -9, -17, -18, -20, -6, -20, -7, -19, -15, -10, 5, 3, -17, -25, 2, -23, -18, -24, 1, -23, 3, -3, -6, -9, -24, -3, -10, -4, -4, -11, 2, -12, -14, -18, 5, -8, -16, -11, 7, -6, -5, -14, 1, 4, -19, -6, -13, -20, -22, 0, -6, -15, -17, 5, -22, 3, -13, -20, -22, -20, -22, -17, -1, -15, -7, -13, -8, -22, -19, -20, 1, -9, -11, -2, -21, 1, -14, -7, 5, -17, -22, -13, 2, -14, 2, -14, 6, 0, -22, -9, 0, -2, -3, -13, -19, -7, -3, -6, 6, -3, -3, -13, 1, -13, -14, -5, -18, 3, -15, 3, -12, -22, 4, -20, -17, -12, -20, 6, -12, -9, -10, -1, 2, 2, -11, 1, -2, 3, -2, -11, -12, -10, -3, 1, -14, -15, -19, 4, 5, -23, -21, -1, 0, -13, -4, -11, -1, -11, -4, 5, 1, -14, -4, -8, -5, -12, -2, -24, -8, -5, -34, -20, -10, 1, -6, -15, 4, -17, -6, -23, 0, -3, -19, -12, -7, -17, -21, 1, -9, -16, -11, -18, -15, -17, -16, -16, 2, -12, 5, 6, -11, -11, -18, -12, -14, 3, -12, 4, -3, 4, -15, 4, -1, -18, -12, 4, -20, -2, -7, -12, -14, -22, 3, -12, -7, -6, -2, -8, -12, -8, -11, -18, 4, -11, -5, -12, -23, -14, 6, -18, -12, -14, 11, -12, -10, -7, 2, -3, -17, -4, -20, 2, 6, -17, -13, -2, -7, -22, -7, -3, -8, -6, -4, -22, -3, 2, -21, -14, -13, -2, -20, -21, 3, -7, 3, -13, 1, 3, -6, -12, -18, -11, -21, -5, -17, -18, -12, 4, -19, -1, 6, 1, -21, -3, -11, 3, -6, -22, -4, -14, -14, -1, 3}
, {18, 12, -4, -10, -100, 13, 109, -111, -20, 1, -1, -14, 37, -9, 67, -6, 8, 54, 16, 6, 18, -31, -50, -7, -6, 4, 16, -21, 17, 3, 22, -16, 31, -32, 3, -114, 7, 21, -3, -17, 21, -1, -9, 42, 13, -13, 37, -45, 58, 14, 65, -17, 6, -27, -138, 40, 17, 2, 22, 49, -37, -22, 8, 45, 12, -143, -58, 51, -62, 6, 163, -6, -67, 1, -10, 50, 16, -21, 57, -105, 22, 26, 5, 28, 46, -74, 18, 0, 11, 77, 16, -28, -6, -13, -87, -30, 50, -43, -12, 2, -59, -123, 8, 9, 69, -6, -45, 50, 20, 9, 69, -2, 28, 21, -94, -90, 10, 62, -24, 8, 39, 3, -1, 94, -4, 4, 10, 61, 0, -108, -79, 54, -210, -5, 33, -84, -140, -3, 3, 84, 54, 12, 87, -120, 16, -18, 6, -13, -60, -14, -64, 1, 9, -135, -6, 7, 13, -71, 62, -111, -15, -35, 74, -174, -63, 116, 7, 13, -23, -2, -28, -40, -9, 6, -17, -10, -22, 15, 56, -7, 0, 17, -34, -54, 29, -2, -8, 57, -37, -2, -2, -9, 14, -102, -22, 42, -13, -1, 42, -9, -170, 9, 2, -43, 32, -34, -7, 16, 6, 48, -7, -40, 5, -17, 12, 12, 15, -55, -2, 7, 19, 25, -49, -1, 19, 96, -38, -54, -16, -23, 2, -54, 22, 3, -35, 53, 21, -10, 29, -84, 7, -6, -71, -20, -51, -53, -175, 3, 71, -4, 3, 97, 13, 48, -5, 5, -1, -81, -4, 64, 19, 18, -69, 83, 79, 22, -1, -99, -6, 28, 22, -55, -20, 30, 6, 33, -62, 80, 2, 9, 2, 44, -1, 41, 7, 52, 9, 57, -16, 93, -30, -96, 22, 72, 2, -11, -6, 3, 27, -42, -5, -7, 28, -115, -21, 18, 133, 1, 41, 73, -69, 23, 91, -20, 14, 95, 4, 31, 20, 6}
, {14, 76, -83, -9, -38, -11, 74, 97, 5, 11, 8, 76, -41, -94, -136, -141, -14, -36, -12, -31, -74, -61, 24, -15, 4, 57, -21, -27, 19, -187, -6, -18, -41, -185, -95, -15, 104, 40, -18, -55, -25, -25, -45, -3, -76, -2, 93, 6, -42, -4, -40, 94, 17, -36, 77, -18, -49, 18, -7, 35, -62, -25, 21, -38, -2, -68, -74, -21, -275, -17, -24, -79, -29, -4, 5, 44, -122, -37, 108, -61, -15, -166, -1, 44, -131, -68, -18, 3, -2, 88, 0, -17, -9, -39, -21, 6, -16, -122, 3, -153, -42, -42, -8, -99, -54, -29, -42, -19, -66, -14, -67, 33, -32, -6, 38, -43, -50, -42, 25, -38, -173, -43, 8, -184, -59, -29, 13, 6, 18, 25, -159, 4, 87, 9, 73, -98, 11, 10, -16, 125, -3, 79, -74, 9, 0, -27, -8, -62, -62, -68, -16, -9, 8, 39, 11, -3, 3, -81, -84, 99, -7, 50, 92, -188, -43, 34, 0, -93, -26, -7, 30, 5, 11, -3, 40, -15, -55, -10, -72, 35, -95, 7, -53, -16, 7, -4, -5, -19, 34, 0, 9, -53, 3, -5, -147, 60, 75, -6, 13, -32, -138, -8, -11, 188, 23, -135, -92, -65, 31, -17, -14, -92, 73, -81, 58, 2, -4, -135, 17, -17, -8, -86, -75, -26, 9, 44, 182, -36, -1, 94, 0, -22, -21, 4, -130, 17, 27, -4, 91, 18, -51, -21, -56, 27, -22, 29, -20, -40, 14, 10, -20, -157, -34, -13, 6, -15, 20, -40, -52, 104, 79, -11, -113, -40, -64, -20, -3, 125, 2, -52, 3, 26, -17, -3, -9, -54, 52, -56, 20, -13, -4, -89, 14, -13, -12, -66, -33, -73, -39, 84, 126, -98, -20, 73, -12, 49, 12, -6, -82, -22, -32, -8, -112, -28, -68, 2, -15, 16, -93, -35, -175, -90, 36, -11, 8, -141, 0, 40, -1, -35}
, {-19, 5, -4, -16, -15, -6, -19, 0, 1, -4, -12, 4, -9, -15, -6, -1, 1, -13, -5, -2, -22, -19, -11, -23, -22, -16, -19, -4, 5, -19, -6, -15, -11, -12, 3, -20, -11, 5, 6, -23, -3, 1, -11, -12, -22, 4, -15, -19, 5, -7, -11, -1, -8, 2, -4, -17, -25, -8, -16, -3, -18, 6, -9, -21, -13, 3, -12, 2, -23, -2, -12, 4, -2, -11, -8, 3, 2, -18, -8, -16, 5, -10, -18, 10, -11, -7, -23, 1, -5, -24, 0, -23, 2, -14, -14, -16, -12, 4, -22, -8, -9, -8, -3, -1, -15, -9, -9, -2, 0, 0, 1, -5, -14, 5, 6, -11, -23, -2, -18, -2, -8, -17, -18, 7, -7, -1, -9, -3, 4, 1, -5, -16, -14, -5, 5, -10, -13, -3, -23, -23, -17, -4, -12, -18, -2, -15, -4, -15, 1, -19, -21, 0, -20, -8, 1, 0, -2, -13, 2, -14, -7, 0, -9, 3, 4, -11, -4, 2, 1, -3, -20, -20, -13, -10, -4, -22, -2, 5, -6, -23, -2, -22, 7, -5, -23, -13, 2, 8, -13, -3, -11, -1, -10, -25, -2, -21, -23, 6, 6, -9, -3, -10, -20, 5, -4, -5, 5, -1, -6, -19, -9, 1, -20, -4, -10, 1, -12, -2, 5, -16, 3, -7, -8, -19, -7, -12, 2, 1, -16, -6, 2, -13, -19, -4, -12, -9, -3, -22, -23, -10, -10, 4, -8, -25, 3, -2, 2, 3, -5, 3, -2, 3, -10, -14, -17, 11, 0, -2, -5, 21, -19, -21, 11, -19, -5, -6, -9, -17, -2, 1, -22, -10, -9, -3, -1, 1, -18, -19, 1, -19, -2, -22, -7, -6, -6, -12, -11, -6, -7, -14, 6, -4, -11, -10, -19, -8, 3, 0, 0, -17, -15, -1, 3, 1, -12, -18, 4, -23, 3, -23, 4, 2, -1, 6, 1, -5, -19, 2, 2, -1}
, {-7, -19, -5, -8, 0, -7, 9, 13, -3, 2, 16, -5, -42, -21, -5, 3, -14, -3, 4, -20, -10, -19, -17, -17, -5, -1, -19, -11, -21, -7, -9, -3, -21, -9, -21, -16, -17, -13, -12, -41, -15, -6, -8, -18, -16, 5, -12, -23, -20, 1, -14, 5, -2, -20, -16, -11, -14, 4, -2, -11, 3, -17, 6, -14, -8, -3, -8, -31, -17, -7, 9, -7, -17, 5, -9, -14, -6, -13, -3, -4, 3, 3, -6, -18, 5, -1, -18, -18, -11, -1, 4, -18, -6, -17, -18, 2, -3, -18, -19, -6, -18, -13, -2, 5, -10, 1, -5, -18, 6, 0, -2, 5, 4, -23, 3, 7, -2, 5, -6, 6, -22, 7, -7, -13, -12, -22, -4, 20, -4, -17, -17, -41, -29, 3, 5, 6, -24, -4, -9, -16, -10, 0, -1, 0, -3, -20, 5, -18, 18, -5, -3, 4, -5, -13, 6, -12, -3, 1, 5, -11, -7, -8, -16, -4, -14, -19, -2, -7, -7, -4, -20, -7, -4, -12, -9, 2, -3, -15, 7, -3, -18, -17, 22, -17, -24, -15, -18, -22, -21, -11, -23, -13, 4, -5, 23, -17, -21, 2, -7, -4, -23, -15, -8, -19, -19, 0, -4, -5, -12, -15, -2, 1, 3, 1, 2, 6, 2, -1, -14, -3, -3, 1, -13, -24, -8, -11, -10, -21, -6, -18, -17, -12, 4, -19, -21, 0, 0, 3, -34, -13, -2, -16, -6, -4, -10, -1, -12, -13, -6, -10, 1, -4, -19, -5, -4, -17, -9, 5, -10, -26, -8, 4, -14, -20, -10, -14, -10, -5, -13, 4, -10, -9, -12, -20, -2, -33, 10, -21, -5, -4, -17, 4, -5, -6, 4, -3, -3, -7, 4, -17, -8, -22, -6, -21, -22, 0, -19, -16, -15, 2, -11, -18, -14, 2, -4, -13, -26, -20, 3, -10, -22, -22, -12, -1, 6, -26, -6, -15, 2, -9}
, {5, -115, -219, -82, -62, -7, -14, -36, -139, 17, 10, -109, -57, 35, 71, -142, 5, -12, -1, 6, -153, -33, -78, -4, 11, -19, 3, -4, -13, -63, 44, -111, 63, -39, -154, -119, -4, -41, 17, 11, -104, 1, -86, -26, -169, -6, -131, -63, 88, 15, 42, -35, -9, -57, -89, 38, -7, -28, 1, -70, 38, 20, 13, -41, 9, 5, 2, -40, 43, -11, 112, 80, -19, -7, 1, -112, -21, 56, 100, 73, -34, 13, 5, -28, 6, -44, -29, 9, 0, -63, 15, -1, -5, -50, -49, 57, -72, -43, 77, -54, 7, -41, -8, -55, -51, 18, 90, -89, -8, -20, 173, -26, 100, 19, -4, -38, 0, -4, 13, -72, 35, 13, -16, 36, -104, 15, 19, -129, -1, 39, -87, 17, -112, 15, 136, -44, -37, 11, -13, 55, -6, 103, -28, 33, -24, 2, -5, 37, 24, -2, 9, -18, 0, -9, -5, 26, -14, -108, 52, 62, -5, -57, -45, -96, 44, -8, -5, 31, -3, 26, -50, 12, -35, -1, 88, 59, 8, 15, -152, 24, -17, -29, -88, -103, 45, 13, -2, 83, 36, -94, -1, 66, 6, -24, -8, 15, 165, 16, 28, 3, -44, 6, 5, -67, 49, -6, 10, -35, -17, -26, 0, -63, -72, -82, -135, 1, 8, -39, -6, 4, -8, -83, -44, 7, 15, 107, -67, -47, -91, -80, 11, 17, -127, 20, 30, 2, -30, 9, 91, -28, 43, -4, 37, 16, 88, 4, 115, -51, -77, -19, -4, -19, 19, -59, 11, 19, 3, 23, -37, 33, 9, 15, 71, -19, -9, 20, 13, 156, -35, -43, 30, -65, 20, -54, 14, 23, -36, -68, 105, -2, 12, 131, 20, -18, 10, -31, -128, 39, 46, 15, -16, 25, -71, 54, -9, 18, -37, -6, 38, 15, -113, -13, 0, -48, 146, 12, -1, -30, -31, -66, -18, -35, 42, -3, 17, 42, 44, 76, 11, 19}
, {7, -2, 42, -25, 196, 13, 4, -26, -5, -9, 14, -46, -9, 7, 3, -83, -12, -29, -3, -67, -41, -96, 1, -30, -5, -59, -8, 34, 17, -34, -30, 50, 32, 53, 34, -63, 76, -6, -8, 6, -73, -22, -59, 46, 23, -14, -17, -13, -69, 9, -42, -39, -18, 56, -102, -36, -30, -43, -18, 10, -39, -7, 10, 36, 8, -61, -39, -32, -37, -8, -4, -57, -23, 6, -10, -52, -42, -74, -43, -37, -5, -72, 5, -65, -34, -65, -44, 8, -10, -84, 7, -38, -27, -71, -138, -77, -106, -49, -61, -59, -74, -27, -14, -59, -64, -11, -81, -20, -76, -12, -29, -76, -9, -1, -84, 45, -29, 11, -117, -66, -32, -32, 2, -54, 82, 13, -14, -27, -2, -53, -37, -82, -141, -15, 11, -63, -44, 9, -7, -87, -174, -67, -68, 38, -32, -104, -4, -73, -91, -97, -79, -16, -1, -12, -1, -9, 19, -78, -130, -121, -149, -43, 64, -76, -172, -73, -3, -20, -105, -9, -214, -1, -96, 1, -45, -153, 13, -6, -93, -35, -48, -5, -127, -102, -48, -28, -8, -63, 51, -13, 10, 3, -14, -24, 21, 39, -80, -13, 33, 52, 1, -13, -7, -35, -4, -77, -38, 61, -19, -143, -14, -47, -53, -52, -37, -9, 7, -92, 6, -4, -3, -49, -47, -90, -70, -73, -67, -14, -86, 29, 4, -6, 40, 5, -53, 44, -24, -6, 144, 183, -101, -6, -104, -1, -5, 15, -21, -37, -6, -3, -17, 121, -2, -33, -5, 59, 10, -34, -71, -34, -90, -4, 52, -48, -6, -12, -8, -124, 89, 45, -76, -13, -40, 85, 11, -48, -59, -132, -45, -8, -1, 31, 6, -5, -4, 20, 22, -19, 49, -24, 128, -19, -58, 58, -13, -66, -32, 9, -50, -26, -31, -15, 65, -92, 31, -14, -85, -11, -29, 28, -52, 355, -33, 14, 9, -59, -24, 4, 0, -3}
, {14, 63, -65, -71, 86, 12, -36, -3, 16, -5, -12, -41, 24, -61, 17, -20, 9, -11, 10, 29, 47, -1, -20, 3, -5, 62, 17, -31, 7, 53, -74, -30, 42, 51, 104, -9, 47, -55, 16, -35, 31, -4, 30, 12, -18, -3, 20, -44, 50, 21, -21, 54, -31, 2, 3, -48, 14, 12, 19, 75, -3, -41, -8, -36, 12, -11, 62, -42, -21, -7, 118, -80, 4, 2, -4, 26, 36, -26, -33, 70, -39, 26, 2, -42, 30, 4, -21, -5, 10, -21, -7, -23, 12, -20, 55, 87, -27, -42, -248, 22, 24, -35, -8, 21, -64, 14, 23, 160, 11, -12, 34, 72, -18, 10, -197, 14, 58, 111, -62, -63, 20, 0, -6, 77, 36, -12, 2, 112, -5, -50, 52, 12, -103, -1, -30, -18, -34, 3, -2, 67, -68, 47, -3, 12, -12, -15, -2, -16, 69, -99, -84, -5, -1, -65, -3, -18, -16, -14, -55, -28, 44, -58, 52, -2, -58, 42, -1, 25, 7, -5, -60, -40, 8, -5, 39, 35, 40, 14, 36, -23, -42, -23, -18, -85, 50, 7, -7, 52, 28, 28, -1, -45, 13, -53, 63, 28, 94, 20, -56, -59, -25, 15, -7, -24, 43, 15, -16, 8, 35, 11, -5, -39, -8, -7, -39, 17, 8, 17, 10, -5, 5, 32, -87, 117, -7, -8, 51, 9, -1, -52, 15, -38, -11, 17, -105, -59, 7, 12, -11, 19, -68, 7, 52, -81, 44, -2, -65, 9, 26, -10, 0, 87, -45, 28, 18, -31, -5, -73, -52, 45, -115, -6, 24, -110, -21, 21, -11, 49, 5, 40, 35, -37, -1, 20, 10, -12, 15, -16, 23, 19, 11, -58, -17, 20, 8, -1, 69, 126, 16, -11, 110, -138, 3, 9, 16, -58, 45, 3, -22, -22, 32, 11, -47, -23, -33, 19, 24, 64, -8, 21, -153, 24, 63, 41, 3, 37, -15, -32, 7, 36}
, {5, 6, -6, -14, -2, -22, -13, 18, -13, -15, -15, -17, -2, -18, -18, -7, -14, -19, -16, 4, -21, -21, -9, -17, -3, 5, 0, -16, 1, -17, 2, -7, 6, 4, -6, -19, 3, -4, -16, -21, -9, 2, 9, -14, -16, -2, 1, -1, -15, 13, -9, 4, -12, -2, -3, 6, -8, -4, -4, -13, -4, -23, -9, -11, -7, -12, -18, -17, -2, -2, -4, -5, -14, -21, -11, -6, -1, -21, 5, -12, -13, -2, -2, -18, 3, -16, -16, 0, -1, -10, 2, -2, 2, -18, -19, -17, -7, 10, -3, 4, -14, -11, -11, -22, -3, -11, -18, 5, -12, 1, -3, -17, -2, 4, -1, -18, -11, -3, -5, -10, 1, -16, -17, -7, -14, -3, 0, -13, -11, 4, 1, -12, 8, -23, -1, 11, 3, -4, -1, 2, -12, -2, -1, -15, -2, 3, -22, -14, -14, -17, -13, -14, -16, 4, -5, -22, -7, -20, -10, -4, -14, -1, -8, 11, 6, -2, -6, -21, -15, -7, -8, 3, -9, -12, -1, -15, -8, -7, -21, -16, 6, -3, -12, -8, -4, 5, -10, -1, 2, -5, 14, 13, 1, -16, -16, -18, -15, -22, 8, 8, -9, -8, -12, -17, -6, 0, -6, 6, -22, -19, 2, -5, 9, 2, -8, -13, -17, -22, -3, -11, -21, -6, 0, 3, -20, -4, -20, -1, -9, -6, 6, -7, -7, -22, -8, 3, -16, -13, 3, -9, -18, -12, 5, -11, -11, -17, -1, -4, 1, -13, -13, -15, -23, 5, -20, -14, 3, 9, -21, -12, -11, -14, -14, 5, -12, -3, 0, -20, -7, -8, 2, -17, -15, 4, -2, -8, -15, -6, -5, -15, -23, -23, -7, 3, -16, -14, -1, -14, 5, -4, -2, -4, -13, 0, 5, -15, -10, -19, -13, -13, 5, 5, -20, 6, -13, -15, -9, -20, 3, -2, 9, -23, -15, -8, -7, -14, -9, -15, -9, -15}
, {-6, -63, -69, 27, -64, -8, 21, -98, 23, -7, -9, -70, -33, 126, -23, -173, -16, 34, 5, 77, -160, 13, -42, -6, 18, -53, 10, -1, -7, 49, 20, 35, -64, 115, -52, -27, -117, 95, 7, -88, 28, 15, 86, -55, -32, -9, 16, -115, 36, -7, -183, -12, -68, 43, 49, 9, 51, 22, 12, 43, -3, -42, 1, -42, 16, -49, -35, 10, -17, -1, 75, 82, -16, 13, -20, -46, -52, -69, -85, -61, 30, -7, 6, 55, -40, -24, -80, -8, 6, 64, -4, -3, 4, 140, -109, -56, -58, 33, -10, -2, 1, 127, 8, -49, -37, -12, 10, 62, -53, 18, -50, -138, -13, 8, 17, 19, -50, -46, -4, 30, -58, 11, 16, -18, 17, 86, 4, -140, -5, 8, 11, 26, 98, 19, 63, -10, 96, 15, -8, 40, -63, -88, 29, -3, -11, 25, -2, 33, -182, 133, -3, 14, 4, 13, 8, -5, 1, 64, -24, -42, -73, 86, -65, -18, 52, -103, 12, -8, -29, -26, 41, -96, 4, -4, -35, -82, -52, -1, -174, -55, -62, -93, 66, -91, 36, 10, 1, 9, -23, -16, 20, -38, 15, 27, -68, -45, 81, 12, 91, 31, 0, 8, 9, 36, 18, 3, -9, -33, -10, 25, 1, -18, -97, -60, -54, 2, -3, 4, 14, -19, 1, 61, -56, -63, 0, 26, 136, 14, -131, 120, 9, -28, 7, 9, 4, 12, -57, 1, 37, 8, 30, 5, -46, -38, -100, -204, 84, -17, 25, -9, -7, -74, -9, -25, 6, 3, 7, 25, -72, 15, 1, 14, 46, 65, 62, 18, 10, 39, -58, 72, -28, -93, -3, 71, 18, 29, 0, 43, 13, 19, 22, 92, 2, 30, -7, 86, -9, -60, -58, -18, -79, -7, -32, -44, 17, -53, -36, 9, 69, -51, -45, 13, -41, -85, 38, 5, -235, -65, -19, -135, 30, -92, 29, 8, 18, 16, -19, 22, 21, -77}
, {12, -11, -32, -142, 96, 1, -6, 27, 4, -3, 17, 47, 36, -46, 46, -127, 6, 12, -3, 53, -14, -35, -48, -14, 17, 0, 16, 27, -28, 34, 51, 6, 12, 71, -1, -4, -59, 9, -1, 7, -39, -1, 55, -16, -38, 2, -10, -11, 1, -4, -61, -34, 33, 23, 8, -71, -21, 15, -7, -51, 55, -35, 8, -8, 3, -43, 17, -63, 30, 5, 25, -32, 6, 7, 3, 42, -8, 3, 49, 79, -29, 107, 3, 47, -38, -36, -28, 12, 0, 17, -4, 7, 15, 70, 60, -38, -43, 29, 81, 48, 80, -10, 46, -18, 2, -37, -47, 39, 25, -2, -101, 46, -29, -4, -101, 66, -64, -66, -12, -9, 49, -68, 11, -91, 84, 98, 8, -30, 12, 0, -39, -57, 65, -4, -138, 32, -39, 12, 3, -20, -2, 10, -9, 1, 9, -61, 7, -24, 27, -197, -57, -6, -2, -64, 4, 29, -32, 82, -31, 55, -63, 37, -1, 22, 6, -68, -6, 25, -9, 18, 64, -70, 23, -6, -126, 39, -124, 2, -26, 26, -57, -69, -6, -20, -30, -42, -13, 15, -14, -60, 17, -8, -6, 16, 25, -140, -38, 5, -28, -45, 28, -7, -1, -81, -107, 111, -55, 4, 30, 8, 8, 33, -40, 57, 33, 0, 5, -12, 17, -40, 7, 31, 21, 97, -63, 9, 29, 36, -58, -115, 6, 21, 44, 30, 95, 11, 7, -11, 60, -66, 10, 3, -34, 0, -56, -86, 18, -29, 33, -62, 10, -99, -26, -70, 0, -27, 0, -35, 54, -30, 89, 6, 10, 78, -16, 18, 18, -57, -105, -51, -22, -50, 30, -44, 8, 89, 3, -3, 26, -15, 18, -32, -1, 27, -18, 52, 52, -6, -15, 71, 110, 15, 13, 4, 7, -5, 6, -31, 35, 80, -62, 4, -139, -58, 0, -2, 74, -2, 26, 2, 25, -33, -70, -43, -6, -44, 22, -30, 14, -7}
, {-9, -72, -26, 4, -4, -9, -11, -49, -6, -17, 6, 5, -44, 4, -15, -6, 1, -9, -12, -25, -40, -13, -34, -10, -5, -6, -1, -2, -23, -16, -22, -24, 4, 7, -10, -47, 4, -10, 0, -19, -22, -11, -8, -20, -22, -3, -39, -22, 0, -18, 10, -8, -34, -20, -45, -5, -8, -7, 6, -18, -3, -18, -13, -27, -13, -21, -59, -31, -8, 5, -1, -48, -17, -17, 16, -23, -4, -8, -23, -8, -8, -22, 5, -16, -71, -17, -38, -19, -2, 4, -8, 3, -3, -12, 5, -15, -1, -29, -14, -46, -18, 2, -2, -5, -22, 4, -26, 6, -17, 4, -54, -15, -3, -5, -9, -2, -21, -32, -52, -13, -19, -20, 4, 3, -21, 6, 5, -10, -4, -39, -45, -23, -12, 2, -6, -61, -1, -1, -17, 2, -21, -12, -1, -26, 3, -9, -1, -43, 2, -5, -2, -8, -22, -22, -19, -18, 5, -14, 3, -23, -19, 0, -11, -76, -7, 4, -4, 35, -9, -19, 37, -13, -33, -19, -46, -9, -3, 2, -12, -23, -6, -30, -16, -4, -3, -7, -15, -47, -9, -12, -7, -2, 4, -50, -49, 6, 6, -20, -13, -48, -11, -5, -1, -23, -44, -19, -3, -19, -16, -13, 2, -18, -15, -7, -7, 5, -6, -20, -21, -6, -7, -16, -22, -7, -16, -24, -3, -53, -14, -8, -10, -36, -9, -12, -7, -6, -67, -11, -36, -46, -1, 5, 28, -12, -6, -15, -52, -9, -17, 5, -8, -39, 1, -17, -1, -9, -5, -56, -46, 45, -4, -6, -8, -23, 2, -1, 8, -20, -4, -5, -10, -5, -8, -1, -15, -51, -10, -9, -71, -15, -3, 0, -2, -16, -16, -7, -18, -1, -20, -10, -9, 16, 0, 3, -11, -25, -12, -8, -37, -20, -53, 14, -58, -20, 2, -22, -17, -22, -10, 35, -9, -1, 3, 1, -15, 11, -10, -4, -17, 1}
, {18, 20, -64, -12, -57, -9, -45, -66, 105, 3, -13, -42, -8, -50, 30, -104, -19, 70, 17, 8, -10, -70, -30, -1, 11, -11, 0, -1, -16, 17, 61, 3, -48, -91, 61, -15, -56, -55, 24, -33, -10, 5, -15, -178, -30, 8, 12, 27, -10, 17, -3, -54, 43, -86, 47, 54, 51, -3, -8, 4, 21, 62, 17, -216, -7, 28, -16, 35, -31, -1, -15, 19, -23, 4, -7, 15, 7, 37, -50, -201, -18, -37, 1, -14, 13, 1, 38, 11, 0, -41, 19, -11, -2, 93, 99, -51, -11, -60, 119, -50, -6, 6, 16, -4, 29, 23, -118, -84, -40, 4, -28, 16, 3, 16, 40, -2, -12, -102, 49, 102, -8, -77, 17, -80, -44, 27, -2, -123, -1, 71, 44, 20, -88, 11, -25, 30, 17, 13, -7, 7, 20, -39, -45, -5, 22, 23, -6, -53, 10, 12, 35, 2, -7, 65, 9, -2, 9, -49, -36, -30, 26, -78, -40, 11, -33, -41, 0, 8, -21, -10, -24, -113, 8, -3, -64, 3, 67, 17, -40, -63, -65, -123, 0, 46, 12, -28, -9, -40, 10, 64, -1, -29, 14, 28, -37, -12, 16, 14, -128, -82, 78, 1, -6, 22, -3, -10, -3, -46, 13, 29, 13, 21, 21, -35, 19, 0, -1, -23, 0, 27, 25, 64, -36, -58, 50, -65, -15, -28, -61, 44, 13, 10, -10, -3, -154, -84, 15, 1, -54, -7, 49, 3, 13, -87, -43, 12, 74, 30, -29, -86, 6, 1, 10, -58, 18, -32, -1, 9, -55, 23, 36, 8, -106, 14, 48, 21, 6, 37, -2, 73, -14, -59, 16, -24, 1, 13, 18, 0, -24, 1, -2, -7, 12, -27, -6, 44, -57, 1, 33, 62, 87, 48, 6, -58, 16, 57, -16, -2, -66, -117, 4, -8, -29, 27, 65, 17, 90, -105, -39, 3, 79, 35, 19, -17, 6, 22, 48, 0, 11, -81}
, {11, 1, -9, -30, -181, 8, 17, 17, -107, -6, 19, -96, -86, -29, -59, 24, -14, -26, -2, -96, -10, -155, -10, 3, 21, -43, 0, 6, 25, -25, -20, 69, -12, -146, -61, 4, -42, -31, 6, 3, 22, 17, -87, -55, -10, -1, -18, 108, -40, -4, 19, 98, -45, -55, 44, 90, -8, -26, 5, 11, 61, 170, 14, -65, 0, -85, -65, -21, 139, -21, 151, 110, -23, 5, 8, -26, -85, -5, -39, 6, -16, -28, -9, 15, -7, -34, -72, 1, 10, 34, 24, 39, -10, -48, -28, -29, -52, 3, -8, -16, 41, -14, 41, -16, -94, 15, 108, 4, -10, -1, 26, 36, -116, -11, -9, -38, 43, 87, 72, 10, -22, 47, -8, -61, -107, -65, -24, 19, 17, -11, -49, -65, -93, -13, 65, 5, 39, 1, 30, -96, -83, 119, 20, -130, 19, 50, 21, 21, -26, -11, 11, 1, 1, -23, 4, -17, 8, 2, -126, 8, 41, 31, 8, -39, -79, 103, 11, -22, 13, 60, -81, -18, -63, -5, 104, -18, -9, -3, -3, 58, -9, -13, -39, 38, -12, -40, -5, -3, -162, -13, 3, -117, -16, -61, -11, -178, -41, -9, 63, -2, 42, -1, 1, -176, -189, -161, 44, 4, 8, -37, -16, -13, -60, -21, -38, 17, 21, -58, 1, 41, 58, -29, -13, 48, -46, 39, 45, -43, -14, 26, 14, -34, -72, -2, -30, -48, -143, -5, -53, -50, -47, 3, 72, 32, -92, -27, -200, -8, -39, -28, -8, -115, 12, -64, -4, -144, 2, 33, 78, 78, -11, 5, -39, 69, 59, 13, 14, 37, -3, 9, 12, 91, -8, 4, 9, 27, 70, -158, 15, 24, 12, 63, 19, -43, -24, 54, -6, -62, -22, -29, -14, 66, 49, -1, 12, 14, -54, 11, -139, -23, -34, 12, 1, 74, -19, 22, 19, -6, 102, -35, 53, 43, -24, 34, 13, 107, 63, 33, 15, -90}
, {10, 27, 5, -41, 142, 20, 8, -11, 34, -4, -16, 94, 13, 39, -34, 34, -21, -36, 12, -83, 7, 68, -1, 12, -8, 11, 5, -2, 24, -21, 46, -13, -38, -44, -124, 25, -61, -20, 17, 18, -38, -2, -28, 9, -66, 0, -8, -4, -70, 17, -83, -39, -69, 29, -90, -74, -76, 32, -1, -107, 7, -35, -3, 3, 13, 16, -19, -68, -18, 17, 26, -24, -16, 16, -24, -8, 50, -47, -17, -66, 3, -70, 19, -67, 32, -25, 19, -13, 19, 80, 12, 11, 22, 27, -64, -17, 25, 38, 9, -21, 57, 1, -1, 35, -23, 7, -9, 105, -32, -6, 33, 61, -9, 4, 116, -26, -14, 1, -64, -20, -9, 26, 15, -73, 103, -35, -4, 8, 19, -14, -10, -34, -10, -4, 33, -74, -75, -3, 7, -39, 30, 23, -204, -36, -21, -33, 2, -165, -78, -44, -18, 21, -6, -29, -9, -31, 0, -54, 104, -47, 8, 114, -27, -21, 62, -22, 1, -24, -92, 5, -31, 41, -4, 7, -14, 16, -13, 1, -33, -18, 1, 6, -79, -46, -11, -31, 4, -78, -7, -41, -1, -27, 6, 47, -3, 9, 131, 0, -82, 60, 41, -5, 14, 32, 19, -38, -144, -29, 40, -48, 9, -167, 7, 56, 60, 8, 3, 17, -2, 9, -5, 3, -99, 58, 33, -50, -57, 44, -56, 6, 10, -16, 4, 3, -94, -97, 5, -2, -20, 11, 33, 1, -78, -50, 25, 14, -1, 36, -88, -4, -6, -93, -11, 40, 9, -137, 9, -24, 36, -43, -57, 11, 72, 17, 1, 10, -2, 128, 8, 24, -5, -20, 76, -65, 11, -108, 47, -96, -4, 16, -6, 7, 5, 35, 9, 25, -18, 12, 33, 68, 24, -15, 66, 15, -2, -40, 43, 18, -19, 46, 37, -6, -1, 15, 30, 11, 69, -65, 47, 41, -25, 20, 52, 5, 13, 0, 25, 122, -3, -45}
, {2, -1, -9, 24, 24, 18, -8, 46, -1, -2, 7, 12, -85, 6, -55, -29, 25, -25, 3, 32, 34, -63, -6, 3, 11, -6, 8, 16, 25, 26, -7, -108, -17, -92, 62, -16, -2, -86, 7, -26, 3, 35, 47, -44, -9, 16, 41, -3, -6, 6, 24, -64, -17, -150, 16, 1, 34, 28, 22, 62, -6, 56, 18, -30, 0, -25, 5, -69, -39, -4, -111, 7, -83, -5, 2, -24, 8, -65, -119, -27, 39, 9, 12, 19, 46, -81, -48, 7, 11, 23, 22, -50, -9, 54, 87, 48, -23, 32, -88, 26, 21, -39, 7, -29, -22, -53, -4, 20, -8, 3, 11, 28, -38, 21, -113, 19, 6, -30, -96, -33, 10, -22, 7, 96, 40, -68, 11, -1, 5, -36, 3, 114, -41, 11, -95, 77, 1, 17, 21, 91, 7, -38, 59, 30, 11, -9, -3, 7, -42, 2, -22, 5, 21, 8, 11, -7, -9, -39, -3, -30, -25, -44, 38, -30, -17, 10, 3, -54, -42, 25, -6, -29, -12, 26, 15, 25, -10, 15, 54, 54, 5, -75, 20, 51, 23, -28, 17, -34, 14, -20, 19, -93, -6, -62, 60, 30, -16, -6, 39, 23, -18, -5, 3, 131, -61, 28, -44, 51, 12, -66, 8, -70, 7, 0, 2, 19, 24, 6, -1, -18, 12, -20, -40, 21, 2, -72, -7, 26, -130, -18, 5, 4, 38, 13, 56, -13, -70, 15, 71, -6, -5, 17, -91, -2, 32, -60, -59, -31, -6, 29, -4, -5, -79, -71, -11, 42, 5, -68, 86, -84, -83, 6, -53, -37, -16, -1, 19, 101, 24, 103, -7, 54, 1, 35, 16, 36, 14, 51, 5, 0, 3, 77, 21, -22, 16, 38, 181, 77, 28, 67, -3, 5, 43, -29, -4, -32, 5, -1, 40, 96, -64, 5, -15, -2, -46, 15, 26, 117, -6, -54, -199, 13, 0, -8, 7, -67, 30, 149, -1, 70}
, {6, 5, -21, -12, -10, -13, -17, -20, -20, 5, -16, -13, -15, -12, -7, -1, -1, -19, -16, -15, 2, -14, -1, -14, -3, -19, 6, -4, 0, -16, -14, -13, 6, 8, 5, -1, -6, -16, -17, -16, -13, -13, -22, -3, -6, -9, -21, -21, -14, -3, -9, -20, -4, -22, 5, -5, 4, -16, -22, -16, -2, -13, -3, -20, -11, -8, -4, 6, 2, 3, 13, -6, -7, -19, -3, -18, -4, -14, -18, -13, -10, -21, -19, 4, -14, -6, -12, -13, -18, 6, -15, 6, 0, 6, -17, -7, -9, -11, -22, -20, -1, -9, -6, -10, 0, -11, -16, -17, 0, 1, -7, -16, 5, -10, 5, -15, -11, 1, -5, -8, 4, -21, -19, -14, -1, 2, -17, -2, 2, -17, -15, -4, -10, 5, 7, -19, 4, -5, 3, -6, -18, -17, -12, -17, 6, 1, -18, -11, -21, 4, -22, -22, -16, -7, -13, -16, -11, 4, -13, 3, 2, -9, -4, -11, 3, -14, -21, -20, -20, -11, -21, 6, 0, -15, -21, -10, -15, -10, 1, 2, -11, -1, -11, 0, 2, 3, 4, -1, 0, -14, -7, -13, 4, -2, 1, -23, -20, 0, 3, 2, -3, -16, -6, 1, 2, -7, -22, 3, -17, 0, -7, 5, -16, -9, -17, 6, -8, -22, 3, -4, -4, -13, -7, -6, -12, -5, -1, 7, -23, 0, -16, -12, -14, -22, 0, -19, 1, -18, -13, -10, -19, 6, 2, -8, -2, -15, -13, -16, -4, -18, 3, -10, -2, -11, -5, 5, 4, 5, -7, -11, -6, 2, 0, -11, -8, 5, -20, -6, -23, -12, -5, -8, -11, -12, -8, 6, -22, -4, 1, 2, 6, -15, -10, -9, -7, -15, 5, -11, -7, 3, -10, -2, -14, -3, -12, -15, -20, -13, -18, -16, -19, -1, -8, -19, -7, -7, -9, 6, -18, -19, 4, 2, 6, -13, 3, -2, -14, 5, -4, -5}
, {0, -27, -31, -22, -28, 14, 81, 43, -40, 5, 14, 8, 11, 34, 44, 56, -32, -18, 19, -27, -48, -40, -95, 14, 0, 26, 17, 118, -7, 43, 0, 51, -32, 68, -47, -56, -43, 11, -16, -79, -57, -4, 26, 6, -141, -13, -90, -23, 60, -12, 33, -57, -147, -34, -91, -96, -15, -28, -5, 66, -38, 71, 14, 15, 3, -5, 10, 58, -11, 2, -74, 46, 31, 6, 6, -18, -26, 48, 21, 138, 17, 112, -4, -75, 62, -84, 103, 9, 1, -16, -2, 9, 13, 41, 59, -19, -131, 14, 18, -137, 86, -7, -2, 28, -91, -10, -91, 16, 72, 4, 80, -54, 90, 12, -22, 79, -121, 16, -72, -2, 35, -36, 0, -63, 84, -9, 23, -19, 8, -98, -37, -29, 42, 20, 22, 150, -31, -1, 9, 125, 21, -82, 7, 58, 8, -20, 20, 18, -81, -9, 7, 6, 10, 32, 10, -19, -15, 29, 113, -87, 47, -36, -45, -107, 19, 66, -1, 42, -37, 9, 48, 32, -31, 7, -26, -16, 62, -4, -5, -18, 12, 59, -8, -91, 0, 25, 12, 5, 0, -40, 16, 62, 13, 60, 8, -126, 5, -4, -23, 23, 57, -6, 10, 21, -58, -31, 37, 28, -19, -32, -4, 11, 13, 57, 62, -10, 1, 20, 2, 6, 11, -34, 105, -86, -71, 49, 46, -62, -38, -36, 11, -4, -37, -2, 16, 22, -207, 3, 58, -55, 3, 6, 55, 14, -43, 13, 56, -80, -16, -46, 7, -78, 30, 34, 12, 33, 10, 32, 65, 53, 107, -7, -1, 65, -60, -4, 4, 158, -16, -87, -68, -54, -33, -106, -4, 49, 45, -31, -54, -2, -3, 58, 22, 27, 9, 79, -31, -10, 7, 23, 95, -47, -18, 46, -1, -24, -16, 19, 103, 28, -67, -2, 16, -42, 122, 16, 49, 38, 85, 41, 37, -33, -11, -23, 1, -132, 22, -49, 3, 16}
, {-21, -142, -32, -54, 128, 2, 21, -5, -42, -1, -8, -68, -4, 2, 27, 61, -20, 67, -16, 43, -34, -36, -41, -4, -6, -22, -9, 2, 1, 1, 38, 16, 18, -77, -66, 26, -47, -94, -17, -20, -20, -20, -6, -88, 36, 13, -16, -56, 126, -9, -118, 64, 71, -6, -18, -78, 32, -56, 5, -19, 98, -63, -15, -71, 0, 0, -125, -8, 85, 4, 132, -64, -11, -6, -16, 2, 96, -135, 19, -89, -38, 7, -14, -21, -38, -57, 1, -19, -18, 33, -10, 11, -21, -57, -80, 18, 18, -10, 60, -46, -79, -38, -6, -23, -46, -17, 54, 15, 9, -21, -10, -6, -15, -20, -29, -50, 78, 6, 20, -40, 62, 5, -8, 106, -8, -25, -7, -54, 5, 2, 2, 7, 40, -16, 19, -41, 35, 3, -18, -84, 20, -3, -54, 28, -50, 37, -6, -72, 62, -37, 39, -12, 2, -4, -22, 6, 6, -48, -3, 38, 38, -26, 26, 33, 48, -63, -21, 17, -47, -18, -3, -40, -35, -1, 18, -17, 1, -20, -3, 12, -100, 25, -94, 16, 18, -34, 8, 24, 10, -65, 0, -57, -10, 28, -92, 7, 53, 0, -83, -62, -24, 2, -4, -46, 39, 22, 10, -59, -44, 25, -13, 0, 41, 63, -28, -8, -2, 0, -3, -23, -10, 44, 16, 45, 31, 15, -73, -78, 72, 7, 2, 15, -29, -8, -27, -138, -14, 0, -51, -9, 26, 7, -20, -29, 19, -45, -60, -14, 4, -23, -9, 40, -69, 37, 7, -134, -10, -4, 44, -133, -23, 0, -38, 56, 56, -1, 3, -8, 8, -18, 8, -8, -9, 45, -5, 12, 3, -19, -5, 0, -5, 53, 2, 15, 4, -17, 76, -62, -15, -40, -16, 21, 27, -70, -17, 50, -41, -13, -100, -94, -72, 9, -48, 11, -4, 5, -11, 85, -1, -54, -1, 10, -24, -20, 2, -2, 69, -34, -2, -24}
, {-9, 56, -147, 0, -73, -8, -90, 25, -34, -5, -8, -36, 79, -96, -95, -60, 7, -97, 22, -96, -73, 50, -88, 3, 6, -81, 6, 50, 11, 10, -169, -4, 32, 38, -69, -17, -108, 47, 6, -4, -159, -4, 14, 40, -161, 2, 2, -141, 100, 13, -51, 33, -39, -31, -18, -62, -27, -18, 5, 52, -78, -10, 4, 46, -7, -10, -114, 22, 5, 3, -61, 65, 47, 2, -19, 31, 5, -8, 103, -8, -18, 98, -4, -62, -94, 86, 33, 0, -15, 9, 4, -44, -10, 129, -57, 27, -131, -12, -8, -121, -31, -12, 4, -10, -53, 50, 46, 9, -78, -23, 55, -88, 21, 18, -45, -29, -14, 73, -54, -12, 8, 2, 4, 6, 81, 75, 17, 62, 2, 29, -65, -10, -53, 4, -124, -37, 30, -4, -8, -48, 14, -12, 72, -123, 25, 41, 3, -32, -38, 68, 67, -2, -5, 72, -23, 108, -15, 6, 52, 12, -99, 56, -213, -60, 98, 42, 6, -63, -136, 12, 66, 20, -37, -15, 34, 25, -46, 2, -6, -11, -3, 79, 47, -130, -22, 44, -2, -38, 38, 77, 4, 10, 0, 0, -94, -70, 8, 0, 8, -22, 135, 17, 3, 22, -22, 115, 75, -67, -43, 25, 2, 2, -6, 41, -69, 14, 1, 74, -6, 4, 26, -21, 105, 5, -42, 2, 144, 3, 146, 13, 14, 40, -109, 20, -40, 8, -13, 7, 28, -54, -56, 16, -96, 10, -12, -8, 18, 41, 65, -4, 14, 5, -29, -65, 8, 21, -7, 110, -116, -10, 33, 10, 19, -138, 18, 2, -5, -7, -36, -40, 20, -42, -30, -29, -4, 67, -73, 43, 6, 0, 5, -45, -1, 12, -9, -97, 74, 21, 43, -9, -104, -110, -44, 56, 6, 34, -3, 18, -33, -21, -48, 3, 88, 29, -111, 1, -159, -8, -10, -10, 32, -42, -16, -38, 3, -73, 26, -14, 23, -19}
, {-20, 3, 6, -3, 2, -6, 11, 6, 11, -2, -8, 6, -11, -19, 10, -3, 1, -32, -17, 3, -9, 2, -16, -20, -3, 3, 5, -6, -4, -30, 3, -11, -10, 1, -4, -20, -1, 1, 6, -7, -18, -8, 2, 9, -32, -3, -41, -24, -5, -3, -3, -20, -4, -10, -29, -26, -24, -19, -1, -11, -4, -7, 14, -19, 9, -10, 16, -15, 18, 6, 0, -18, -22, -18, -11, 9, -13, 10, 12, -1, 13, -15, 17, -34, -37, -4, 8, 22, -1, 1, -2, -3, 16, -6, -8, -6, 10, -5, 18, -17, -6, -13, -9, -23, -20, 0, -18, -23, -14, 4, -2, 2, -3, -1, 16, 4, -5, -22, 10, -24, -13, 7, 8, -15, -7, -16, 4, 5, 2, -1, -16, -16, -6, 21, -11, -30, -32, -12, 10, 11, -34, 7, 8, -2, 15, -23, 5, -17, -37, -3, -15, -3, 13, -34, 2, -2, 22, 3, -11, -7, -14, -19, 18, -11, -4, -4, -13, -29, -9, 14, -19, -12, -6, -16, -19, -30, -6, -4, 18, -2, -19, -2, -25, 7, -11, 21, 20, -14, -6, 4, 18, 4, -9, 24, -7, -21, -3, -15, -17, -27, -18, -1, -7, -1, -15, -16, 3, 6, -8, -4, -1, 3, -21, -19, 2, -5, 3, 2, -4, 0, -22, -37, -10, -11, -6, -15, -1, -24, -10, -2, -9, 2, -17, 6, -23, -25, -10, 0, -41, -20, -6, 1, 15, -18, -15, 7, -26, -15, -17, -4, -5, -10, -1, -22, 10, -2, 21, -41, -14, -9, -8, -4, -5, 1, -13, -4, 12, 10, -17, 2, 3, 4, -13, -8, 2, 18, -37, 1, -6, -7, 8, -15, -9, -16, -16, -17, -6, 17, -24, -14, -20, -9, -14, -7, -10, -17, -19, -1, -24, -27, 1, -23, -35, -39, 17, 4, 5, -25, -5, -1, 25, -2, -2, -2, -11, -20, -15, 5, -10, 7}
, {10, -18, -23, -18, -3, 19, -12, -24, 17, 7, 2, -24, -9, 15, -7, -20, -13, -19, -5, -33, -19, -22, 5, -7, -7, 7, 1, -14, 3, -24, 0, -16, -13, -12, 2, -26, -17, -20, -1, -32, -39, 10, -23, -7, -13, -4, -17, -3, 5, 19, 0, 2, -8, -10, -20, 9, -11, 16, 20, -20, 18, 15, 7, -2, -2, 6, -6, -1, 4, 12, -12, -8, -14, 21, -5, 3, -9, -17, -4, -11, 19, -35, 4, -17, -19, 6, -15, -8, 0, -2, 2, -6, 12, -30, 8, 7, -9, -24, 2, -17, 0, 19, -1, -23, 20, 6, -8, 8, -1, 2, -19, -14, -13, 17, -4, -16, -5, -27, -14, -12, -3, -8, -3, 4, -1, -2, -7, -16, -5, -17, -23, -8, -48, -21, 5, -8, -19, 17, -9, -16, -1, -1, -8, -21, 4, -17, -23, -21, 4, 0, -20, 0, -4, -19, -3, -4, -17, -29, -17, -7, 3, -47, -9, -15, 3, -13, -2, -31, -18, -9, -12, 0, -3, -8, -34, -11, -21, 11, -9, -5, -32, -9, -25, -15, 19, -1, -3, 12, -22, -6, -1, 3, 2, -18, 3, -15, -26, -4, 0, -16, -4, 2, -21, -3, -30, -4, -5, -14, -10, -9, -12, -9, -10, -1, -1, 1, 0, -10, -10, 6, -13, -24, -10, -21, 3, -33, -9, 11, -23, -10, 2, -7, 4, -1, -11, -13, -13, -17, -23, -5, -1, -8, -17, -9, -12, -10, -23, -16, -20, -15, 3, -11, -5, 1, -18, -24, -11, 0, -11, -18, 0, -11, 0, -7, -16, 4, -14, -15, -12, -22, -36, -3, -13, -1, -2, -15, -22, -8, -2, -7, -5, 0, 0, 4, -10, -11, -3, 3, -2, -9, 1, -23, -10, 4, 21, -15, -22, -22, -24, -35, -17, -12, -21, -13, -11, -1, -4, -17, -22, -13, -22, -18, -3, -4, -19, -3, 0, 2, -14, -1}
, {-5, 4, -9, -4, -16, -23, -2, -7, -21, -8, 1, -4, -6, -22, -8, 2, -6, -8, -20, -10, -21, -6, -5, 2, -17, -2, -14, -7, 5, -3, -8, -5, -1, -26, -13, 2, 0, -22, -6, -15, -20, -5, -3, -15, -1, -22, -13, -18, -3, 4, 4, -5, 3, 1, -11, -7, -3, -8, -1, -21, -21, -15, -13, -6, -6, 7, -18, -13, -10, -2, 7, -11, 1, -7, -8, -4, -2, -12, -1, -6, -17, -22, 1, -18, 7, -22, -19, -1, -11, 0, -10, 2, -9, 0, -13, -4, -7, -22, -7, 17, 2, -15, -2, -10, -11, 2, -12, -16, -10, -18, -8, 2, -18, -10, 0, 4, -3, -24, -6, -22, -22, -4, -16, 1, -17, 6, 1, -8, -13, -13, -3, -11, -2, -17, 0, -14, 1, 0, -6, -12, -12, -2, -13, 2, -2, -19, 5, 7, 2, 2, 1, -17, -1, -21, 3, -9, -20, 0, -4, -23, 0, -3, -5, -7, -6, -16, -7, -22, 3, -8, -9, -10, -19, -8, -2, 4, -17, -13, 3, -11, 5, -21, -11, -12, 1, -11, -22, -3, -7, 0, -21, -17, -8, -13, 0, -2, -12, 3, -10, 6, -15, -2, 5, -7, 2, 4, -9, -10, -21, -20, -10, 3, -25, -11, 2, 5, -23, 3, -8, -13, 3, 0, 3, -23, 6, -22, -7, -9, -12, 5, -1, -11, -3, 0, -19, -7, -11, 1, -23, -16, 4, -23, 7, -5, -11, -3, -3, -21, -15, 0, -21, -7, -11, -10, -16, -13, -1, -20, -22, 16, 5, -19, 1, -4, -3, 2, 6, 2, -21, 1, -14, -3, 1, -10, 1, -2, 0, -15, -13, -21, -11, -8, -3, -3, 1, -21, -17, 5, -4, -9, 5, 16, 1, -23, -21, 1, -7, -6, 3, -8, -17, -7, -20, 6, -5, 0, -22, -10, -18, -10, -2, -17, -21, -9, -12, 1, -15, -6, -23, -4}
, {-14, -17, -10, -16, -4, -2, -11, -12, -15, -17, -3, -19, -15, -5, -17, -14, 6, -12, -1, -2, 3, -13, -14, -15, -14, 6, -13, -5, 2, -22, -12, -19, -16, -16, -21, -6, -15, -21, -18, -1, -13, -5, -21, -16, -2, -13, -12, -20, -22, 5, -14, -4, -2, -22, -10, -10, -16, -4, -14, -16, 0, -16, -4, 1, -3, -8, -18, -21, 2, -11, -6, -2, -2, -21, -20, 3, -22, 5, -12, -1, 1, 6, 5, -16, -23, -11, -22, -7, -21, -17, -18, -2, 4, 0, -7, -10, -16, -13, 5, 2, 4, -19, -1, 3, 4, -20, -2, -2, -16, 5, -7, 5, -14, 6, -5, -16, -21, -3, 4, -8, -18, 1, -18, 5, -11, -16, -16, -7, -5, 4, -18, -13, -14, -16, 1, -5, -11, 0, 3, 2, -7, -1, -2, -19, -11, 3, -15, -18, -21, -10, -18, -11, -15, -7, -12, 2, 1, -11, -6, -17, 1, -22, -17, -16, 4, -13, -4, -5, -16, -2, -2, -5, 2, -8, -13, -12, -23, -6, -10, 6, -4, -18, -16, -11, -18, 3, -22, 4, -12, -9, -14, 5, -18, 6, -7, -1, -12, -16, -5, -4, -5, -1, 4, -5, -9, -21, -23, -17, -15, -3, -18, -16, -5, -19, -20, -1, -8, -16, 3, -18, -6, -23, 5, -9, -5, -22, -6, -10, 3, -11, -15, -12, -19, -9, 3, -4, -10, -19, -2, -9, -14, 1, 7, 2, -7, -7, -3, -5, 0, -20, -8, -17, 3, 5, -14, -15, 2, -18, -6, -11, -4, -5, -3, 9, 1, 1, -15, -19, 4, -8, 4, -23, 6, -2, 0, -18, -18, -19, -2, -1, -7, -1, -14, 0, -7, -8, -4, -9, -17, -15, -17, 6, 0, -1, 2, 2, -3, -12, -18, 5, -14, 6, 1, -17, -20, 5, 4, -8, -7, -2, -6, -16, -14, -18, -12, -20, 2, -1, -10, -5}
, {-6, -116, -50, 28, 22, 19, -5, -20, -60, -2, 5, 25, 80, -96, -26, -25, -6, -25, 14, -27, -72, 32, -69, 2, 3, -85, -9, -60, -1, -10, 31, -72, -57, 7, -64, -59, -90, 11, -7, 21, 12, 54, -20, -8, 26, 2, -67, 11, -100, -3, 34, -43, 35, 6, 26, -41, 78, -34, 0, 14, -6, 87, -35, -23, 16, 21, 7, -15, -170, 6, -74, 30, -4, 20, -12, 67, -24, 35, 114, -52, 63, 37, -2, 30, -4, -1, -45, 5, 4, 40, -11, -84, 22, 34, 53, 0, -82, 47, 90, -94, 3, -15, 13, -74, -47, 57, -112, 117, -91, -18, 32, 7, -143, 21, -41, -38, -33, -26, -116, -63, -18, -77, -22, -37, -5, -34, 5, 154, 0, 30, 79, -54, -42, 7, 19, -16, 13, -32, -5, -42, -19, -33, -19, 55, 8, -50, 17, -19, 88, 156, 30, 5, -16, 67, -14, 58, 16, 30, -24, 34, 18, -43, 46, 37, 48, -34, -10, 31, -41, 114, 39, 51, -36, 3, 10, 28, -41, 11, 13, -114, -72, -15, 5, 69, -108, -5, 2, -114, -57, 2, -36, 23, 13, 10, 23, 75, 47, 31, -73, 69, -19, -7, -18, 38, -20, -18, 30, 26, -19, -18, 23, 13, -8, 18, -7, -13, -11, -73, 9, -20, 95, -6, -33, 2, 38, -120, 7, 36, -17, -14, -19, 2, -12, 49, -28, 21, 13, 0, 24, -18, 138, 19, 6, -18, 38, -51, 6, 76, -39, 45, 25, -37, -23, -67, 11, 9, -15, -64, -56, 21, -81, 10, -33, -21, -24, 20, -8, -67, 31, 27, 12, -10, -2, 23, 0, -56, -29, 4, -86, 20, -14, -57, 8, -71, 53, -27, 16, 91, -33, -12, -66, -62, -33, -27, -21, -27, 49, 8, -51, -17, 53, -2, 36, -20, -22, 21, -61, -31, 6, 3, -58, 77, 9, 41, -7, 47, 97, -41, -11, -45}
, {-15, 4, -1, -2, -20, -18, 19, -14, -16, -15, 9, -14, -10, -17, -19, -16, -11, 7, -14, -23, -6, -11, -18, 5, -13, -12, -19, -13, -3, -5, -16, -3, -2, -35, -4, -28, -13, -20, -8, -6, 2, -8, 4, -3, -17, -17, -25, -6, 3, 4, 8, -10, -5, -3, -5, 3, 3, 0, -5, -8, 3, -19, -1, -6, -3, 3, -19, -14, 8, 7, -5, -26, -16, 1, 0, -11, -3, -13, 6, -1, -21, -13, -2, -3, -17, 1, 2, -1, -19, -16, -21, -15, 6, -10, -2, -11, -15, -20, -17, -18, 1, 1, -20, -25, -9, -18, -21, 1, -18, -17, -28, -8, 2, 2, -11, -7, -22, -3, -21, 1, -2, -6, 5, -9, -19, -23, -2, 1, 4, -4, -17, -40, -9, 6, -15, -3, -2, 2, -2, 3, 4, -18, -1, -12, -3, -21, -17, -9, -7, 0, -13, -15, -3, -4, 2, -4, -8, -20, -12, -7, -10, -18, 5, 4, -15, -3, -3, -17, -6, -1, 1, 2, -5, -19, -20, 2, 1, -23, 4, -15, -2, -21, 9, -12, 5, -2, 3, -21, -8, 3, -16, -1, -16, 6, -9, -8, -14, -13, -1, -24, -14, 0, 11, -9, -23, 2, -7, -11, 5, -22, -17, -17, -4, 1, -1, 3, -10, 6, 3, -1, -11, -16, 4, 6, -10, -8, 4, -3, -3, -16, 4, -22, 5, -4, -17, -16, -19, -18, -23, -7, 6, -8, 18, -9, -24, -12, -8, -4, -1, 4, -8, 13, 3, -6, -1, 2, -13, -8, -5, 1, -3, -6, 4, -30, 3, -6, -9, -4, -18, 6, -18, 0, 0, -13, -4, -1, 18, -21, 6, -6, -3, -11, 6, -11, -21, 1, -21, -14, -21, -26, -16, -11, -22, -1, 3, -36, -23, -5, -4, -10, -3, 3, -10, -18, -21, -18, -1, -5, -12, -23, -23, -9, -20, -6, -20, -4, -2, -20, -23, 1}
, {0, -43, -9, 82, 68, -9, 33, 0, -56, -8, 5, 2, 40, -66, -199, 28, -44, 68, -17, 2, 15, -139, 46, 7, -6, 73, 0, 24, 25, -39, -32, -58, 18, -93, -90, 38, -74, 25, -14, 30, 19, -2, -51, -185, 25, 7, -58, 55, 40, 6, 65, -35, -12, -122, -109, 45, 30, -30, -21, 22, 53, 37, 5, -192, -9, 28, -5, -54, 78, 3, 54, -6, 8, -16, 20, 23, -3, -16, -51, 5, -8, -74, -14, -60, 84, -16, 26, -23, -15, -52, -12, 4, 30, 25, 52, -35, -67, 129, -49, 17, -64, 23, -13, 5, 2, -14, 35, -151, 13, -4, 125, -9, -8, -17, 23, -31, 27, 137, 82, -38, 56, 43, 1, 75, -61, -107, -3, -72, -2, -31, 28, -27, 35, -1, 36, 13, -27, -7, -12, 106, 48, -53, -102, -5, 11, -63, -6, -64, 35, 13, -8, -1, -31, -81, 18, -14, 47, 33, 144, -1, -1, 29, 37, 16, 26, -24, 1, -3, -5, -27, -31, -161, -64, 4, 79, 49, -47, -21, 96, -101, 65, -71, -99, -124, 26, 14, -11, -44, -205, 147, 26, -112, -15, -57, -24, -38, 48, -8, 178, -61, -30, -1, 8, 25, 14, 95, -43, 41, -29, -34, 2, -117, 52, -22, -95, -8, -20, -85, -21, -16, -3, -68, 22, 11, -20, -83, -57, -33, 43, 64, -20, -61, -88, -9, 110, -61, -103, -13, 13, 56, -49, 4, -149, -112, -43, -15, -126, -17, 64, -61, 5, 13, 139, 54, 5, -171, -17, 37, -2, 66, -13, -2, -43, -31, 1, 6, 11, 13, -52, 103, -83, 57, -22, -71, 6, -83, 90, -47, 4, -8, 4, -104, -18, -30, -20, -90, -60, 65, 52, 3, 23, 5, 46, -68, -20, -15, 44, -7, 143, -35, -7, -10, 34, 43, 31, -16, -38, 40, -116, -212, -33, 1, 61, -51, -19, 75, 73, -71, -11, -97}
, {-18, -58, -59, -14, -50, 4, -15, 19, -58, 2, 1, 1, -40, -31, 3, -35, 7, -66, -2, -131, -32, -47, -18, 3, -19, -36, -13, -11, 4, -27, -16, 19, -28, -95, -18, -57, -36, -28, 1, -42, -53, -10, -95, -14, -35, -16, -35, -8, -33, 1, -91, -30, -22, -59, -20, -31, -38, -19, -13, 46, -2, -8, -20, -36, -15, -94, -80, 6, -75, 1, -5, -97, -113, -12, -16, -22, -33, -6, -32, -111, -5, -40, 0, -58, -23, -25, -80, -18, -11, -141, -5, -17, -18, -59, -4, -48, 2, -59, -74, -107, -30, -70, -6, -49, -101, -19, -48, 18, -35, -13, -44, -19, -54, -4, -95, 2, -40, -51, -50, -36, -58, 3, 0, 25, -41, -3, -5, 26, -4, 1, -51, 29, -81, -20, -14, -81, -35, -3, -16, 31, -9, -16, -49, -76, 1, -41, 3, -107, 6, -11, -71, -11, 6, -10, 6, -1, -1, -82, -1, -22, -34, -62, 13, -68, 28, -60, -11, -55, -104, -7, -49, -7, -57, -18, -32, -47, -39, -2, -34, -12, -80, -99, -26, -45, -14, -4, 4, 27, -14, -19, -11, 14, -4, 7, 20, -17, -91, -15, 4, 23, -39, 1, 7, 14, -18, -8, 8, -30, 0, -8, -21, -70, 51, -85, -81, -1, -22, -15, 1, -13, -23, -18, -7, 18, -1, -53, -42, 13, -32, -86, -6, -39, -3, -17, -16, 6, -17, -7, -34, -37, -52, -20, -37, 11, -6, -52, 2, -23, 30, -17, 3, -31, -32, -17, 6, 33, 1, -68, 8, -45, -98, -20, 6, -12, -62, 4, -2, 87, -24, 3, -30, -46, 3, -52, -10, -52, 21, -4, -36, -9, -12, -103, -21, -22, -10, -59, 5, -49, 9, -49, -55, -42, -53, -27, -10, -76, -26, -6, -41, 6, 7, -15, -132, -67, -29, -8, -57, -13, -24, -1, -17, 45, 0, -16, -3, 22, -15, -5, -16, 15}
, {0, -49, 57, 0, 30, 5, -48, -63, 10, -7, -21, 81, -52, -35, -2, -5, -19, -59, 20, -15, 56, -36, -61, 1, 19, -24, 19, -93, 11, 61, 62, -10, 4, -36, 5, 25, -19, -131, 12, -2, 29, -12, 86, -163, 18, 11, 19, -3, -27, -10, -34, 25, -64, -96, -23, -5, 71, 5, 15, 21, -20, -13, 20, -57, -10, -46, 4, -9, 17, 14, 29, 0, -82, 1, 41, -149, 1, 58, 32, 60, -34, 70, -9, -18, -25, -37, -99, 4, 6, -4, 5, -2, -10, -60, 52, 46, 60, 48, 61, 7, -2, -70, -18, 51, -26, 41, 136, 50, 35, -5, 7, 1, -8, -25, 48, -25, 53, 185, 7, -64, 53, 8, -2, 37, 46, -61, -31, 57, 21, -15, -56, 54, -1, 1, -107, -20, 25, 15, 1, 11, 18, 53, 79, -8, 41, 31, -7, 18, -1, -11, -19, 4, 5, -32, 13, -7, -6, -46, -12, -10, 4, -29, 95, -28, 35, -71, 5, -57, -44, 24, -12, -76, -46, 14, 47, -36, 8, -2, 68, -99, -4, -33, -10, 11, 46, -44, 3, 67, -61, 43, -7, -125, -2, 15, -20, -31, -9, -3, 9, -88, -3, 20, 8, 23, 8, -70, -8, -57, 15, -45, 19, -25, 40, 25, -12, 12, -8, 34, 13, 72, -3, 30, -84, 2, -24, 38, 74, -10, 17, -48, 15, 21, -19, 11, 40, -52, -70, -7, 43, 53, 93, 1, 111, -28, 54, -37, -83, 92, 34, 0, 8, 41, 0, -33, 8, -91, -7, 24, 34, -144, -51, 11, 20, -49, 39, 16, 0, -13, 23, 38, 103, -61, -7, -20, -5, -87, 40, 71, -13, 10, 14, 40, 12, -13, 0, 30, 101, 72, 64, -9, 13, -7, 27, -96, 14, 32, -36, 17, -32, -10, -21, 15, 12, 76, 21, -2, 41, 28, -43, -14, -1, -87, 12, 28, 14, -31, 33, 160, -1, -48}
, {15, 3, -5, -17, -11, 5, 14, -3, 2, 18, -3, -21, -2, 16, -16, -12, 22, -31, 15, -8, -8, 4, 4, 2, 2, -24, -20, -16, 2, -15, 13, -13, -15, -4, -2, -16, 13, 11, 18, -13, -25, -6, -4, -4, -19, -6, -5, -7, 1, 9, -13, 0, 14, -31, -8, 2, -11, 14, 19, -32, 13, -23, 7, 4, -9, 1, -9, -16, 6, -8, -14, -4, -7, 6, -4, -22, -1, 2, 2, 7, -21, 1, 0, -13, -1, 12, -11, 1, -18, -1, -19, -13, -7, -12, -15, 13, -11, -20, -5, -19, -16, -17, 17, -23, -11, 5, -11, 7, -19, 0, 3, -21, -8, -6, -19, -26, -7, -26, -15, -9, -21, -14, 5, -12, 1, 3, -23, -9, 5, -24, 3, -4, 1, -11, 12, -13, -20, -16, -6, -1, 0, -1, -3, -21, 3, -2, 4, -18, -5, -4, 3, 6, -16, -12, 6, 1, -17, -20, -6, -1, 1, -39, -12, -8, -16, -29, 10, -22, 5, 14, 6, -5, -3, 1, -33, -18, 4, 0, -10, -7, 4, -13, 0, -20, -5, -9, -8, 16, 4, 5, -10, -1, 16, -2, -17, 7, -9, -1, -4, -13, 0, 11, -9, -16, -8, -15, 4, -12, -20, -11, -20, -10, -29, 6, -18, -17, 1, -12, 5, -21, -15, 3, -23, 9, -20, -24, -25, 3, -21, -3, -8, -30, -17, -2, 3, -2, -3, 1, -10, -22, 19, 2, -8, -15, -17, 3, -9, 12, -11, 4, -8, -15, -20, -18, 4, -11, 20, -8, -15, -10, 2, 18, -12, -8, 4, -16, -8, -22, 0, -1, 1, -3, -12, -26, -14, 3, -20, 10, -5, 5, -7, 3, -9, -5, -13, -9, 1, -1, -22, -47, -9, 8, 1, -19, 11, -17, -2, -3, -16, 9, -1, -12, -35, 4, 0, 0, -23, -4, -5, 8, 11, -9, 2, -1, -4, -4, -16, -3, 4, -7}
, {3, -109, 36, 59, -115, 13, -43, -113, 39, -6, 11, -63, -10, -84, 25, -53, 26, 22, -7, 43, -136, 34, -78, -1, 14, 85, 2, 12, 36, 66, -30, 44, 69, 122, -263, -22, 14, 0, 13, -35, -54, 11, 33, 61, 50, 7, 23, 59, -102, 19, 66, 25, -20, 16, 26, 13, 10, -15, 13, 73, 5, -34, 4, 52, 15, -128, 67, 26, 148, -3, 2, 109, -2, 6, -4, -40, -112, -39, -1, 102, 8, 9, -4, -40, -59, -12, -61, 16, 13, -19, 11, -33, 9, -3, 12, -111, -74, -9, -7, -35, -13, 15, 0, -91, -58, 16, -22, 72, -65, 4, 56, -145, -40, 19, 42, -53, -15, -14, 0, 90, -49, 17, 6, -117, 2, 165, 2, 85, 17, 49, 18, -119, 27, 5, 29, 17, 69, 5, -15, -27, 26, 12, -43, 20, 10, -38, 11, 2, -38, 115, -67, 0, 19, 0, 11, -12, -4, -24, -185, 31, -55, -40, -135, -11, -42, -19, 15, -158, -55, -24, -64, 15, -131, -4, 49, -27, 26, 6, -276, 34, -48, 36, -88, -123, 32, -26, -7, -217, -1, -22, 11, -29, 10, -81, -14, 18, 49, 21, -111, 97, -46, 19, -2, -43, -24, 62, 46, 65, 12, -12, 19, 87, 9, -71, -178, 19, 5, -1, 9, -16, 19, 13, 69, 73, -9, -55, -73, -66, 3, 35, 17, -49, 12, 5, -2, -53, -8, 18, 7, -33, 21, 12, -296, -91, 57, -111, -87, -34, 23, -5, 17, 95, -128, 91, 1, -53, -4, -42, -40, -56, 157, 21, -12, -44, -47, 5, 4, 101, -31, 21, 47, 67, 24, 34, 0, 15, 62, 95, 63, 10, 13, -86, 2, 0, 10, -21, -23, -98, -100, -100, -36, 2, -29, -7, 18, -83, -26, 17, 0, 49, -87, 0, 9, -42, 74, 17, -209, 19, -7, 95, -18, 4, -2, 31, 9, 137, 54, 55, -6, 65}
, {-13, -21, -4, -18, -8, -6, -7, -17, -19, 5, 4, -7, -8, -9, 5, -9, -14, 6, -2, -3, -9, 4, -1, -12, -3, -14, -6, -11, -19, -17, 0, 4, -18, 10, -3, 6, -8, -9, 0, -16, -18, -11, -15, 2, 0, 5, -10, -3, -18, 4, 4, -15, -10, -20, 1, -22, -11, -19, -20, 7, -3, -18, -11, -9, -5, -13, -11, -1, -13, -9, -18, -2, 1, 5, 4, 4, -15, -9, -6, -2, -21, -7, -1, -5, 4, 5, -7, -10, -2, 4, -18, -3, -4, -5, -18, 0, -9, -1, -19, 8, 4, -15, -19, -14, -18, 6, -11, -6, -12, -15, -16, -10, -1, -15, -6, -18, -10, 6, -13, 6, -20, -15, -6, 3, -15, 5, 0, -11, -21, -22, 1, -10, -6, -2, -15, -13, 4, 1, -18, -10, -16, 4, 0, -11, -9, -1, -1, -15, -4, 6, -19, -12, -14, -12, -20, -10, 1, -19, -5, 1, -14, -7, -23, -10, 5, -15, -6, 5, -10, 4, -11, 5, -9, 7, -15, -5, -4, 3, 1, 5, -3, 1, 14, -14, -13, -7, 0, -10, -8, -10, -1, -18, -17, -8, 0, -11, -7, -9, 3, -9, -15, 3, -21, -23, -16, 5, -7, 5, 3, -4, 0, -3, 8, -6, 3, -14, -3, 0, 3, 6, -1, -9, -5, -21, -2, 3, -5, -12, 5, 1, -10, -19, -1, -11, -15, -9, -5, -10, -6, -3, -2, -9, -10, -2, 1, -13, 1, -3, -9, -15, 3, -13, 1, 2, -18, -3, -8, -14, 4, 8, -14, -15, -18, -2, 1, -4, -11, 1, -5, -20, 0, 0, -5, -10, -9, -12, -23, -16, 3, 5, 0, 0, -10, -2, -21, -11, -18, 2, -13, 0, 2, 8, -5, -3, -1, -22, -23, 6, -3, 3, -5, -19, 6, -9, -3, -7, -16, -2, -3, -18, 1, 2, -8, -9, 3, -15, -8, -18, -6, 5}
, {20, -112, -63, -26, 21, 3, -5, -31, -53, -3, 12, 0, -33, 6, 42, -15, 4, -66, -8, -24, -72, 21, -7, -1, 5, -82, 8, -8, -8, -39, 47, -26, -84, -7, 154, -28, 24, 12, 15, -80, -81, 18, -1, -18, -60, 4, -72, -41, 25, 12, -12, 41, -25, -77, -21, -92, -50, -37, 9, 36, -42, 22, 19, -28, 12, 49, 16, -39, -76, 0, 0, -78, -72, -6, -6, -49, -70, 3, 17, -28, 38, -54, 4, -46, -56, -80, -29, -3, 2, -132, 8, -10, 8, 10, 64, -38, -44, -10, -34, 13, -1, -50, -11, -59, -44, -2, -124, -2, -44, 4, -183, -68, -2, -2, 17, 12, 15, 45, -90, -32, -63, -29, 10, 2, 43, 6, 5, -6, 15, -87, -39, 3, -90, 1, -2, 37, -41, 4, 17, -59, -20, -30, 3, 4, -20, -55, 0, -56, -64, -48, -107, -11, 6, -84, 12, 11, 4, -42, 46, -38, -13, -54, 11, -104, -21, -72, -4, -60, -68, 17, -129, 48, -41, 9, -105, -60, -18, 3, -58, -23, -14, 29, -41, -88, -75, -60, 2, -6, -28, 1, 4, 41, 17, 60, -38, 34, -44, -17, -4, 32, -20, 9, 9, 13, 5, 29, 3, 11, -14, -19, 13, 43, -69, -52, -50, -6, -11, -29, 8, -17, 1, 7, -21, -22, -24, -26, 13, 40, -47, -70, 15, 24, 36, -7, -58, -14, 8, -15, -33, -78, 16, -12, 26, 18, 64, -53, -97, -21, -11, 4, -14, 60, -32, 0, 19, -7, 15, -49, -44, -42, -61, -13, 5, -81, -107, 17, 18, 11, -12, -47, -1, -50, 18, -36, 0, -26, -90, -39, -6, -15, -2, -46, -7, 10, -15, -45, -15, -28, -35, -92, -39, -88, -58, -26, -6, -35, -41, -5, -58, -33, -57, -2, -120, -20, -15, 0, 40, 23, -5, -61, -5, -41, -100, -30, 7, 46, -15, 16, 17, -45}
, {10, -10, -58, 7, 89, 7, -40, 3, 41, 21, 19, 37, 53, -10, -124, -16, -24, -12, 16, -59, -2, 93, -5, 18, -2, -77, 5, -20, -25, 32, -7, 3, 38, -116, 172, -21, -53, -36, 17, 29, -5, 20, -5, 5, 25, 17, 63, 20, -74, 18, 107, 30, -14, 166, -64, -15, 24, 14, 18, -42, 18, 8, 11, 61, 17, -49, 34, 28, -9, -1, 60, -25, -56, -2, 16, 81, 4, 21, 13, -5, -68, -62, 2, -43, 55, 20, -5, 9, -5, -15, 4, -43, 25, -106, 48, 61, 29, -30, 10, 35, 55, -60, 0, 38, 17, 2, -45, -8, -19, 18, -8, 43, 24, 11, 83, 71, 69, -8, -51, -39, -21, 59, 10, 12, 53, 33, 1, -46, -2, -37, 23, -34, 29, 9, -49, -41, -28, 8, -7, -31, 33, -26, 149, -22, -19, -10, 18, 67, 26, 11, -48, 10, 8, -110, 4, -30, 4, -24, -127, -84, -30, -56, 12, 11, 20, 45, 16, -24, -89, 17, -19, -8, -19, 14, -24, 38, -7, 0, 4, 31, 94, -31, -3, 37, 30, -4, 2, -79, 28, -54, 9, -59, 15, 38, -42, -34, -43, -1, 75, -22, 15, 14, 32, 44, 13, 46, -87, -79, -18, -52, 2, -26, 37, 2, -14, 8, 20, 37, 20, 14, -5, 45, -9, -21, 52, -88, -5, -9, 48, 52, -1, 16, 13, 11, -29, -10, -11, 17, -20, 67, 93, 13, -19, 16, -1, 55, 11, 57, 16, 57, 6, -15, -36, -103, 8, 9, -4, 34, 0, 0, 27, -2, -23, -151, 113, -6, 7, 43, 18, 18, 66, -46, -30, -66, 9, 29, 48, -37, -30, 3, 20, 23, 3, -43, -14, -20, -66, 1, -10, 19, -97, -36, 17, -136, 12, -31, -6, -10, -128, -20, 9, 15, 0, 62, -54, 14, -205, 12, -28, -114, 51, -6, -20, 23, -3, -132, -68, 1, 3, -27}
, {-12, -6, -92, 40, 45, -21, -28, -70, -186, -5, 1, 53, 2, -27, -121, -118, -14, -8, -3, -43, 12, -50, 50, 1, 5, -192, -2, 27, 24, -57, 82, 28, -42, -130, -121, -59, -124, -88, -2, -33, 16, 0, 49, 65, -67, 1, -29, -30, 52, -16, -75, 74, -86, 44, -102, -5, -15, -2, -12, -59, 53, 18, -8, 64, -1, -17, -7, -84, -72, -14, -1, -111, 94, -8, 4, -16, -75, -133, 71, 33, 8, -40, -12, -19, 30, 135, 8, 33, 3, 12, -13, -2, 10, 23, -88, 176, -54, 54, -57, 18, 7, 51, -49, -19, -36, 11, -85, -149, -73, -17, 18, -3, 74, 1, -55, 133, 12, -133, 51, -14, 43, -86, 5, 6, 37, -52, -1, -177, -13, -85, 100, -64, 52, 4, -82, 116, -61, -6, 11, -93, -11, 122, -19, 57, 9, -78, 3, 11, 83, 80, -76, -4, -19, -14, -18, 46, 30, -41, 38, 44, 29, -16, 106, 35, -60, -72, 0, 58, 28, -3, -159, 14, 19, 23, 38, 7, 43, 2, 3, -115, 45, -49, -72, -57, 40, -33, -5, -29, -20, 34, 0, -41, -5, 51, -19, -67, -105, -23, -179, -22, 25, -5, -17, 83, -63, -68, -50, 19, 13, -5, 4, -46, -84, -123, 18, 5, -18, 72, -2, 27, -15, -1, -31, -7, -18, 50, 77, -4, 20, 45, -22, -21, -51, -8, -38, -172, -79, -1, 41, -92, -2, -10, 59, -85, 59, -138, -37, 64, -27, 27, -3, 84, -109, -18, -10, 27, 1, -7, 0, -99, 84, 5, 31, -55, 5, 6, 4, -9, 17, 0, -9, -11, -26, 16, 0, -46, 104, -37, -114, 5, -9, 8, -9, 20, -8, -145, 2, 131, -43, -17, 55, 23, 33, -151, -2, -3, -99, -11, 4, -131, -24, -6, 117, 93, -3, -14, 121, -168, -4, -6, -106, 8, 151, 5, -3, 59, 12, 133, 0, -128}
, {-3, -65, -32, -84, -40, 6, -4, -25, -19, -2, -11, -41, -33, -2, -43, -40, 0, -55, -22, -9, -80, -27, 26, 1, 5, -3, -6, -5, 3, -6, -20, -16, -2, -11, -22, -26, -1, -23, -4, -11, -42, -17, -12, -75, -88, -15, -48, -40, -10, -12, -15, -14, -7, 3, -105, -3, -95, -20, 4, -69, -23, 2, -9, -6, -22, -82, -15, -116, -23, -14, -8, -88, -73, -3, -12, 15, -45, -8, -26, -30, -6, -74, -10, -1, -12, -14, -33, 3, -13, -88, 5, 0, -8, -28, 4, -30, -13, -18, 0, 8, -26, -18, -12, -27, -20, -12, 84, -61, -59, -12, 24, -34, -12, -17, 10, -22, -14, -18, -93, -3, -78, -11, 0, 65, -24, -21, -14, -31, -4, -55, -81, 112, -111, -10, -10, -42, -50, 4, -7, 27, -55, -12, -26, -30, -1, -78, -22, -15, -18, -26, -29, -17, 6, -73, 0, -17, -7, -31, -16, -31, 8, -54, -7, -55, -34, -13, -22, -36, -38, -7, -76, -9, 66, -18, -51, -28, -17, 3, -12, 3, -32, 43, -37, -30, -15, -9, -22, 38, -2, 1, -21, -27, -5, -14, -9, -55, -28, -3, -12, -72, -8, -4, 12, 32, -78, -11, 7, 16, 4, -64, -15, 19, 31, 1, -39, 4, -22, -59, -18, -22, -22, -24, -16, -44, -15, -47, -17, -42, -3, -10, -18, -1, -17, -22, -91, -53, -122, -19, -25, 1, 2, -11, -63, -23, -22, -91, -43, -3, 21, -14, -5, -121, -20, -14, -18, -66, -10, -47, 17, -123, -32, 2, 7, 5, -16, -8, -18, -80, 1, -25, -23, -36, -19, -61, -23, 11, 50, 48, -30, -16, -22, 66, -4, -14, -9, 74, -24, -38, -9, -35, -9, -12, -23, -30, -22, 3, -8, -4, 19, -14, -47, -14, -84, -10, -11, -19, -3, -25, -16, -54, -52, -46, -5, 2, -16, -57, -3, 39, -8, 33}
, {7, -39, -5, 7, -11, -2, 6, -19, -25, -7, -16, -8, -39, -3, -1, -16, 10, -35, 11, -10, 4, -32, -32, 0, 7, 2, -1, 8, -4, -1, 19, -15, -6, -7, 13, -42, -4, 15, -1, -28, -28, 3, 8, 11, -28, 2, -29, -18, 5, 15, 7, 2, -8, -5, -20, -37, -47, 11, -1, -44, -31, 1, 10, -10, 17, -49, -28, -25, 3, 12, 12, -31, -17, -6, 2, -13, -38, -11, -18, -4, -15, -33, 0, -64, 0, -8, -7, -10, -6, -11, 1, 6, -13, -14, 17, -14, -18, -15, -7, -54, -11, -2, -10, -5, -8, -3, -11, 21, -39, -9, -30, 3, 11, -11, 30, -1, -44, -31, -60, -10, -32, -16, 9, -45, -21, 2, 10, -2, -2, -60, -20, 18, -21, -5, 6, -14, -52, 14, 2, -35, -25, 0, -11, -17, -13, -34, 9, -52, -14, -8, -21, -14, 0, -9, -4, -14, -17, -13, 2, -8, -4, -57, -1, -59, -20, -12, 19, -21, -28, 16, -2, -6, -20, 2, -57, -23, 1, -7, -37, -12, -36, -8, -39, -18, -24, 13, 2, -23, -9, 3, 16, 5, 8, -9, -2, 3, -9, 0, -7, -43, -8, -2, 2, 1, -42, 3, 0, -5, 1, -6, -2, -59, -11, -7, 14, 7, -4, 9, 4, -3, -13, -14, -4, -19, -15, -27, 2, -47, 17, 12, 0, -18, -11, 11, 9, -23, -24, -10, -36, 6, 8, 4, -7, 1, -1, -41, -34, 6, -11, 11, 13, -13, 17, -3, 9, -16, -1, -46, -6, 28, 13, 20, -13, -51, 0, 14, -3, -4, -18, 12, -25, -32, 2, -17, -6, -44, -20, -3, -11, 5, -5, -3, 0, -3, 9, -6, 10, -23, -16, -42, 11, -49, -7, 6, 2, -30, -4, 20, 12, 7, -9, 5, -44, -1, 15, -6, 10, 4, -18, -23, -8, 1, -23, 5, 6, -31, 9, 5, -9, 4}
, {-22, 29, -7, -69, -62, -22, 5, -53, -42, -18, 6, -7, -108, -14, -18, -51, -9, -4, -21, -41, -2, -35, -54, 5, 6, -71, -8, -12, -19, -14, 11, -15, 6, 46, -11, -57, -17, -50, -17, -87, -19, 1, -55, -3, -23, -21, -56, -17, 1, -19, 17, -13, -27, 13, -16, -7, -15, -3, -3, 47, -9, -16, 2, -5, -18, -54, 20, -40, -33, 4, 1, -41, -38, -12, -5, 1, -57, -14, -15, -21, 2, 1, -9, -44, -19, -14, -67, -15, -11, -6, -9, 1, -11, -6, -16, -3, 1, 4, -44, -54, -3, -26, 0, -62, -31, -14, 0, 5, -34, -7, -8, -9, 0, 1, 23, -1, -35, -69, 63, -39, -39, -7, -9, -46, -48, 22, 5, -8, 6, -41, -27, -12, -53, -22, -16, -42, -45, -19, -4, -2, -22, -10, -6, -44, 0, -12, -8, -59, -48, -2, -67, -19, -23, -11, -16, -22, -16, -10, 1, -33, -14, -43, -31, -32, -6, 20, 1, -84, -10, -10, -4, -37, -41, 1, -19, -11, -1, 0, 17, -5, -60, -7, -74, -24, -24, -1, -8, -48, 4, -7, -19, -50, 1, 54, -23, -18, -63, -15, -50, 5, -26, -4, 3, 0, -30, -16, -29, -34, -6, -31, -11, -21, -65, -13, -16, -5, -23, -20, 0, 6, 6, -50, -38, -2, -24, -54, -41, -28, -20, -2, -9, -52, 0, 3, -2, -5, -15, 1, -15, -24, -3, -5, 24, 5, -46, -53, 30, -35, -30, 0, -12, 0, -14, -16, -10, 2, -12, 9, 39, -46, -20, -1, -11, 30, -38, 2, -10, 11, -63, 5, -28, -21, -14, -4, -11, -33, 6, -44, 3, 2, -5, -27, 6, -15, -16, -29, -52, -28, -25, -38, -24, -1, -3, 11, -8, -64, 3, 0, 6, -18, -24, -13, -40, -10, -12, -3, 83, -1, -59, 15, 61, -45, -11, -10, -20, -12, -4, -1, -22, -15}
, {-7, -20, 9, 17, 21, -9, 5, -21, -34, 4, 9, 26, -59, -4, -9, 17, -4, -16, 2, -35, -21, -6, 30, -13, -12, -17, -4, 16, -25, -16, -1, -26, -6, -33, 5, -48, -16, -27, -3, -33, 4, -11, -4, -26, -50, 10, -53, -43, 4, -10, 10, -11, -21, -8, -32, -21, -26, -20, 12, -57, -3, -1, -4, -12, 14, -51, -31, -37, -2, 1, 18, 2, -18, -13, -9, 1, -4, 14, -17, -28, -5, -2, 10, -68, -72, -9, -27, 1, 9, -40, -4, -12, -5, -27, 3, -34, -2, 13, 16, -46, -24, -6, -4, -48, -22, 6, -28, -28, -25, -8, -55, -46, 1, -1, -14, -3, 60, -39, 1, -13, -21, 3, 9, -29, 4, -10, 16, -13, -7, -41, -47, -12, 11, -12, -4, -42, -28, -7, 12, -10, -41, -11, -18, -17, 6, -28, -5, -50, -53, -18, -50, -1, -11, -26, 20, 14, -8, -16, 10, -4, 1, 26, -19, -52, -22, -23, -7, -40, -9, 0, -24, -16, -37, -1, -51, -29, 11, -10, -34, -11, -46, 0, -88, -19, -25, 11, 0, -13, -11, 19, 5, -51, 20, -56, -31, 29, -6, 1, 5, 12, -14, 20, 17, -10, -60, 7, -23, -20, -11, -14, 18, -33, -62, -11, -37, 14, -4, -5, 13, -7, -12, -33, -12, -4, -8, 10, 11, -31, -10, -36, 14, -54, 6, -16, -20, -32, -34, -3, -41, -37, -6, 0, -4, -8, -30, -9, -70, -18, -10, 19, -10, -68, -7, 5, 6, -48, -3, 5, -36, 29, -7, 3, -12, -24, -13, -4, 3, 39, -52, -11, -21, -20, -8, 0, 10, -57, -7, -26, 22, -10, -6, -7, 21, 15, -13, -30, 3, -17, -20, -27, -4, -55, -20, 3, -15, -22, 4, 3, -18, -51, -39, 13, -23, -20, 17, -17, 17, -16, -45, -11, -37, -12, -32, -2, 1, -22, -1, 0, 5, -17}
, {-23, -30, -15, 3, -12, -3, -4, -15, -40, -2, -13, -51, -17, -1, 9, -24, -11, -17, -1, -2, -12, -2, -10, -20, 6, -21, 5, -1, -4, -12, -7, -19, -17, -34, -13, -27, -7, -10, -22, -36, -39, -3, 11, -41, -2, -15, -8, -10, -12, -18, 6, -6, -2, -72, -14, -29, -18, -13, -5, -1, -3, -16, -1, -49, -11, -46, 4, -61, -17, -23, -6, -24, -52, 6, -4, -12, -55, 2, -29, -8, -33, -24, -14, -22, -2, -36, -13, -23, -14, -31, -13, 2, -21, -32, -18, -8, -12, -20, 5, -12, -9, -50, 6, -51, -21, -20, -1, -91, -64, -1, -15, -10, 6, -13, 6, -26, -35, -9, -39, -4, -16, -12, 3, -69, -13, -12, -20, -55, -2, 1, -35, 20, 4, 3, 6, -39, -36, -6, 9, -29, -55, -17, -13, -37, -10, -28, 4, -6, -42, -16, -37, -19, -5, -33, -11, -21, -14, -30, -15, -12, -23, -2, -11, -6, 5, -6, -12, -23, -2, -12, -19, -5, -4, -4, -14, -22, 5, -20, 15, -10, -17, -26, -6, 9, -22, 4, 2, 21, -2, -12, -7, -3, -23, -24, 2, -7, -15, -10, 5, -36, -24, -20, -8, -29, -77, -5, -6, 3, -14, -19, 5, 0, 26, -3, -7, 1, 2, -25, -15, -20, 1, -38, -5, -27, -6, -27, 2, -2, -6, -7, -7, -5, -15, -17, -34, -15, -43, 3, -24, 5, 4, -14, -1, -23, -2, -42, 22, 10, -39, -17, -18, -52, -21, -12, -8, -4, 3, -7, 15, -57, -23, -2, -8, 23, -39, -4, 6, -59, -13, -21, -16, -12, -22, -41, -11, 6, 35, 2, -15, -5, 1, -21, 0, -14, -17, -34, -18, -31, -10, -6, -5, -6, 1, -3, -21, -14, -9, -17, -4, -20, -35, -16, -17, -15, -2, -18, 2, -30, -36, -75, -1, -19, -13, -26, -16, -37, -4, -14, 3, -1}
, {-14, 4, 3, -46, -15, -15, 5, 6, -22, -8, -18, -3, -7, 4, -12, 3, -3, -14, -23, -36, -11, -8, -14, 1, 13, -22, -1, 7, -14, -11, 7, -15, -19, -56, -44, 4, -21, -11, -20, -17, 0, -15, -12, -3, 5, -12, -2, -13, -19, -13, -11, -8, -15, -15, -24, 1, 0, -21, 2, 7, 3, -15, 5, -18, 2, 5, -22, -5, -15, -22, 4, -10, -9, -16, -13, -19, -11, 10, -5, 2, 4, -14, 1, -16, 7, -19, -6, -11, -13, -9, -17, -17, -12, -4, -11, 9, -20, -39, -51, -3, -2, -9, -2, -3, 0, -14, -15, 23, -9, -2, 4, -13, -5, -7, -4, 3, -23, -7, 1, -10, -15, -19, 4, -27, -8, -3, -6, -22, 4, -10, 8, -18, -17, 6, -15, 6, -10, -1, 15, -19, -15, 0, 0, -22, -2, 5, -14, -8, -18, 8, -9, -16, -3, -2, 15, 0, -5, -4, -7, 8, -6, -42, -44, -11, -11, -13, -16, 6, -5, -8, -13, -17, -1, 5, -9, 8, 9, -3, -9, -13, -13, -1, -22, 4, -23, -9, -16, -8, 6, -6, -1, -4, -15, 2, -16, -28, -17, 5, -3, -9, -19, -4, 3, 0, -2, -11, -12, -21, 10, -15, -12, -29, -12, 2, -1, -1, -21, -27, 3, -19, -11, 0, -6, -5, -13, -20, -51, -11, -18, -2, 9, -11, -10, -12, 6, -21, -15, -2, -24, -24, 5, -7, 1, -12, -20, -10, -22, -21, -4, -3, 0, -44, -20, 6, -8, -18, -22, 0, 2, -52, 0, -9, -16, -19, -29, -4, -22, -16, -2, -9, 6, -19, -4, -9, -3, -23, -14, -8, -29, -21, 1, -5, -8, -1, 1, -11, -20, 2, -21, -56, -31, -17, -20, -9, 7, -14, -13, 0, -17, -19, 4, -18, -21, -25, -7, 7, -19, -24, -1, -25, -9, 0, -20, -12, -10, 3, 1, 5, 8, 2}
, {5, 9, -16, -99, 48, 21, 43, 22, 3, 19, 10, 60, -80, 71, 2, -19, 23, -23, 12, 57, -7, 66, 40, 15, -7, -2, 5, 24, -12, 43, -72, 102, 98, -132, -80, 30, -23, -32, -3, -27, 81, 16, -84, 10, 21, 24, -19, 5, 11, 8, 30, 38, -38, -62, -71, -88, -52, -1, 18, -60, 55, 0, 17, 30, 20, 57, 45, -78, 44, 11, 43, 75, -61, 3, -4, -97, -138, 63, -111, -12, 25, -9, 14, -48, 18, -12, 10, 0, 0, 54, 20, 32, 4, -8, 64, 13, 6, -1, -14, 23, 73, -12, 14, -32, -30, 13, -62, -31, 69, -2, 37, 74, -4, 21, 74, 7, 26, -108, 73, -22, -46, -5, 20, 31, 40, -11, -3, -38, 2, -23, -7, -51, -73, 15, -66, -8, 30, 11, 16, -57, -43, -31, 80, 18, 6, -2, 7, 0, -3, -7, -33, 3, -4, -72, -4, 4, 16, -50, 78, 1, 70, 84, -78, -72, 12, 0, -2, -83, -7, 14, -138, -133, -85, 3, 87, 42, -19, 1, 37, -23, 31, -65, 44, 85, 8, -10, 14, 21, -34, 79, 5, -106, 21, 48, -82, -51, 118, 7, -2, -34, -46, -6, 7, 55, -132, 82, 126, -8, 6, -143, 14, 38, -59, -27, 7, 1, 0, -93, 19, -9, -11, -56, 19, 26, -12, 14, 71, -87, -15, -16, 3, -98, -12, -6, 35, -104, -119, 4, 61, -40, 38, 5, 52, -40, 4, 8, -32, -22, -171, 10, -1, -3, 3, -77, 16, -88, -5, -28, -117, 2, 42, 3, -8, -59, -16, -1, -1, 180, -12, 1, 103, -39, -13, 13, 13, 44, -23, 6, 73, 17, 12, 72, 19, 3, -2, 30, 19, 96, 96, 61, -9, -43, 42, 115, 20, -63, 24, 1, -10, 23, 0, 20, 5, 18, 25, -4, 63, 49, -40, 60, 24, -46, 60, -31, 13, -88, 25, 40, -2, -72}
, {-20, -17, -20, -4, -6, 12, 2, -15, -10, 11, -11, -11, -16, -5, -23, -22, -9, -19, -5, -16, -12, -9, -21, -23, -4, -16, -13, -9, -25, 3, -8, -5, -3, -2, -14, -10, -14, 1, 14, -11, -23, 2, -13, -24, -1, -3, -21, -3, 13, 17, -8, -3, -17, -12, -23, -5, -17, -11, -22, -17, -3, -17, 0, 7, -20, 0, -22, 1, 16, -14, -12, 4, -19, -7, -16, 5, 6, -17, 6, 2, -7, -6, -14, -1, -18, -15, -15, 0, -4, -16, -4, -24, 1, -10, 15, -8, -5, -28, -3, -2, 2, -6, 0, 2, -11, 10, -12, 1, -16, -16, -6, -10, -5, -1, -16, -3, -5, -13, -18, -18, 10, -14, -7, 6, 2, -1, 4, 2, -10, -13, -13, -7, -22, -17, 12, -9, -3, -21, -17, 3, -4, 4, -7, -9, -18, -19, -4, 3, -9, -21, 5, -1, -5, -17, 4, -14, -11, -15, -20, -2, -26, -50, 1, -17, -15, 5, -8, -24, -5, -14, -6, -27, -22, -13, -22, -3, -20, -21, -21, -5, -21, 3, 4, -7, -11, -21, -22, 7, -18, -6, -1, -5, -20, 2, 3, -14, -20, 2, 7, -14, -11, 6, 16, -6, -24, -18, -10, -9, -24, -4, -11, 5, -4, 1, -15, -3, -20, -20, -14, -16, 2, -21, -3, -9, -20, -39, -15, -9, -5, -20, 4, 2, -22, -3, -10, -8, -22, 1, -12, -12, -12, 19, -8, 3, -7, -6, 5, -13, -21, -12, -12, 6, 4, -22, -4, -11, -13, 15, -1, -18, 2, -15, -17, 12, -16, 6, -20, 1, -4, 2, 7, -15, -1, -22, 0, -6, 3, 3, -21, -19, 3, 2, 1, 2, -18, -12, 5, -14, 4, -44, -4, 2, -12, 2, -16, -7, -8, -21, -1, 1, -7, -21, -10, 2, -2, 6, -13, -1, -15, -11, 5, -24, -3, 0, 3, -7, -11, -16, -22, 2}
, {11, 71, -6, -125, 37, 18, 40, 73, -26, -14, 3, -1, -35, -25, 11, -60, 2, -5, 3, -38, 53, -12, 71, 7, 14, -12, 18, -82, -17, 85, 86, 57, 18, 20, -22, 57, 13, -50, -17, 7, -92, 8, 76, -54, 4, 4, 56, -24, -94, -20, -58, -107, 41, -74, -61, -56, -57, 62, 6, 44, 57, -3, 16, 11, 14, 33, -38, 49, -39, 4, -25, 55, -14, 0, -8, 15, 86, 27, -15, -24, -21, 20, -6, -9, 4, 70, 3, -11, -19, -49, 9, -7, 60, 38, 3, -6, -10, -19, 34, 21, 140, -36, -9, 57, -82, 11, 45, -125, -8, 6, -72, -4, 53, -6, -58, -6, 40, 88, -22, -9, 50, 12, 5, -33, -62, 97, 9, -175, -22, 19, 14, 16, 18, -18, -61, -24, 95, -22, 10, 34, 25, -102, 108, -76, 20, -60, -12, 9, -18, -92, -55, -8, -7, -30, -3, 17, -17, 51, -1, 23, -39, -40, 72, -3, -29, 145, -8, -46, -90, -15, 2, -77, -162, 1, -80, -24, -81, -7, 69, -21, 22, -13, 90, 14, -34, 19, -12, 2, 51, 28, 10, -48, -21, -27, -77, 68, -20, -12, -181, -9, -3, -3, -36, 12, -41, -40, 6, -46, 7, 56, -9, 0, 19, -154, 20, 2, 4, 42, 2, -43, -14, 42, -63, -37, -127, 0, 117, -5, 56, -43, -19, -20, 36, -11, 34, -35, -36, 10, 17, -103, 18, 1, 20, 30, -18, 33, 73, 69, 60, -32, -23, -66, -64, -158, 4, -31, -20, -34, 16, -153, 13, 4, -109, 40, -12, -1, -3, -66, -50, -44, 36, 11, -5, 16, 2, -25, -18, 0, 25, -21, -8, -9, 22, 12, -32, 92, 92, -44, -56, 34, -93, -47, -66, 1, -5, -58, 15, -26, -18, -42, -53, 7, -9, -45, -39, -3, -41, 24, -10, -40, 34, 69, -4, 16, -10, 37, -2, 100, 7, -104}
, {-11, 3, -72, -33, 106, -2, 37, -13, 3, -13, -28, -33, -21, -2, -125, -38, -89, 35, -21, -156, 25, 57, -12, -5, -14, 56, -7, -9, -22, -32, 173, 20, 11, 84, 86, -26, 51, 22, -14, -32, -58, -19, -70, -46, -15, -5, 19, 45, -62, -9, 74, -60, -36, 90, -3, -10, 12, -46, -11, -43, 54, 20, -4, -101, -14, 38, -21, -23, -6, -19, 62, -8, 22, -15, -27, 67, 27, -64, 89, -21, -12, 23, -3, -17, 6, 39, 26, -14, -23, 13, -3, 74, -4, 43, 6, -6, -30, -25, -80, 23, 2, -43, -25, -21, -31, -10, -6, 62, -20, 1, 15, -49, 18, 3, -19, -38, 50, -100, 8, -20, -18, 65, -6, -46, -1, 64, -2, 49, -2, -30, -13, 30, -85, 5, -49, -13, 7, 2, -3, -30, -15, -22, -97, -50, -36, -65, 5, -37, 44, -44, -14, 6, 0, -1, -7, 0, -7, 5, 13, 32, 10, -107, 10, -29, -5, -2, -15, -53, -48, 9, 46, -89, 6, 1, -40, 21, -58, 19, 47, 68, -35, -17, 31, 34, -35, 30, -14, 53, 15, -10, -18, -65, -31, 35, -50, 38, -26, -19, -8, 63, 79, -9, -7, -68, 27, -52, -14, -31, 27, 41, -8, 2, 21, -7, 9, -15, -14, 73, 20, -31, -18, 25, -39, 100, 19, -34, 41, -23, -38, 4, -8, -50, 25, 3, 26, -15, -13, 9, 30, -15, -25, -6, -35, -42, 32, -57, 32, -9, 3, 17, -11, 45, -22, 2, 5, -52, -4, 21, 13, -115, -15, -6, 5, 95, 66, 4, 1, -53, -89, -66, -144, 0, 1, -74, -18, -40, 24, -66, 21, -7, -22, 86, -2, 8, -14, 44, -60, 69, -3, -50, 0, 63, 44, -7, -14, 15, 1, -5, 49, -88, -84, -2, -50, 4, -60, -18, -75, -9, 2, -57, 69, 59, -38, 9, 4, -20, 13, 35, -14, -110}
, {5, 13, 11, 15, 11, 26, -134, 26, -2, -2, -9, -39, 4, 75, -50, 6, 19, -10, 1, 11, 49, -137, 50, -8, 16, 15, -3, 85, 22, -19, 17, -44, 25, 94, 85, -14, -3, -5, -28, 66, 30, -19, -110, -138, 36, 26, -30, 19, -45, -14, 62, -31, -10, -130, 1, 9, 30, -27, 3, 10, -81, 73, 26, -27, -13, -58, 73, 23, -24, -17, 50, 35, 35, 10, 8, -74, -76, -12, -26, 0, -2, 63, 13, -80, -53, -128, -48, 0, 24, -58, 11, 83, 47, 44, 109, -10, 34, -6, 80, 10, 83, -16, 14, -57, -86, 16, -99, -109, -43, 5, -36, 4, -36, 2, 20, -55, 37, -61, -52, 120, 5, -12, 3, -64, -158, 78, 3, -161, 14, -71, -29, -87, 50, 22, 30, 31, -81, -30, 35, -69, -61, -78, 73, -70, 57, -90, -8, -8, 19, -7, -136, -35, -9, -13, 18, -44, -48, -27, 29, -2, 47, -37, -12, -14, 31, -55, -38, -37, -91, 71, -196, -128, -116, 3, -18, 91, -72, -3, 4, -125, 22, -48, -69, 62, -17, 3, 13, -23, -107, -77, -20, -327, -22, -44, -24, -51, -39, 5, 57, -66, -12, -15, -9, 37, -57, 51, 67, -99, 8, 55, -7, 65, 13, 18, -93, -6, 2, -72, 9, -9, 27, 6, -61, 72, 66, 34, 38, -21, 11, -56, -19, -45, 6, 52, -50, -70, -14, -17, 1, 17, 37, -2, 24, 67, 16, -180, 44, 19, 87, -42, 3, 34, -14, 60, -11, -114, -36, 39, 0, 62, -105, 0, -53, -67, 59, -1, 16, 25, -9, -21, 109, -23, -56, 49, -3, 64, -6, -99, 3, -17, -1, 60, -16, -70, 134, -33, -8, 7, 4, 3, -37, -48, 95, -82, -35, -19, 66, -82, -132, -64, 2, 1, -45, 42, 12, -22, 76, 49, 15, -61, -8, 29, -50, 41, -15, 112, 87, -111, -21, -39}
, {-16, 3, -141, 2, -44, -4, -62, -89, -118, -22, 0, -49, -62, -15, -44, -80, 11, -109, -17, -16, -105, -2, -65, -1, 0, -63, -10, -58, 26, -34, -133, -12, -174, -4, -29, -3, -68, -66, -19, -6, -117, -12, 25, -17, -98, 12, -119, -86, 12, -4, -40, 39, -104, 74, -121, -22, 48, -18, -11, 0, -47, 35, -14, 30, -11, -64, 21, -102, 16, -4, -5, -2, -35, -11, -11, 109, -12, -81, -23, 152, 26, -57, -13, -5, 23, -25, 172, -17, -6, -26, 5, -6, 7, -8, -13, -51, -97, 57, -96, -40, 23, -58, 4, 40, -69, -16, 16, 6, 60, -9, 6, -46, 91, -2, -16, 48, 65, 61, -23, -16, -81, -4, -11, -66, -88, -24, 0, -43, -4, -44, -81, 6, -60, -1, -57, -54, 12, -20, -10, -115, 55, 31, -52, -27, 16, -24, 0, 14, -42, 43, -72, -14, -12, -82, 17, 13, -16, -66, 46, -80, -205, 22, 65, -28, 95, -17, -13, 52, -182, -21, -56, -23, -55, 2, 26, -107, -101, -22, 7, -51, -8, -35, -13, 12, -55, 27, -19, -129, 18, 10, -17, -109, 6, -90, -164, -17, -189, 6, -50, -274, 43, 5, 20, 62, -85, 0, 14, 11, 44, -95, -22, 17, -86, 4, 61, -6, -1, -10, -7, -20, -6, -117, 18, -4, -19, 67, -20, -106, -113, -9, -4, -7, -157, -2, -76, 93, 19, -10, -158, -106, 2, -10, -39, 25, -40, -112, -141, -48, -2, -6, -17, -133, 69, -9, 1, 55, -8, 76, -85, 80, 83, 3, -31, -77, -27, 5, -5, 65, 96, -89, -2, -133, 8, -116, 0, -11, -61, -53, -96, -12, 4, -119, -3, -21, -1, -94, 62, -33, -174, 12, 163, -8, -18, -57, -2, 19, -109, 5, 86, 41, -108, -7, 1, -28, -124, -11, 127, -26, 35, 44, 27, -99, -146, 43, 4, 67, 77, -15, 3, -6}
, {5, -20, -2, -14, -16, -8, 1, -15, -15, -22, -19, 6, -20, 5, -13, -16, -7, -13, -8, -11, -9, -2, 3, -21, -17, -18, 5, -1, -15, -21, 5, -1, 6, -11, -5, -20, 3, -16, 0, -21, -2, -10, -20, -13, -23, -7, -3, -10, -19, 4, -3, 3, -16, -20, 7, -20, -5, 3, -7, -11, -18, -7, -22, -9, -6, -19, 3, 2, -10, -10, -12, 9, -8, -16, 6, -11, 2, -19, -19, -20, -22, -8, -14, -13, -10, -9, -20, -20, -1, -4, -18, -11, -17, -9, -6, -17, -6, 2, -16, -11, -17, -3, -20, -20, -19, -7, -11, 0, 2, -21, -20, 2, -6, -19, 4, -23, -18, -22, -8, -1, -2, -5, -19, -21, -23, -8, -14, -17, -22, -3, 3, -6, -3, 4, -13, -7, -17, -8, -8, -16, -22, -11, -9, -14, -3, -1, -4, -2, -13, -19, -1, -15, -16, -20, -9, -6, -2, -14, -21, -19, -11, -13, -1, -19, 0, -15, -14, 4, -15, -21, -17, -7, -7, 2, -17, -14, -7, -22, -21, -15, 4, -10, 5, -12, 5, -20, 5, -5, -21, -20, -5, -17, -19, -15, -11, -1, -12, 2, -3, 4, -3, 4, -16, -2, -21, -10, -1, -13, 1, -20, 3, 1, -20, 4, -21, 6, -3, 2, -3, 4, 0, -7, -12, -7, -18, 5, -16, -6, -2, -8, -18, -7, 6, -2, -6, -16, -15, -10, -8, -1, -8, -8, -2, -10, -12, -1, -13, 3, -4, 3, 0, -8, 1, 1, -10, -19, -14, 4, 6, -2, -16, 2, -5, 9, -1, -3, 5, 6, -14, -6, -7, 1, -14, -8, -9, -19, -21, -14, -2, -15, -15, -9, 5, -14, -14, -19, -20, -15, -11, -7, 2, 2, 2, -16, -17, -23, 2, -13, -14, -11, -17, -20, -7, -1, -20, -18, -20, -9, -9, 3, 10, -1, -5, -22, 1, 4, 4, 6, 4, -1}
, {2, 59, -23, -194, 37, 13, 71, -21, 49, 18, 2, 7, -79, 50, -87, -15, -12, -140, -11, -147, 25, 35, 8, 5, 2, 81, 18, -59, 11, -18, -11, -115, -32, -59, 51, -16, 27, -23, -5, -50, -44, 5, -99, -51, -89, -15, 92, 26, -6, -9, 64, -92, -24, 26, -51, -62, -114, 60, 13, -6, 116, -14, 18, -56, 5, 10, 5, -158, -32, 1, 83, 75, 36, -2, 1, 106, -61, 116, -203, -22, 39, -85, -7, -131, 22, 53, 34, -25, -11, -39, -2, 106, -3, -9, -70, -9, 27, -88, 54, 24, 90, -40, 17, -13, 1, 16, 45, -53, 43, -2, 21, -11, 19, 13, 14, 44, 96, -10, -45, 42, -16, 116, -14, 53, 115, -43, 25, -42, -12, -8, 18, -35, -76, -2, 24, 74, 24, 5, -8, -146, -23, 117, 55, 41, 13, -48, 17, -110, 59, 53, 47, -5, 21, 18, 8, 31, -28, -3, -34, 4, -4, -103, 2, 20, -21, -15, 9, -23, -62, 20, -26, -97, -18, 16, 31, 57, -45, 9, 46, -4, 64, -85, -66, -14, 33, 33, -1, 18, -12, 110, 9, -10, 2, 48, 8, -23, -168, 3, 25, 57, -18, 12, 6, -36, -75, 6, -67, 16, 20, -9, 10, 15, -1, 52, 62, -5, 10, 44, -3, -21, 5, 39, -147, 8, 31, -224, -17, 17, -31, 34, 9, 9, -3, 2, -95, 34, -36, 9, -29, 2, 1, 0, -24, 29, 33, -33, -45, -18, -91, 38, -3, 0, -76, -6, 13, 72, 2, 17, -2, -42, -143, -1, -51, 16, -14, -7, -4, -2, -80, 83, -174, 5, -11, -46, 4, -45, -2, 62, 23, -12, 7, 1, 3, -50, 58, 32, -5, -13, 93, -73, -59, 23, 13, -8, 21, -23, 39, 18, 4, -43, -2, 18, 22, -5, 5, 13, -29, -45, 1, -99, -32, 14, -2, 4, 6, -33, -16, 114, -1, -24}
, {5, 31, -10, -49, -10, -11, -20, -4, -18, 6, -17, 3, -9, -13, -12, -32, -7, -11, -18, -49, 12, -20, -23, -16, -22, 4, -18, -12, -21, -7, -21, -4, 1, -26, -20, -29, -16, -9, -5, -49, -5, -9, 11, -31, -22, 3, -23, 2, -8, -13, 41, -7, -9, -36, -1, -5, -22, 3, -8, -13, 7, -10, -21, -52, -10, -41, -39, -16, 18, -1, 8, -25, -17, 1, -12, -11, -54, -22, -16, -25, 3, -8, -17, 3, -28, -24, -3, 2, -15, -13, -5, -21, 3, -6, 2, 0, -5, 17, -19, -4, -15, -21, -7, -15, 1, -3, 3, -38, -22, -9, -58, -13, 5, -23, -21, -19, -20, -40, -43, -18, -15, -8, -9, -38, 0, -19, -9, -24, -3, -20, 0, -29, 8, -11, 8, -22, -16, -17, -13, -26, -15, -2, -15, 3, 3, -12, -4, 4, -5, -17, -13, -20, 4, 7, 4, 6, 2, 1, -6, 6, 0, -7, -9, -9, 1, -1, -16, -18, 5, -19, -42, -60, -23, -12, -27, -6, 0, 1, -40, -14, -12, -66, -26, -9, -14, 2, -18, -27, 3, -23, -2, -18, -13, 10, -10, -28, -10, -8, -11, 0, -16, -12, -20, -11, -25, -20, -34, -17, -17, -15, 0, -42, 19, -42, -9, 1, 2, -1, 7, -13, 3, -4, -21, -21, -2, -4, 4, -10, -7, -5, -13, -27, -1, -2, -15, -26, -5, 3, -40, -17, -3, -3, -6, -10, 4, -48, -20, -17, -14, -7, 6, 2, -8, -15, 0, -5, -17, -14, -14, -58, 20, -18, 18, -15, -2, -1, 3, -9, -26, 5, -19, -14, -5, 0, -16, -40, -17, -18, -14, -16, 3, -3, -22, 1, -5, -9, -9, -5, 1, -27, -17, -19, -8, -4, -15, -20, -20, 0, -36, -33, 2, -14, -46, 3, 0, 6, -5, -19, -17, -43, -19, -8, -22, -7, 2, -9, -8, -4, -20, -22}
, {-21, 0, -17, -1, -18, 2, -15, 0, 0, -22, -7, -3, -19, -16, 1, -13, -21, 3, -11, 4, -11, -21, -17, -12, -16, 0, 3, 2, -17, 7, -9, -12, -16, -10, -4, -14, -13, -10, -22, -11, -17, 0, -19, -11, -20, 2, -10, 5, -9, -20, 2, -14, -2, -14, -1, -19, -4, 3, -1, -12, 2, -19, -11, -21, 1, -18, 6, -14, -9, -7, -7, -9, 6, 3, 1, -22, -5, -8, 5, -22, -12, -11, 3, -1, 3, -14, -14, -20, 4, 1, -14, -4, -7, -23, -6, 4, 3, -10, 3, -8, -13, 1, 4, -20, -19, -1, -3, -10, -5, 4, -8, -17, -20, -17, 0, -11, -21, 6, -18, -10, -15, -4, 3, -5, -3, -10, -17, -15, -16, -3, 5, 5, -12, 5, -16, -12, -13, -21, 6, -19, -6, -17, -10, -4, -4, 5, -15, -17, -6, 3, -9, -19, -23, -14, -5, -5, -19, -9, -10, -6, -19, 11, -17, -13, -18, -15, -19, -16, -13, -21, -7, -12, -7, -9, -10, -11, -2, 4, -21, 6, -3, 6, -20, 5, -10, -13, -9, -18, -11, -18, -8, 0, -20, -4, -3, -3, -17, -10, -14, -14, -7, -22, -19, -18, -2, -15, 2, 6, -18, 0, -1, 6, 0, -4, -5, -2, -21, -12, -11, -8, -23, -8, 2, -17, -22, -4, -19, -15, -11, -11, -11, 0, 1, -4, -4, -8, -12, -8, -1, 6, -13, -4, -22, 2, -3, 5, 5, -21, 5, -12, -1, 6, 1, -22, -20, 6, -23, -18, -13, -2, 6, 4, 19, 12, -13, -20, 4, 1, -14, -9, 0, -19, -17, -21, -4, -16, -4, -6, -2, 4, -18, 0, -10, -3, -4, 5, -14, -15, -7, 3, 1, -7, -6, 3, -2, -17, -4, -7, -21, -10, 4, -19, 6, -2, -14, -11, -3, -22, -8, -14, -12, -8, -16, -8, 3, -13, 7, -8, 5, -7}
, {-15, 20, 20, 59, -104, 7, -67, 47, -106, -17, -8, 11, 30, 14, -57, 34, -23, 20, -17, 26, -10, -53, 88, 3, -8, -6, 7, 42, -92, -71, 17, -22, -9, -22, -11, 32, -84, 112, -9, -4, 82, 6, 44, -217, -33, -3, -33, 1, 8, -16, 13, 20, 18, -68, -43, 105, -47, 0, -16, 34, -161, -52, -31, -82, -1, -54, -58, -68, 21, -1, 66, 112, 30, -8, 3, 13, 27, 2, -74, -34, 12, 7, 5, -67, -54, -75, 21, -22, -18, -198, -4, 53, 36, 45, -21, -11, 59, 85, 171, 2, -38, 177, -2, -57, -108, -9, 41, -29, -59, -26, 118, -8, -38, -7, 53, -45, 53, -36, -38, -33, 27, -39, 5, 44, 156, -42, 8, -92, -2, -133, 26, 32, 70, -6, 2, 26, 20, -14, -17, 118, -104, -44, 32, -17, -74, 31, 0, -128, -50, -97, -45, -14, -3, -91, -18, -74, -17, 11, 14, 8, 31, 48, 114, -186, -58, -49, 17, -140, -52, -50, 20, -6, -157, -7, 102, -54, -28, 8, 140, -36, -92, 70, -293, -55, -26, -41, -5, 29, -49, 98, -10, -143, -18, -41, -48, 15, -28, -8, 3, -37, 68, -18, 17, 36, -13, -144, 87, -11, 54, -47, -11, -102, 25, -12, 8, -13, -20, 113, -16, -106, 55, -27, -175, -141, 5, 73, -163, -40, -22, 2, 2, -72, -45, -48, -13, 42, -56, 2, 7, -113, 9, -10, -158, 4, -207, 72, -223, -113, 19, -43, -16, -41, 129, 44, -10, -52, 33, -2, -4, -25, 4, -21, -21, 14, -51, -6, 9, 55, -95, 72, -96, -24, -70, -35, -6, 26, 22, 32, 34, -12, -8, 7, -16, 81, -13, -7, 129, 18, 101, -80, 7, -6, 12, -12, 16, -20, 80, 27, -36, 20, 58, -17, 33, 31, 31, 7, -39, 96, 89, -146, 2, 104, -22, 30, 5, 138, 51, -20, -27, -112}
, {-14, -61, 4, -85, 56, -1, 13, 20, 23, 11, 0, -40, -47, 127, -19, -23, -13, -30, 18, 22, 10, -13, -89, 17, 5, 17, -5, -52, 108, 9, 94, 88, -42, -23, -26, -39, 50, 13, 11, 11, -16, 23, 111, 3, -14, 38, 61, 63, 88, -12, 70, 122, 21, 76, 74, 7, -14, -31, 52, -49, 133, 5, 18, -22, 5, 65, 50, -14, -9, -37, -94, -9, 102, 7, -10, 5, 96, -18, 17, 7, -42, 37, -5, -12, 10, -5, 37, 116, -19, 118, -18, 49, -25, -31, 67, 17, -47, 10, 44, -1, 123, 68, -8, 22, -65, 64, -81, 73, -11, 14, -67, 27, 18, 9, 104, 72, 3, 33, 12, 74, 111, -31, 12, 8, -15, 72, -11, 34, -16, 18, 12, -44, -80, 0, -111, 0, 90, -41, -9, -69, 10, -44, 73, -21, 101, 22, -28, 5, 79, 15, -14, -21, 11, 73, -5, 25, -119, 39, -59, -35, -16, -24, 90, -18, 37, 61, -21, -19, -37, 122, -47, -9, 24, 0, -34, -35, -113, 17, 69, 56, 6, 50, 67, -23, 2, 3, 17, 74, 81, 54, -5, -90, -16, 29, 25, -135, -65, 2, -108, 41, 15, 8, -15, -83, -45, 48, -6, -5, 6, 4, -43, 2, 30, 137, 82, 39, 32, 51, 9, -27, -145, -6, 80, -53, 20, 25, 49, -16, 31, 30, -27, 7, -16, 7, -78, -33, -37, -3, -30, -17, 90, -10, 90, 60, 43, -27, -30, -47, 50, -35, 29, 41, 80, -80, -13, -55, -41, -41, 39, 9, -55, 14, -49, 68, 51, -4, 5, -7, -13, -53, 84, 18, 12, 24, -7, -11, -4, 26, -28, 42, 2, -4, -11, 46, 39, 33, 122, 38, 23, -66, -35, 0, 20, -2, -33, -30, -31, 43, -67, -98, 2, -24, -48, 18, -80, -3, 97, 43, 41, 59, -27, 13, -27, -14, 4, 14, 71, -118, 11, -87}
, {15, -33, -73, -53, 53, -16, 16, -40, -145, -7, -9, -20, -12, -38, -221, -70, -51, -152, -20, -52, -176, 53, -36, 0, 6, -112, -9, 17, 13, -75, -90, -162, -130, -3, -8, -61, 14, -14, -4, -67, -105, -13, -245, -22, 85, 5, -201, -172, -22, -18, -174, -266, 33, -206, -100, -72, -180, -90, -15, -7, -164, -1, 3, -35, 20, -85, -64, -37, -11, -9, -11, -74, 57, -14, 12, 96, -19, -92, -75, -7, -67, 111, 1, 93, -85, 63, -166, -6, -10, 121, -7, -12, -1, -86, -13, -55, -105, -38, 85, 15, 71, -5, -1, 28, -44, 2, -28, -13, -40, -15, 20, -167, 24, -3, -240, -249, -27, -97, 43, -112, -69, 53, -6, -13, -173, -1, -24, 44, 28, -60, -135, -5, 19, 2, -40, 1, -94, 8, -11, -44, -24, -41, -63, -68, -78, -32, -13, 1, -4, 23, 3, -4, 11, -86, -11, -7, -17, -141, -57, -81, -122, -19, -85, -20, -96, -71, -17, 19, -10, -5, -182, 55, -8, -10, -60, 47, -15, -6, -257, -131, 69, -102, -103, -52, 12, -36, -20, 55, -90, 9, -2, 73, -12, -110, -75, -57, 66, 3, -43, -51, -195, -1, -9, -25, -12, -61, -88, -123, -58, -53, -16, -83, -80, 26, -164, -1, -14, -198, 0, 6, -25, -107, 25, -34, -79, -19, -93, -6, -75, -136, 0, 1, -74, -7, 67, 63, -72, -11, 10, 97, 61, -12, -299, -53, 6, 17, 0, -23, -64, -65, -12, 41, -93, 9, 1, 96, -10, -60, -65, -39, -34, -11, -25, -57, -113, 6, -10, -30, -27, -45, -176, -38, -90, -48, -9, -41, 195, 104, 149, -15, -7, -54, -3, 33, -22, -122, -86, -87, -50, -125, -93, -84, -51, -15, 3, 67, -106, -12, -157, 77, -100, 11, -174, 73, -6, 1, -56, -137, -106, -56, -60, -48, 110, -36, 0, -100, 3, 0, -31, 76}
, {-10, -11, 0, -7, -6, -10, 0, -1, -6, -13, 2, 5, -13, -8, -1, -17, -7, -9, -8, 1, -17, -16, 5, -20, -22, -8, -20, -19, -13, 4, -4, -15, -5, -22, -20, -1, -2, 5, -10, -4, -7, -22, -22, -14, -11, -11, -22, -18, 4, -21, -13, 1, -15, -25, -9, -2, -16, -11, -12, 3, -21, -6, -9, -1, -9, 12, -22, -18, -1, -20, 12, -12, 8, -5, -16, 3, -26, -8, -21, 2, -5, -8, -17, -27, -20, -21, -5, -17, -18, 0, 3, -21, -1, -10, -17, -24, 2, -28, 4, -22, 1, -16, -17, -27, -6, -1, -11, -11, 10, -11, 7, 1, -9, -18, 5, 1, -19, -11, -33, 9, -1, -16, -19, -19, -24, -22, -19, -19, -15, -13, -11, -29, -11, -15, 1, -7, -12, -13, -5, -19, -11, -7, -16, -15, -11, -11, -12, -23, -3, -18, -4, -23, -8, 7, -1, -25, -1, 0, -12, -22, -5, -10, -7, 7, -22, -12, -1, -26, -9, -8, -24, -5, -7, -14, -4, 1, -4, 5, 2, -22, -16, -12, -13, -23, -14, 3, -1, -19, -6, -5, -17, -3, -12, 3, -16, -14, 6, 3, 4, -16, -21, -11, -3, -22, -11, -20, -9, 2, -25, -14, -24, -12, -12, -17, -8, -12, 2, -11, -6, -10, -14, 2, 3, 1, 3, 1, -4, -1, -7, -8, -13, -4, -3, -8, -12, -24, -22, -21, -11, -19, -16, -19, -6, 3, -21, -17, 0, -20, -5, -8, -2, -19, -12, 0, -8, 2, -3, 8, -24, -26, 17, -15, -12, 17, 1, -22, -2, -10, -26, -16, -1, -10, -19, -3, -16, 6, -15, -19, -8, -24, 5, -20, -16, -25, -21, -20, -2, -6, -14, 4, 0, 4, -23, -26, -8, -20, 0, -6, -12, -20, -14, -2, -2, -2, -4, -23, -2, 4, -11, -8, 4, -1, -19, -9, -11, 5, -17, -13, -22, -5}
, {-18, -8, 0, -3, -9, -21, -18, -21, 3, -16, -5, 3, -5, -11, 4, -22, -13, -4, 0, -11, -20, 3, -20, 3, -10, 4, -19, -9, -3, -21, -22, -12, -9, 7, -9, -19, -3, -17, -7, -10, 3, -8, -14, -12, -20, -18, -2, 6, -12, -5, -7, -14, 7, 5, 5, 0, -23, 4, -19, -8, -11, 2, -11, -14, -12, -19, 3, -3, -15, -13, 5, 2, -22, -19, -10, 4, -2, -18, -1, -11, -4, -9, -21, 1, 3, -15, -5, -7, -18, 0, -10, -6, -10, 1, 6, -3, -13, 6, 0, -13, -16, -8, -21, -20, 4, -16, -21, -23, -19, -11, -8, 1, -13, -23, -20, -5, -19, -12, 3, -20, 3, -4, -16, -10, -1, 2, -13, -21, -22, -6, -6, -15, 5, -21, -9, -15, -23, -4, -15, 5, -3, -11, -6, -11, -12, -2, -10, -20, -3, -2, -15, -13, -4, 1, 1, -8, -5, -1, -14, -20, -10, 4, -20, -11, 5, -16, 3, -18, -16, -1, 7, -9, -7, 1, -1, -4, 0, 1, -8, -11, -9, 5, -12, -4, -5, -10, 1, -5, -18, 6, -11, -21, -21, -8, -11, -9, -5, 0, 4, -7, -14, -9, -2, 0, 4, -19, -20, -11, 4, -4, -21, -1, -16, -22, -16, -10, -18, -2, -21, -1, -23, 3, -8, -4, -9, -2, 3, 1, -20, -14, -11, -18, -9, -22, -23, -6, -22, 1, 6, -5, -14, -12, -1, -3, 4, -15, 3, 3, -14, -1, -7, 4, -10, -21, -17, -11, -22, 5, 2, -19, -6, -17, 4, -8, -3, 6, -13, 6, -20, -3, -12, -8, 6, -16, -8, -5, -17, -1, 4, 2, -9, -6, 1, -9, -4, 0, 2, -15, -19, -1, -7, -11, -12, -11, 4, -3, -23, -16, -19, -10, -16, -2, -7, -9, -2, 6, -22, -2, -3, -19, -6, 4, 2, -8, -2, 4, -15, -16, -5, -1}
, {-2, 8, 36, -39, -232, 3, -9, 73, 10, 0, 3, 22, -98, -72, 97, 20, 30, -103, 9, -34, 41, -93, 50, 20, 13, 55, 5, 32, 16, 18, -43, 35, 51, 5, 85, -12, 162, -14, 15, 1, -28, 13, -52, -13, -16, 11, 44, 82, 79, 15, 73, -98, -5, -4, -39, -17, -144, 11, 9, -31, -63, -75, 8, 23, 5, 13, 0, 23, -97, 15, 110, -40, 63, 10, -8, -28, -164, 41, -24, -4, -20, 35, -3, -12, -93, -184, 114, 19, 22, 55, 19, 5, 0, -49, 83, 41, 38, -4, -78, 2, -172, -42, 20, -2, 53, -20, -62, -72, -9, -3, -21, -28, 19, 0, -52, 1, -129, -27, 89, -8, -1, -33, 15, 34, -6, 100, 18, -77, 6, -7, -24, -36, 127, 2, 63, -41, 2, 9, 7, -40, -175, -4, 91, 3, 64, -93, 18, 63, 54, -41, -17, -5, -6, 78, 11, 31, 18, 52, -96, -25, -25, -61, -41, 19, -29, -20, 5, 36, -32, 11, -153, 24, -59, 20, -67, -24, -18, 19, -3, -28, 28, -11, -11, -30, -44, -3, 22, 135, 35, -50, 15, -53, 4, 5, 6, 0, -60, -3, -69, -63, 43, -2, 4, 96, -82, -145, -26, -26, 11, -9, -1, 23, -62, 64, -20, 4, 20, -16, 0, -22, -10, -52, -63, -46, -78, -6, -56, 7, -2, -41, 2, 4, 66, 21, 1, -49, 36, 17, 24, -22, 128, 18, 27, -69, 21, 4, 108, -49, -5, 6, -10, 14, -15, 87, -6, -75, -10, -53, -18, 101, -46, 13, -53, -63, 42, 6, 6, 18, -1, -94, 36, -62, -17, 35, -9, -25, -109, 32, -22, 7, 8, 4, 8, 69, 23, -99, 130, -95, -13, 60, -11, -55, -16, -40, -16, -53, 2, 28, -70, -112, -42, 11, -29, -15, 39, 17, 102, -35, 7, 10, 34, -24, -86, 39, -3, 66, -121, -98, -15, -137}
, {-4, -230, -12, -5, -24, -18, 54, -139, -8, 6, 18, 55, -103, -2, 5, -29, -4, 8, -23, -61, 89, 10, -42, -13, -2, 16, -9, -6, -8, -35, -78, 54, -46, -28, -29, -119, -13, -31, -20, -29, -2, -2, -92, -85, -63, -13, 83, 34, -15, -7, -163, -83, -24, -107, -71, 36, 52, -19, -13, 34, -18, 21, -2, -13, -3, -119, -51, -66, -44, -3, -23, 44, -100, -7, -2, 1, -128, 11, 61, 22, -17, 23, -19, -12, -42, 17, -41, -6, 4, -53, -15, 15, 5, -101, -24, -31, 26, 7, -3, -54, -54, -88, -9, -123, 24, -14, 114, 0, -92, 0, 97, 65, 13, 5, -112, 43, -41, -15, 124, 59, -77, -9, -3, -30, 79, 15, -18, -35, 3, -28, -105, -75, 145, -2, 90, -20, 1, -3, 3, -135, -93, 27, 175, -16, -15, 65, -19, -2, -23, -71, -144, 2, -18, -83, 0, -14, -18, -98, 0, 25, -7, 15, -21, -160, -83, -91, -7, -209, -71, -5, 10, -81, -31, -11, 157, -80, 0, -4, -97, -36, -69, -145, 23, 30, -31, -21, -9, -39, 16, -25, -13, -69, -9, -11, 73, -113, -52, -13, 77, 8, 74, 0, -1, -38, -73, 34, -26, -6, -49, -60, -19, -81, 13, -45, -100, -6, -21, 44, -16, -17, -7, 55, -84, -36, -19, 36, 254, -68, -58, 50, -26, -112, -99, 0, 43, 30, -74, -22, 26, -136, -23, -17, -97, -38, -114, -91, 5, -99, 33, -4, -16, -70, -59, 144, -18, -35, 0, 164, 34, 79, -164, -15, 96, 33, -11, -10, 16, -49, -27, 4, -70, 137, -16, -57, 2, -31, 11, -5, -1, -16, -2, 43, -13, -34, -2, -82, 91, -12, 64, -122, -189, -180, 7, 38, 5, -14, -33, 0, 20, -27, 35, -9, -3, 22, 12, -13, -65, -50, -105, -208, -49, -71, 7, -39, 3, 124, -26, 99, -19, 3}
, {0, -34, -9, -43, 9, 15, -157, -2, 15, -4, -4, 3, -53, 115, 87, -23, 8, -25, 7, 25, 18, 37, -13, 7, 13, -30, -5, 33, -1, 87, 99, -10, 52, -75, -31, -49, -74, -23, 7, -20, 14, 22, 85, -37, -21, 26, 55, 11, -32, 6, 25, 4, -22, -12, -23, -40, 74, -9, -1, 16, -31, 18, 15, 1, 13, -51, -36, 46, 128, -1, 77, -15, -36, -5, 24, 25, 32, -28, 52, -55, 9, 43, 18, -40, -27, -131, -27, -1, 10, 10, 0, 21, -3, -21, -44, -15, -13, -17, 33, -30, -49, 7, -5, 24, -24, 6, 27, -36, -6, -5, 32, 106, -36, 20, -15, 60, 0, -44, 19, 20, -33, 18, 17, -34, 59, -7, -9, 13, 15, -35, 18, 51, 13, 0, -54, -69, 52, 13, 9, 109, 6, 47, -18, 3, 23, -16, 17, -70, -25, -39, -35, -6, -7, 76, 6, 8, -4, -16, -16, 22, -16, -107, 54, -32, 69, -32, 8, -17, -40, 17, 54, -110, -95, 9, 28, 67, -24, 4, 90, -49, -19, -80, 40, 71, -9, -10, -5, -79, 67, 59, 0, -41, 20, -15, 0, -8, 33, 3, 15, 49, 36, 16, -1, 60, 9, 32, -43, -3, -2, -41, 15, 61, 16, -4, -9, -6, 20, -40, -4, 15, 4, 30, 64, -6, 2, -12, 99, -15, 12, -29, 21, -42, 31, 22, -103, -85, -81, 14, 58, 19, -33, -3, 33, 22, 57, 2, -7, 72, 33, 19, -3, -3, -43, -55, 13, 7, 13, -46, -17, 55, 50, 20, 94, -72, -46, 20, 15, 158, -41, -23, 50, 49, 5, 4, 5, -4, -18, -51, 30, 12, -1, 8, 18, 2, 3, -42, 6, -10, 44, -31, -145, -36, -29, 51, 2, -60, 66, -8, -44, -6, -21, -6, 58, 25, 14, 12, -11, 14, 3, 18, 50, 6, -59, -15, 0, -53, 39, -28, 10, -96}
, {-12, -48, -25, -154, 18, 3, -50, -113, -99, 21, 11, -124, -66, 179, -86, 111, -42, -44, 16, 74, 1, 15, -46, 17, -8, -89, -6, -89, 7, -37, 90, -38, -64, 26, -102, -153, -66, -18, 8, -26, 8, -2, 33, 45, -89, 10, 1, -17, 12, -1, -27, 65, -104, 13, 21, -204, -172, -6, -1, -41, 108, -18, 6, 38, -9, -22, -50, -52, -140, 3, 59, 25, 135, 14, -2, 67, -81, 74, 76, -56, -8, -85, 3, -44, -20, -49, 126, 1, -8, -3, 12, -2, -6, 72, -87, 46, -22, 70, 121, 51, 62, 4, 5, -76, -62, 38, -199, -21, -107, -5, 87, 60, 59, 8, -100, -40, 32, -200, -35, 233, 29, -94, -12, 94, -76, 93, 5, -10, -13, -219, 2, -90, 15, 17, 79, -342, 75, -14, -19, 142, -58, -1, -26, -148, -41, -28, 17, -58, 82, 11, 90, -4, -13, -26, -10, 15, -23, -48, -125, -76, 10, 8, -5, -93, 57, 19, -12, 35, -89, -8, -68, 64, -69, -13, -6, 13, 98, 2, -20, 27, 14, -73, -138, -28, -52, -55, -2, -122, 99, -91, -17, 33, -21, -373, -6, -89, 82, 0, 80, -82, 34, 9, -1, -56, -164, 11, 11, -39, -46, 62, -5, -43, 56, -88, -107, 14, -14, -121, 25, -27, -16, -50, -28, -158, -29, -7, -192, -126, -5, -19, 3, -10, -37, 44, 41, -5, -67, -18, 16, -28, -41, 20, -102, 9, -45, 19, 30, 86, 26, -20, -22, -52, 49, 50, 15, 37, -1, -214, 11, 84, -33, 16, 134, -63, 130, -6, -16, -60, 36, 94, -47, -44, -43, -15, 18, -25, 36, -36, -79, 14, -6, -57, 10, 25, 12, 58, 9, 65, 66, -30, 82, 14, -17, -68, -8, -49, 35, 10, -27, -133, 26, -6, 102, 51, 159, 17, 23, -19, -33, -24, 197, 31, 147, -49, 6, 6, -21, 80, 5, -148}
, {-20, -9, -11, -14, -22, -12, -1, -6, -6, -21, -5, -23, -8, -10, -5, -15, -9, -6, -2, 5, -1, -22, 2, 3, -14, -2, -3, -3, -3, -8, 4, -6, -16, -14, -7, -1, -19, 4, -4, -4, -10, 3, 6, -21, -19, 6, 4, -7, -7, -7, -5, 0, 3, -8, -17, -3, -5, -5, -20, -18, 4, -19, -4, -20, -7, -16, -4, -4, -20, -20, 11, -8, -15, -18, -7, 4, 6, 6, -17, 3, -3, 3, -14, -9, 0, -2, -22, -10, -10, 0, -3, -3, 0, 1, -1, -19, -6, 1, 3, -11, 3, 2, -21, -2, -7, -5, -22, 1, -17, -16, -8, -17, -7, 6, -8, -17, -1, 0, -1, -8, -16, -4, -18, -11, -8, -16, -4, -14, -3, -22, 1, -1, -21, -14, -9, -2, -21, -16, 4, -3, -13, -21, 7, -21, 0, -18, -22, -1, 3, -3, -12, -7, 2, -13, -8, 0, -10, -13, -1, -2, -15, -14, -15, 2, -4, -19, -20, -21, -14, -12, -4, -16, -4, -8, 6, -6, 4, 3, 2, -22, -2, 2, -1, -14, -5, -2, 6, 4, -6, -17, -1, 5, -7, -13, -13, -8, -13, -12, -6, -21, -13, -11, -21, -4, -15, 2, -14, -17, -1, -5, -21, 1, 2, -3, -11, -4, -5, -3, -22, -22, -8, -18, -16, 4, 4, -5, -16, -11, -4, -3, -2, -21, -1, 3, -10, -10, -22, -4, -5, -11, -14, 2, 0, 5, -1, -4, -2, -15, -8, -19, 4, -12, -15, -10, 1, -20, -18, -8, -5, -13, -14, 4, -13, 15, 0, 6, 1, -20, -21, -2, -6, -23, 0, -18, -6, -24, -13, 2, 4, -10, 3, 3, -6, -1, -1, -8, -16, -14, -16, -4, -7, -12, -12, -13, -21, -13, 5, -7, -13, -17, -12, -21, -11, -13, -5, 1, 2, -17, -18, -23, -1, -9, -1, 0, -7, -15, -21, -16, -13, 6}
, {-18, 4, -8, -2, -3, -5, 3, 7, -15, -1, -7, -21, -15, -23, 5, 0, -25, -11, 4, -14, -17, -23, -23, 4, 1, -9, 2, -20, -19, -13, -13, -7, 3, -8, -4, -8, -12, -18, 3, -17, -2, -13, -10, -4, 3, -18, 1, -21, 0, -20, -23, -5, -7, 1, -20, -1, -23, 0, -4, 3, -20, -10, -15, -9, -1, 5, -1, 6, -1, -14, -13, -4, -13, -10, 3, 3, -25, -7, -17, -17, -16, -2, -5, -22, -15, -9, -15, -22, -16, -12, -16, -7, -24, 2, -21, -12, -6, -11, -7, -10, -9, -16, -16, -15, -6, 1, -3, -15, -2, -18, -4, -3, -1, -22, -13, -25, -16, 4, -16, -20, -21, 1, -7, -14, 0, 5, 4, -11, -19, -11, 3, 4, -7, -9, -6, 6, -8, 3, -20, -14, -2, 4, 2, -10, -12, -18, -15, -22, -17, -13, 4, -20, -15, 5, -22, -1, 4, -8, -1, 0, -11, -16, -5, -8, -23, -16, 3, -14, -7, 1, -6, -5, -16, -2, 0, -22, -12, -10, -16, -23, -24, -18, 9, -11, -12, -9, 2, 3, -5, 5, 3, 2, 3, -9, -20, 3, 4, 3, -13, -21, 3, 5, -11, -23, 0, -1, -16, -24, 1, -1, -6, 3, -7, -24, -19, -14, -13, -22, -5, 1, -9, -1, 1, -10, 1, 11, 1, -9, -19, -13, 2, -21, 2, -16, -6, 2, -22, -10, -18, -9, 3, -3, -14, 0, -2, -7, -21, -23, -9, 2, 3, -19, -18, 5, -15, -8, -5, -2, -1, -5, -3, -21, 10, 6, -21, -7, -11, -14, -12, 4, -23, -10, -6, -10, -2, -25, -19, -17, -5, -11, -21, -22, -9, -17, -13, 2, -19, -14, 0, 7, 2, -18, -23, -25, -2, -20, -10, 3, 3, -18, 1, 1, -17, 3, -17, -23, -21, -17, -8, -23, 1, -3, -12, -20, 4, -13, 0, -5, -15, -18}
, {7, -18, -115, -9, -36, 4, -87, -151, 36, 8, 7, -7, -70, -31, -214, 0, -13, 61, 19, -143, -37, 1, -8, 10, -1, 17, -9, -15, 2, -111, 67, 61, -3, 45, 14, -54, -14, 42, 18, -65, 23, 7, 16, -117, -5, -30, 61, 47, -18, 14, 21, -148, -195, -18, 57, -50, 35, 20, 14, 19, 2, -26, -5, -162, -6, 47, -58, -93, 61, 5, -73, -69, 23, 15, -7, 99, -50, 45, -50, -77, 5, 59, -7, -14, 1, 15, 31, 9, 13, 110, -5, 50, 12, -77, -14, -13, -16, -62, 14, -67, 2, -54, 47, 5, 6, 11, 11, -73, 6, 12, 29, 4, -10, -2, 37, -49, -16, -42, 86, 45, 46, 5, 10, -39, -73, -31, 13, -66, 9, 36, -29, -123, -35, 7, -44, -47, -27, 17, -7, -150, -33, 15, -32, -42, 9, -87, 4, -163, 36, -54, 38, 12, 18, 51, -9, -18, 12, -19, -29, -37, -10, -187, -84, 31, 35, -8, -4, -25, -14, 19, 11, -126, 5, -1, -2, 25, 4, 10, -32, -19, -35, 64, 60, -21, 26, 0, -1, -207, 62, 138, 18, -14, -10, 19, -57, -137, -43, 9, -30, -12, 14, 8, -7, -34, -4, -10, -216, 13, 47, -100, 11, -104, -11, 51, -28, 21, 15, 57, 5, -90, -13, 38, -97, 31, 19, -148, 8, -29, -24, -44, 6, 12, 0, -4, -203, 101, -34, -3, 26, -18, -78, 18, 50, 65, -73, 27, 49, -57, -47, -4, -7, -27, 83, 62, 15, -42, -4, 30, -4, -89, 109, -3, -131, 19, 30, 18, -12, -66, -47, 34, -40, -3, 29, -122, -8, -87, 44, -35, 34, 8, 4, 75, 5, -7, 7, 40, -116, -6, 86, 62, -23, 57, -1, -40, 17, 20, 48, 20, 44, -117, -13, 0, -10, 42, 31, 0, -65, -50, -70, 133, 25, -28, -3, 7, 1, -196, 31, -123, 6, -45}
}
;

#undef INPUT_SAMPLES
#undef FC_UNITS
/**
  ******************************************************************************
  * @file    fc.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 january 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _DENSE_28_H_
#define _DENSE_28_H_

#ifndef SINGLE_FILE
#include "number.h"
#include <stdint.h>
#endif

#define INPUT_SAMPLES 128
#define FC_UNITS 64

typedef int16_t dense_28_output_type[FC_UNITS];

#if 0
void dense_28(
  const number_t input[INPUT_SAMPLES], 			      // IN
	const number_t kernel[FC_UNITS][INPUT_SAMPLES],  // IN

	const number_t bias[FC_UNITS],			              // IN

	number_t output[FC_UNITS]); 			                // OUT
#endif

#undef INPUT_SAMPLES
#undef FC_UNITS

#endif//_DENSE_28_H_
/**
  ******************************************************************************
  * @file    fc.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "dense_28.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#elif defined(WITH_NMSIS_NN)
#include "riscv_nnfunctions.h"
#endif

#define INPUT_SAMPLES 128
#define FC_UNITS 64
#define ACTIVATION_RELU

// For fixed point quantization
#define WEIGHTS_SCALE_FACTOR 7
#define BIASES_SCALE_FACTOR 7
#define TMP_SCALE_FACTOR 7
#define INPUT_SCALE_FACTOR 7
#define OUTPUT_SCALE_FACTOR 7
#define OUTPUT_ROUND_MODE ROUND_MODE_FLOOR
#define NUMBER_T int16_t
#define LONG_NUMBER_T int32_t


static inline void dense_28(
  const NUMBER_T input[INPUT_SAMPLES], 			      // IN
	const NUMBER_T kernel[FC_UNITS][INPUT_SAMPLES],  // IN

	const NUMBER_T bias[FC_UNITS],			              // IN

	NUMBER_T output[FC_UNITS]) {			                // OUT

#if !defined(WITH_CMSIS_NN) && !defined(WITH_NMSIS_NN)
  unsigned short k, z; 
  LONG_NUMBER_T output_acc;

  for (k = 0; k < FC_UNITS; k++) { 
    output_acc = 0;
    for (z = 0; z < INPUT_SAMPLES; z++) 
      output_acc = output_acc + ((LONG_NUMBER_T)kernel[k][z] * (LONG_NUMBER_T)input[z]);

    output_acc = scale(NUMBER_T, output_acc, WEIGHTS_SCALE_FACTOR - TMP_SCALE_FACTOR, OUTPUT_ROUND_MODE);

    output_acc += scale(NUMBER_T, (LONG_NUMBER_T)bias[k], BIASES_SCALE_FACTOR - TMP_SCALE_FACTOR - INPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);


    // Activation function
#ifdef ACTIVATION_LINEAR
    // Linear (MEANS NONE)
    output[k] = scale_and_clamp_to(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
#elif defined(ACTIVATION_RELU) || defined(ACTIVATION_RELU6)
    // ReLU
    if (output_acc < 0) {
      output[k] = 0;
    } else {
#if defined(ACTIVATION_RELU6)
      if (output_acc > scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE)) {
        output_acc = scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE);
      }
#endif
      output[k] = scale_and_clamp_to(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
    }
#else
#error "Unsupported activation function"
#endif
  }
#else

#if BIASES_SCALE_FACTOR > WEIGHTS_SCALE_FACTOR
#error "CMSIS-NN does not support BIASES_SCALE_FACTOR larger than WEIGHTS_SCALE_FACTOR"
#endif

  static q15_t bufferA[INPUT_SAMPLES];
#ifdef WITH_CMSIS_NN
  arm_fully_connected_q15(
#elif defined(WITH_NMSIS_NN)
  riscv_fully_connected_q15(
#endif
                             (q15_t*)input,
                             (q15_t*)kernel,
                             INPUT_SAMPLES,
                             FC_UNITS,
                             INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - BIASES_SCALE_FACTOR,
                             INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR,
                             (q15_t*)bias,
                             (q15_t*)output,
                             (q15_t*)bufferA);
#ifdef ACTIVATION_RELU
#ifdef WITH_CMSIS_NN
  arm_relu_q15((q15_t*)output, FC_UNITS);
#elif defined(WITH_NMSIS_NN)
  riscv_relu_q15((q15_t*)output, FC_UNITS);
#endif
#elif !defined(ACTIVATION_LINEAR)
#error "Unsupported activation with CMSIS-NN"
#endif


#endif
}

#undef INPUT_SAMPLES
#undef FC_UNITS
#undef ACTIVATION_RELU
#undef WEIGHTS_SCALE_FACTOR
#undef BIASES_SCALE_FACTOR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
#undef NUMBER_T
#undef LONG_NUMBER_T
/**
  ******************************************************************************
  * @file    weights/fc.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

#define INPUT_SAMPLES 128
#define FC_UNITS 64


const int16_t dense_28_bias[FC_UNITS] = {14, 67, 29, 27, -42, -42, 122, -9, -42, 30, 53, 139, 35, 67, 57, -64, 46, -13, 62, -31, -49, -16, 71, -12, 119, -11, 81, 74, 27, -31, -4, 65, 4, 42, 58, 66, 16, -12, -44, -12, -51, 80, -15, -22, 102, 8, -42, -9, -31, 5, -84, -23, 93, 70, -49, 52, 54, -49, 2, 79, 23, -3, 33, 51}
;

const int16_t dense_28_kernel[FC_UNITS][INPUT_SAMPLES] = {{10, 39, 3, -48, -19, 8, 3, -5, 39, -34, -14, 67, 33, -78, -42, -5, -26, 16, -6, 19, 30, 12, -10, -44, -23, 37, 45, -10, 6, -4, -3, 83, -41, -13, -31, 29, -11, -36, -16, -40, 45, 18, 10, 11, -35, 53, -2, 32, 18, -19, 17, 27, -11, -8, 15, 3, 105, 28, -67, 23, -13, -18, -23, -15, -18, 13, 5, -73, 6, 9, 6, 2, -51, -12, -6, -112, 28, -11, 64, 9, -60, 22, -19, -30, 28, -8, 9, 5, -5, -49, 15, 31, 27, 4, 1, 44, 14, -29, -78, -97, 6, -16, -36, 12, -4, 27, 1, 25, -43, -34, 8, 107, 26, 29, 14, 24, 101, 5, -195, -19, -2, -20, 123, -29, -168, 5, 11, -13}
, {25, -25, -59, 6, 36, 41, -4, -99, 12, -111, 18, -42, 50, 41, -10, 2, -32, -5, -99, -69, 26, 1, -25, 0, 4, -2, -31, 9, 32, -4, 21, -25, -66, 22, -6, 1, -45, -6, 12, 30, 67, 19, -26, 26, -11, 49, 26, 15, -20, -56, -2, 31, 4, -14, 11, 34, -14, 15, 11, 28, 16, -18, -23, -6, 16, -21, 51, 138, 16, 12, 64, -4, -3, -12, 7, -43, 15, -58, 56, -32, -14, 29, -26, 12, 68, 5, -8, 11, -9, -29, 22, -1, 68, -28, 6, 42, -6, 20, -49, -125, 27, 10, 55, -37, 8, 14, -44, 4, -51, -144, -66, 28, 0, -20, 39, -9, 23, -31, -25, 1, 28, 10, 65, 40, -163, -13, -15, -19}
, {-6, -13, 14, 14, -43, 4, 22, -1, 26, 74, -12, -47, -12, -30, 42, 12, 25, 29, 15, -30, -8, 0, -116, 20, -80, 25, -31, 4, 22, -96, 8, 18, -32, -18, 15, 10, -74, -41, 3, -25, 23, 13, 40, -3, 17, -54, -3, 8, 4, -45, 7, 94, -10, -11, -15, -21, 43, 25, 21, 38, -15, -23, -20, -15, -48, -2, -21, -2, 10, -13, -12, 11, 44, 27, 8, -60, 19, -38, -21, 5, -34, -8, -35, -66, 12, -4, 21, 22, 11, -52, -2, -102, 36, -101, -1, -38, -1, 83, -30, 28, 34, -16, -43, 22, 15, 10, 0, 27, 3, -48, -51, -34, 14, -44, -7, 10, -108, 40, -93, -28, 11, -22, 49, -18, 26, 25, 22, -24}
, {-18, 32, 3, 8, 11, 13, 14, -31, 121, -78, -6, 7, -3, -77, -58, -1, 46, 16, 22, -53, -14, -10, -103, 45, -30, 24, 31, 7, -50, 7, -4, 44, 83, 8, 40, -11, -41, -83, 25, 9, 9, 15, -87, 14, 41, -5, 24, -42, 1, -19, 5, 68, 1, 9, 24, 2, 9, -12, -17, 19, -12, 5, -38, -7, 25, 20, 65, -103, 20, 21, 75, 2, 34, -6, -30, -59, 2, -28, 24, -9, -58, 14, 37, -30, 3, -3, 23, -7, 25, 0, 1, -16, 24, 43, -19, -41, 6, -15, -54, -26, -23, -20, 28, -16, 32, -1, -42, 10, -19, -50, -8, 40, 16, -111, 7, -7, -16, -22, -53, 1, -4, 48, 33, 0, 93, 0, 25, -58}
, {2, -11, -104, -41, -40, -11, -20, -55, -59, 9, -24, -36, -22, -101, -17, 11, 36, -8, 46, -56, 0, 9, -8, -65, 25, 7, -16, -14, -37, -51, -28, -14, -30, -51, -39, -5, -44, -66, 2, -25, -167, -9, 74, 5, -63, 32, -30, 3, -4, -45, 5, 13, 3, 7, -2, 10, 5, -9, -7, -48, 10, 13, -65, 14, 26, -28, -100, 53, -6, -6, -62, 24, -61, 2, -1, -69, 5, -9, -17, 60, -38, 11, -22, -40, 9, -4, -18, -22, 14, -47, -6, -155, -46, -113, -17, -103, -2, 57, -42, -24, -9, -1, -22, -28, -29, -6, -37, -7, -12, -30, -69, -62, 13, -76, -13, 4, 17, 6, 58, 5, 4, -97, 4, -31, -8, 2, -8, -3}
, {-25, -6, -44, -28, -46, 0, -5, -1, -46, -28, 16, -49, -14, -20, -34, 8, 31, 8, 29, -77, 1, -9, -20, -75, 1, 10, -94, 14, 3, 9, 9, -58, -17, -21, -4, -4, -16, -58, -25, -10, -28, -22, -8, 7, -27, 15, 13, -11, -13, -3, -31, 1, -20, -5, 6, -54, -56, -25, -83, -20, 12, -28, -22, -5, -27, 1, 4, -14, -22, 0, 1, 4, -34, 7, -3, 14, -14, -30, -9, -49, 9, -10, -43, -10, 3, -1, -8, 14, 5, -18, -25, -12, -28, -64, -13, -39, -23, 42, -29, -69, -10, 1, 24, -20, -11, 10, -8, 5, -43, -32, -33, -10, -20, -6, -1, -14, -44, -61, -8, -1, -13, -33, -6, -27, 1, -9, -12, -6}
, {60, -16, -22, 18, 16, -34, -36, -9, -145, -37, 32, 20, 58, 27, 88, -16, 23, 22, -3, -175, -14, -20, -105, 82, -108, 14, 13, -15, 11, 80, -20, -47, -81, -15, -154, 14, -19, -30, -1, -3, -3, -34, -30, -20, -48, -40, -19, -27, 4, -143, 16, -127, 1, 11, -9, 19, -23, -21, 34, 8, 0, 47, -9, -7, -55, -7, -9, 4, -13, -7, -76, -14, 2, -8, 79, 41, 4, -1, 51, -85, 19, -19, 45, -33, -8, 28, -6, -6, 20, 38, 5, -21, -30, 29, 12, 11, 14, 51, -86, -144, -18, -13, 33, -28, 52, -5, -60, 12, 12, -32, 6, -81, -1, -61, -57, 12, 52, -20, -71, -36, -15, 40, 55, -12, -48, 0, -17, -139}
, {-12, -13, -30, 2, -40, 5, 13, -32, -18, 15, 0, 4, 8, -10, -22, -4, -18, -23, -30, -8, 13, -18, -23, -29, -17, -1, 3, 9, -14, -12, 21, -17, -8, -10, -24, -14, -47, -32, -12, -5, -21, -8, -20, 2, -28, -15, -27, 9, 17, -9, 5, 2, 7, -27, 3, -27, -15, -27, -46, -26, -22, 13, -32, -5, -26, -5, -13, 0, 8, -27, 7, 31, 0, 12, 13, -25, -11, -16, 9, -33, -37, -4, -21, -17, -24, -20, 12, 5, -12, -30, 11, -4, -1, -23, -14, -13, -3, -9, -16, 7, -4, -13, 10, -24, -5, 26, -20, 11, -2, -20, -9, 8, 11, -43, 0, -9, -27, -25, -26, 1, -21, -26, -24, -9, 22, 1, 13, -40}
, {-3, -14, -17, -19, -20, -12, 14, -78, 33, 41, 4, 12, -3, -25, -49, 2, -27, -13, -94, -43, -2, 6, -69, -97, -63, -23, -2, -27, -10, -37, -24, -69, -76, 51, -47, 10, -6, -32, 3, -30, -66, -30, -21, 14, -49, 19, -19, 12, -29, 7, 14, -40, -60, -25, -18, -62, 20, 7, -48, -1, -6, 5, -119, -19, -45, -1, -26, 73, -19, 13, 36, 23, -41, -20, -40, -11, 4, -24, -29, -40, -54, -10, 2, -54, -51, 5, -16, -18, 10, -20, 5, -67, 4, -27, -13, -48, 7, -37, -83, -39, -1, -6, -4, 14, -27, -8, 38, -20, -22, -60, -20, -33, -4, -59, 8, -19, -43, -114, -22, 4, 8, 18, 48, -43, -4, -23, -7, -21}
, {8, -13, -38, 86, 17, 28, 44, -14, -5, -89, 9, -63, -9, -150, 70, 10, -113, 18, 7, -12, -6, 8, -24, 74, 3, -2, 20, -1, 14, -50, 5, 18, -19, 26, -31, 29, -24, 20, -7, 0, -62, 12, -55, 5, 62, 32, 9, 7, -6, 11, 6, -219, -151, 18, 14, -78, 6, 4, -73, 69, 18, -18, -27, -10, -50, 21, -63, -35, 15, 13, 10, 79, -24, 9, 25, 8, -25, -16, -81, 28, 43, 21, -54, -12, -43, -7, 14, 15, 1, -93, 15, -54, 26, -48, -24, -65, -6, 36, 9, -81, -66, -18, -39, 27, -4, 5, -5, 19, 18, 30, -68, -74, -3, 28, 25, 25, -50, 17, -114, -1, -8, -70, -7, 17, 0, 23, 4, 34}
, {52, 56, 6, -36, 21, -35, -38, -27, -82, 31, 17, -50, -4, -105, 15, 0, 27, -13, 0, 29, -17, 14, -84, 45, 9, 17, 50, 1, -75, 3, 22, 5, -19, -74, -54, 3, 3, -5, -19, 16, -10, -18, 53, 13, -6, -27, -2, 91, 5, 17, -7, 11, 60, -19, -9, 31, 73, -1, 4, 6, 2, 32, -3, 1, -113, -8, -136, -121, -22, -14, -103, -12, -86, -4, 2, 18, -24, -31, 36, -61, 1, -13, 25, -6, -106, 10, -7, 1, -12, 10, -13, -52, -11, 39, 14, -110, 4, 14, -33, -9, -8, 13, -54, 15, -13, 1, 90, -20, 19, -19, -12, -24, 13, 8, -15, -10, 38, -4, -72, 13, 18, 1, -54, 21, -84, -13, -14, -44}
, {3, -5, 26, -178, -34, -1, 3, -70, 3, -85, 6, -85, -30, -41, 53, -11, -164, -27, -63, 2, 23, 13, -5, -70, 21, 68, -4, 16, 16, -7, -5, -44, -21, 37, -88, 2, 7, -12, 18, 57, 24, 21, -34, -26, 65, 10, -32, 41, -27, 19, 27, 56, -20, 14, 26, -12, -95, -9, -35, 43, 16, 2, 32, 1, -11, -29, -36, -39, 13, -9, -86, -27, -64, -24, 46, -1, -8, 24, -63, 31, -73, 19, -15, -57, -20, -11, 26, -6, 25, -47, 8, -35, -66, -55, -33, -31, 0, 16, 7, -89, -9, 20, 41, 27, -29, -24, 41, -1, -18, 13, -109, -29, 5, 17, 15, 24, 18, -92, 55, -26, -28, -56, 15, 38, -87, 11, 11, 23}
, {-16, 52, -88, -68, -29, 20, -15, 24, -39, 34, 17, 53, -51, -87, -36, -1, -6, -21, -74, -20, -12, 22, -34, -69, -32, 22, -7, 29, 51, -74, 10, -4, -59, -23, -83, 2, 8, -37, -2, 5, -54, -13, -17, 10, -81, 39, -8, -37, -16, 29, 8, 54, -116, -2, -2, 93, 35, -17, -86, -27, 3, 3, 19, -12, -39, -12, -19, -10, -5, 9, -57, -24, 20, 17, -39, -6, 27, -78, -85, -92, -61, -5, 23, -39, -81, -7, 19, 14, -6, -131, -10, 57, -9, -79, 15, -55, -26, 21, -11, 33, 30, -3, 69, -29, 17, 40, 38, 10, 8, -6, -15, 61, -12, 67, 2, -4, -29, -37, -19, 1, 24, -6, 1, -9, -20, -6, -9, 12}
, {-31, 44, -45, -19, 19, 19, 12, 0, -13, 30, -8, -11, -9, -55, -63, -20, -29, -55, -9, -41, 3, 6, 1, -107, 3, 77, -18, 15, -32, -41, -9, 42, -56, 36, -3, -27, 6, -21, 18, 32, 42, -38, -17, 8, 31, 2, -13, 19, -16, 35, -30, 23, -38, 4, -3, -47, 10, 15, 25, 25, -11, 34, 16, -15, 44, -16, -26, 67, 0, -15, -102, 131, 2, 5, -40, -37, -25, -20, 51, -9, 31, -9, -135, -25, -90, -24, 11, -20, 9, 7, -8, -12, 13, 19, 48, 83, 0, 58, 39, 38, 64, -27, 53, -17, 32, -8, -40, -7, -67, -67, -49, -42, 10, -66, 21, -20, 32, 52, 83, 7, -19, -41, 23, 65, 81, -9, -10, -78}
, {72, 43, 2, 2, 11, 0, -3, -18, 16, -11, -26, -2, -9, 0, -16, 16, -74, 11, -28, 14, 6, 3, -66, 3, 64, 14, 2, 30, 45, -30, 14, 12, -79, -13, 25, -13, 7, -84, -18, 6, -18, -14, 44, -11, 24, 66, 16, -28, -35, -52, 6, 17, -24, 19, 0, -45, -1, 2, 19, 59, 17, 39, 4, -9, 69, 11, -49, -3, 25, 21, -85, -51, -26, -11, -70, -50, 17, 5, -9, 12, -40, 7, 20, -81, -144, 6, 19, 24, -5, -21, 22, 60, 24, -8, 21, -103, -6, 6, 21, 12, -12, 17, -13, -5, -3, -16, -62, -6, -109, -22, -40, -35, 27, -34, 9, 16, -127, -25, 0, -14, -2, 47, 4, -34, -16, 1, -8, 8}
, {-24, -15, -23, -10, -26, 18, -13, -66, -6, -49, 15, -25, -17, -12, -19, -21, -19, -20, -7, -34, -9, 5, -38, -5, -58, 13, -26, -4, -41, -40, -22, -86, -13, -17, -27, 15, 2, -18, 13, -25, -29, 5, -92, 10, -19, -20, 3, 10, -14, -26, -11, -25, 5, -1, -23, -25, -60, -11, -35, -20, -10, 11, -36, 9, -32, 3, -48, -11, -27, -2, -27, -19, -40, -6, -25, -48, 3, 6, 7, -41, 2, -10, -12, -49, -5, -3, 6, -24, 3, -62, 6, -54, -3, -19, -21, -18, -10, -79, -23, 8, 0, 4, -25, -2, 12, 18, -20, -25, -8, -16, 0, 12, 7, -2, -13, -4, -33, -16, 0, -12, 5, 6, -26, -37, -24, -6, 4, -27}
, {83, -26, -65, 72, 53, 31, 38, 18, -30, 45, 22, -9, 4, -45, 26, 26, 26, 9, 19, 17, -11, 12, -63, 69, 3, 4, -60, 12, -80, 0, 16, 22, -39, 2, 76, 12, -17, -17, -8, 22, -104, 25, -115, 19, -22, 43, 17, 11, -32, 38, 7, 82, 35, 26, -3, -76, 36, -10, -51, 61, 4, 5, -1, 23, 43, 7, -153, -1, -8, 19, 17, 3, 14, 4, -61, -22, 24, 14, -72, 47, 16, 5, -31, 29, -86, -5, 1, -14, -14, -42, -6, 25, 1, 26, -17, -96, 4, 108, 25, 29, 69, -14, -10, 42, -8, -9, -7, 1, 39, -38, -30, -44, 1, 1, -12, -8, 19, -15, 5, 1, 23, 8, 18, 31, 102, -1, 28, 10}
, {-3, 9, 3, -22, 6, -29, -20, 77, 6, 35, 7, -34, -2, -37, -65, -1, 9, 7, 12, -24, 3, 2, -52, -24, -62, 14, -10, 12, 10, -18, 2, -26, -22, -40, 4, 12, 55, -31, 12, 5, 12, 3, 9, -30, -85, 4, -9, -23, -17, -15, -2, -64, -65, 0, -29, -71, -9, 6, -52, 0, 14, 3, -100, 14, 17, -15, 7, -14, -4, -8, 21, 28, 18, -10, -46, 39, -1, -64, 48, -50, -15, 13, -23, 25, -52, 1, 7, 0, 2, -41, -18, -62, -20, -35, -5, -10, -23, 63, -79, -7, -6, -3, -19, 13, 13, 17, -23, 10, -42, -34, -110, -6, 8, -7, 2, 9, 32, -127, -62, 3, 14, -76, -8, 6, 101, 5, 9, -12}
, {-2, 7, -30, 89, -24, -55, 35, 10, -14, -43, 8, 0, 47, -85, 39, -9, -61, 0, -2, -1, -11, 13, -78, 42, -37, 15, 13, -9, -5, -55, 12, 4, 7, -37, -70, 7, -49, 15, -11, 0, 44, -13, -9, 8, -11, -9, 4, 30, -14, 4, -15, -60, -41, -7, -6, 8, -13, 14, 41, 33, 18, 23, 31, 28, -50, 11, -28, 102, 21, 18, -1, -8, -3, -14, -11, 19, 1, 11, -23, -35, -6, 0, 53, -49, 39, 8, -14, -4, -9, 19, 12, -114, -32, -56, -12, 19, 19, -16, 7, -81, -42, -4, -8, -1, 13, -17, 43, 18, 47, 5, 42, -69, 21, 9, 22, -6, -69, 25, -114, 6, -15, -12, 13, 23, 36, 9, 28, 7}
, {-17, -1, -17, -17, -54, -11, -7, -25, -42, 14, -24, -18, -16, -5, -12, -1, -17, -21, -69, -34, -30, -16, -24, 12, -45, -17, -33, 12, -27, -47, -19, 1, -2, -31, -4, 9, -54, -26, 6, 0, -24, -7, -17, 3, -10, 9, 6, -16, -5, 10, -10, 60, -69, -2, 8, -44, -33, 1, -39, -6, -7, -25, -23, 8, -31, -10, -31, -8, -2, 8, 14, -11, -17, 3, -23, -92, 3, -68, -63, -15, -21, -28, 4, -27, -22, -23, -24, 6, -19, -22, -17, -26, 12, -67, -21, -29, -12, -31, -72, -12, 11, -9, 13, 12, -24, 9, -44, -19, -27, -3, -94, 12, 6, -9, -1, -15, -1, -39, -18, -24, 8, -4, 7, -27, -12, -24, -19, -17}
, {12, -10, 5, -51, -36, -45, -24, -35, 28, -126, 1, 21, -29, 33, 30, -2, 50, -3, -65, 1, 4, -24, -29, -57, 14, -5, -48, 9, -62, -9, -28, -33, 8, -63, -54, 5, -30, -78, -25, 10, 54, -27, -45, -26, -82, 0, -17, 6, 4, -54, -16, -98, 33, -30, 27, -23, -10, 13, -59, 57, -5, -32, -42, 11, -59, -6, -33, -58, -28, 12, 14, 9, -48, -18, -62, -95, -21, -100, 14, -27, -14, -15, -77, -45, -63, 11, -27, -25, -12, 60, -2, 8, -21, -153, -18, -55, 4, 59, -89, 50, -30, -30, -8, -19, 7, 12, 100, -14, -45, -83, -33, 36, 4, 4, -15, -30, 39, -95, -76, 13, 0, -72, 84, 58, -62, -1, 9, -25}
, {6, 22, -21, -30, -22, -3, -28, -12, -4, -23, -3, -4, 19, 0, 10, 26, -7, -10, -2, -2, -15, -2, -9, -4, -32, -11, -1, -9, 12, -20, -6, -23, 0, -4, -2, -23, -6, -1, 2, 29, 12, 25, -14, -28, -26, 16, -3, 13, -28, -17, 6, -18, -20, -15, 8, -20, 10, -12, -20, -10, 7, 2, -8, 8, -43, 2, -6, -12, -7, 11, -21, 16, -19, -8, -4, -20, -29, -5, -23, -30, -35, 2, -7, -4, -16, -24, 14, 11, 4, -3, -14, -5, -4, -3, -7, -15, 16, -17, -15, -18, 7, 7, 12, -5, 10, -13, -20, 14, -7, 7, -51, -20, 9, -31, -5, -4, 12, -29, 6, -18, 4, -24, -26, -32, 0, -24, 17, -28}
, {-14, 76, -2, 8, -36, -11, 7, -30, -48, -40, 27, -80, -33, -112, -15, -7, -104, -17, 5, -1, 4, -1, 23, 12, -21, 5, 31, -3, -28, -67, 8, -27, -92, 29, 44, -7, -6, -36, 17, 2, -4, -7, -5, 13, 1, 31, -14, -18, -1, -22, -5, 24, -58, 13, -8, 46, 56, -27, 36, 41, 25, -25, 4, 23, 16, -7, -88, -74, 5, -1, 37, -129, -36, -5, -49, -61, -4, -14, -18, -41, -26, -15, -25, -6, -6, -15, 6, 0, -7, -37, -11, -22, 12, -32, -23, 83, 13, -40, 25, -127, 18, 2, -20, -25, -17, -38, -47, 12, -21, -18, 13, -76, -17, 54, -16, 3, 7, 26, -39, -17, 8, 44, 35, 4, 8, 6, 16, 8}
, {6, 8, -73, 3, -44, 2, -2, -57, -36, 50, 5, -23, 11, 109, -42, 23, -8, 9, -52, -28, 22, -13, -22, -4, -27, 24, -6, -6, -42, -55, 22, -20, -25, -21, -30, 25, -24, -68, -7, 27, -6, 23, 77, 0, -35, -48, 8, -4, 14, 24, -6, -114, -52, -2, -3, -9, 19, -4, 0, -12, -31, 9, -25, -7, -3, 3, 4, 0, -5, -5, -19, 9, -38, 18, 14, 10, 4, 13, -63, 44, -32, 30, -9, -9, -15, 8, -22, 4, 7, -2, -8, 23, 28, -19, -19, -33, 14, -44, -20, -71, 57, 8, 17, -5, 18, -18, -15, -20, -19, -9, 0, -13, -19, -54, -18, -12, 12, -80, -118, -20, 10, -13, -6, -25, 18, 18, 14, -83}
, {50, 49, 25, 56, -7, -14, 28, -102, 39, 12, 19, -14, 18, -57, 18, 7, -81, 62, 21, 61, -1, -10, 79, 69, -5, -28, 60, -11, -124, -12, -10, 26, -57, 10, 19, -22, 19, -19, -28, 43, -8, 21, -142, 20, 3, 21, 44, -23, -38, -2, 2, 70, -66, 9, 21, 63, 41, -13, 12, -17, 6, -22, -54, 16, -47, -11, -76, 35, 13, 23, 80, 20, -70, -18, -61, 25, -12, 7, -35, 24, -1, 12, 33, 27, 71, -10, 24, 26, 16, -31, 3, -48, -66, 10, -4, -1, 15, 8, -5, -72, -20, 8, 10, 42, -2, -14, 41, 24, -7, 63, -11, -46, -1, 24, 20, 18, 19, 23, 5, -11, 20, 54, 17, -21, -74, -11, -13, 29}
, {11, -23, -35, -3, -108, -32, -16, 46, -31, 9, 13, -48, -11, -8, -17, 2, -52, -7, -16, 13, 10, 3, 9, 11, -52, 5, -48, -7, -13, -52, 11, -30, -32, -43, 0, -27, -81, 3, -12, 19, -28, 10, -23, -19, -7, -2, -4, -8, 7, 27, -17, -15, -87, -4, -26, -39, 40, -1, -62, -24, 14, -18, -86, 4, -90, -13, -5, -94, -2, -7, -29, 12, -23, 6, -25, 17, 10, -14, -45, -6, 22, -10, -25, 2, -96, -14, -25, 11, -2, 28, 10, -48, -5, -25, 10, 4, 2, -35, -25, 65, -3, -26, 12, -19, 5, -20, -37, 0, -61, -23, 24, -49, 12, -138, 2, -15, -15, -150, -22, 12, -24, -98, -4, 6, 47, 0, -5, -83}
, {43, 19, -75, 4, 15, 79, 8, 73, -33, 108, 27, 55, 29, 113, -13, -13, 15, -23, 67, -71, 7, 17, -72, -13, 10, -77, -61, -20, 80, -6, -1, -45, 39, -41, -31, -9, 6, -10, 16, 25, -40, 8, -8, 4, -12, -54, 36, 48, 52, -31, -5, -64, 11, -14, 14, -44, -67, -11, 28, 70, 9, -12, 43, 19, 2, 24, -80, -51, 13, -28, 48, 8, 14, 28, -36, -24, -38, -53, 17, 10, 14, 7, 46, -44, 15, 20, 6, 6, 9, 54, -5, 2, 12, 3, 20, -15, 2, 115, 6, 118, 42, -2, -18, -72, 27, -12, -27, -1, -5, -77, -1, 48, 12, -46, -18, -4, -38, 14, -51, -12, -10, -1, -33, 66, 129, 20, 13, -80}
, {47, 65, -51, -46, 32, 34, 19, -4, -108, 89, -16, 24, -8, 41, -29, -21, 4, -17, -4, -35, -4, -9, -98, 1, -2, 68, -43, -27, -92, 110, -2, -10, -16, 55, -105, 5, 8, -25, 3, 57, -13, 3, -47, -23, 30, 33, -40, 11, 6, -93, -19, -54, 61, -30, -6, -13, -66, -15, 6, 50, -27, -29, -3, 8, -7, 6, -56, -56, -25, -13, 41, -24, -16, -6, -24, -37, -24, -14, -6, -42, 22, -20, -127, 12, -44, -20, -11, 8, -14, 39, -10, 65, 5, 26, -8, -26, -26, 40, -15, -45, 78, 4, -22, 71, -10, -6, 52, -28, 9, 18, 6, 48, -18, -59, 17, -11, -13, -25, 33, 15, -24, -60, 21, 22, 58, -25, -19, -120}
, {10, 13, -69, -17, -92, -7, 21, 9, -70, -67, -30, -30, -16, 10, 4, 8, -42, 14, 58, -52, -5, -11, -87, 20, -81, 14, 24, 2, 5, -54, -13, -114, -79, -18, 2, 8, -76, -76, -25, -11, -26, 5, -56, -11, -45, -29, -23, 22, 5, -146, 7, -47, -82, -23, 3, 7, -80, -5, -94, -16, 13, -18, -35, -15, -109, -12, -80, 59, 10, -17, 134, -8, -13, -21, -91, 33, 9, -2, -38, 45, 57, -25, -40, 5, -98, -11, 13, 10, 2, -3, -15, 12, 18, -116, -8, -92, -2, -79, -2, 7, 5, 22, -54, 22, -11, 5, -32, -24, -103, -24, -96, -139, 6, -185, -7, 14, -49, -65, -21, -32, 6, -53, 44, 16, -42, -28, 1, -17}
, {-8, -16, -7, 13, -41, -33, -1, -9, -23, -17, 0, 67, -3, -12, 66, -9, -24, -31, 21, -37, -1, -2, -19, -16, -30, 25, -95, -22, -92, -29, 9, -16, -55, -54, -18, 8, -34, -27, -17, -33, -1, 10, -75, -13, -15, 7, 1, -23, -22, 5, 12, -20, -1, 11, -10, -44, -52, -9, -93, -25, -13, 3, -35, -11, -15, -10, -63, 20, 9, -4, -97, -16, 3, 0, -24, 62, -21, -42, -10, -65, -75, 8, -17, -32, -50, 14, 8, 9, -8, -117, 5, -94, -5, -28, -28, -61, 11, -22, -48, 7, 5, -2, -2, 6, -18, -4, -6, -19, -21, -79, -18, -80, -18, -47, -5, 13, 12, 14, -29, 11, 20, 7, 6, -18, -52, -22, 2, -9}
, {90, -31, -48, -94, -9, -41, 11, -57, 0, -62, 1, 40, -16, 54, -9, 20, -4, -17, 43, 40, -7, -8, -57, -4, 58, -47, 28, 4, -10, 56, -5, -4, 48, -35, -97, 24, -46, -18, -5, -3, -46, 3, -40, 28, -16, 3, 4, 0, 27, -23, -4, 44, 14, -11, 16, -14, 28, 14, -26, 67, -11, -2, -41, 20, -76, -11, 27, -71, 22, 7, -8, 5, -59, 8, -5, -51, 0, -60, 11, -24, 27, 2, 36, -66, 22, -25, 22, 2, -14, 31, -10, 50, 4, -5, -14, -101, 15, 16, -25, -18, 47, 2, 20, 45, 35, -1, 41, 18, 8, -24, 19, 56, -1, 32, -5, 2, 46, -11, -71, 10, -12, 33, 130, 15, -66, 25, 31, 26}
, {-5, -30, -69, 20, 42, 42, -11, 17, -48, 72, -14, 76, 12, -30, -57, -8, 50, -10, 40, -3, -3, 0, 15, 27, -104, 21, 45, -1, 1, 50, 22, -26, 14, -82, -124, -19, 50, -34, 15, -31, -38, -27, -71, -7, -16, 31, -3, 3, -7, 26, 12, -48, 2, -30, -7, -88, 22, -29, -25, -12, 10, -38, -31, -7, 11, 8, -46, -125, -30, -14, -37, -2, -11, -7, 22, -1, -44, -13, 40, 27, 30, -7, -95, 19, -97, 16, 17, -7, -24, 51, -23, 16, -61, 35, -17, -19, 1, 147, 38, -4, 53, -5, 42, 28, -42, 10, 5, -24, 28, -28, 30, 4, -28, -51, -3, 3, -16, 22, 29, -4, -4, -7, 232, 6, 4, -5, -11, -74}
, {-25, -17, -83, 18, -50, -18, 7, -49, -55, -43, 31, -63, 18, -27, -46, -9, 39, 6, -10, -31, 3, -13, -59, -19, -63, -1, 9, -1, 6, -79, -22, 67, -11, -4, -3, -11, -34, -40, -5, -9, -28, 15, 23, 4, -13, -42, 12, 5, -2, -18, -9, -14, -40, -13, -12, 1, 20, 13, -24, -42, -5, -22, -48, 28, -33, -27, -38, 15, 11, 21, -24, 6, -70, 3, -14, -28, -14, -76, 47, -73, -48, -3, -32, -82, 15, 5, 8, -14, 12, -41, 23, -24, 0, -73, 19, -42, 29, -40, -24, -71, 6, -8, 22, 24, 3, 9, -36, -13, -40, -54, -58, 24, -5, -51, -8, 2, -24, -65, -85, 27, -6, -46, -1, -23, 15, 5, 12, -102}
, {-68, 24, 33, -89, -90, -48, -4, 55, 3, -47, -6, -47, -45, -45, 4, -11, -69, -31, 33, 4, -5, 13, 5, -16, 28, 25, -40, -5, 48, -94, -20, 35, -3, -25, 110, 5, -93, -50, -9, 6, 34, -9, -61, 14, 21, -48, -1, 29, 11, -124, 7, -59, 44, 14, -13, 33, 9, -28, 56, 13, -14, 3, 25, -2, 13, 28, 20, 11, 13, 12, -25, -107, 18, 0, -37, -53, 0, -1, 70, -13, -87, -11, 18, -34, 68, 0, -3, -29, -26, 23, -7, -34, 9, -58, -15, -34, 12, -32, 15, -27, -25, -6, -16, 10, 1, 5, 32, -4, -72, -66, -65, 87, 21, 22, -6, -21, -12, -39, -76, -3, -3, 24, -59, 26, -104, 6, 14, 11}
, {20, -27, -27, 83, -9, -10, 17, -25, 6, -18, -32, -4, 57, -27, -21, -29, 25, -1, -9, -29, 0, 11, -72, -64, -56, 30, 2, 31, 42, -12, 0, 50, -13, 47, 37, 0, 40, -33, 13, -71, 63, 7, -51, 0, -34, 43, -39, -18, -21, -67, -5, -150, -20, -12, 5, -70, -47, 14, 2, -12, 9, 7, 22, -11, -9, -17, 6, 3, -20, -24, -107, -50, 66, -3, 10, 11, -19, 21, 57, -51, -2, -21, -19, 30, 16, -7, 3, -7, 9, 31, -15, -57, -58, 41, -5, -118, 24, -24, 35, -36, -39, -21, -16, -6, -11, 7, 7, 6, -11, -7, 8, -84, -6, -29, -4, -10, -105, 32, -70, -27, 13, -11, 33, 37, -67, -27, -24, -81}
, {28, -46, -38, -42, -107, -78, -11, 0, -51, -65, -2, 23, -65, 21, 49, 5, 35, -25, 57, 26, -28, -2, -9, -84, 69, 8, 38, -9, 47, -2, 49, 25, -55, -63, -207, -3, 29, -48, 14, 41, -53, -7, -22, 0, -39, -36, -1, -4, 29, -79, 4, 113, -31, 6, -1, -94, -98, 10, -11, -3, -1, -4, -90, -7, -6, -15, -55, 2, -10, -28, 45, -50, -34, 21, -76, -37, -2, -85, -4, -47, 6, -22, 21, -99, 36, -2, -5, -17, -30, -15, -21, 37, -57, -89, 32, -30, 6, 22, -21, -70, -4, -6, -60, 35, -2, 18, -7, -16, -37, -61, -87, -86, -18, -112, 6, -6, -66, -87, -68, -5, 2, -69, -90, 83, 91, -16, -14, -86}
, {-11, 28, -23, -5, 33, -5, 16, -61, 61, 52, -5, 37, 7, -53, -19, -14, 3, -8, -154, -35, 27, 27, -16, 12, -24, 5, -93, 29, -32, -40, 12, 18, -54, -37, -25, -2, -66, -111, -8, 24, -44, 8, -28, -7, -14, -45, 17, 17, 5, -45, 7, -81, -18, 5, 23, 17, -19, -14, -97, -18, 2, 3, -48, -13, -152, 8, -81, -29, 17, 14, -86, -23, -42, -10, -60, -15, -10, -125, -13, -60, -53, 24, 11, -32, -52, -9, 16, 22, -11, -2, 0, 43, 10, -28, 11, -24, 7, -19, -10, 35, 17, -21, 18, -1, -25, 19, 42, 18, 34, -39, -33, 40, 6, -52, 29, -7, 7, -62, -46, 13, -11, -52, -44, -70, -13, 16, -10, -53}
, {-7, -25, -39, -16, -61, 12, -25, -51, -42, -31, -28, -13, 5, -27, -5, 2, -20, -8, -20, -25, 8, -15, -4, -24, -3, -20, -30, 7, -21, 23, -12, -53, -35, 0, -5, 5, -22, -10, -5, 0, -17, -16, -66, 13, 5, -4, 13, -15, -15, -33, -4, -30, 1, 11, 11, -59, -8, 1, -16, -4, -17, -12, -67, 1, -37, -27, -55, 8, -18, 4, -36, 17, -13, -23, -44, -49, -20, -65, 20, -43, -73, 11, -46, -28, 21, -27, -16, -3, -3, -12, -18, -19, 1, -28, -30, -70, 12, -39, -78, -60, -20, -31, -2, 9, 4, -1, -11, -19, -32, -22, -16, -46, 13, -7, -25, 1, 0, -64, 45, 4, -16, -46, 0, -27, -55, -25, -9, -4}
, {4, -12, 3, -23, -42, -15, 21, -18, -20, -12, 11, -1, 8, -8, -24, 13, -38, 13, -24, -38, -3, 8, -32, -9, -27, 1, -12, 28, -17, -21, 1, -46, -56, -8, 19, -24, -36, -45, 3, -3, -74, 16, -10, 15, -28, 2, 25, -13, 11, -22, -11, -5, -27, -1, 6, -30, -11, -1, -35, -46, 28, -51, -27, 3, -28, -8, -17, -50, 6, -8, -20, -17, -39, 14, -32, -31, 25, -51, -5, -24, -29, -10, 1, -23, 16, 21, 26, 6, 25, -21, -13, -9, 13, -32, 2, -17, 7, -4, -7, 8, -14, 8, 29, 11, -13, 7, -12, 23, -48, -21, -4, -22, -15, 7, -13, -21, -11, -19, 15, 9, -21, 2, 9, -11, 26, 15, 3, -5}
, {2, 39, -147, -5, -22, 5, 14, -74, -82, -31, -27, -43, 14, -65, -61, -15, -60, 12, -14, -51, 3, -10, -52, 7, -52, -20, -52, -12, -13, -72, -15, -35, -11, 4, 3, -9, -78, -66, -23, -1, -1, -14, -47, 5, -48, -35, 13, -5, -33, -38, -16, -77, 72, 13, 5, 38, -53, 3, -29, -15, -12, 9, 8, 3, -60, 8, -23, -28, 12, 5, 3, -31, -27, -18, -64, 10, 5, -81, 63, -49, -48, 14, -48, -6, -73, -25, 10, 11, 4, -48, 11, -135, -9, 5, -25, -11, 10, 61, 11, -48, 8, -2, -32, -11, -12, -13, -103, 2, -36, -5, 40, 15, -15, -46, 3, 13, -8, 46, -15, -14, 9, -3, -10, 30, -26, 14, 13, -177}
, {-26, -22, -3, -12, -36, 3, -4, -54, -27, -34, 1, -39, 6, -41, -13, -19, -6, 10, 10, 13, -18, -7, -22, -17, -52, -9, -17, -6, 2, -13, 20, -86, -56, -68, -68, -6, -36, -62, -19, -21, 11, 1, -14, -6, -41, -40, -15, -5, -5, 4, 5, -8, 31, 7, -3, -31, 68, 1, 14, 62, 6, -24, -44, 10, -8, -3, -41, -36, -23, -24, -12, 16, -26, -14, -50, -84, 6, -99, -43, -103, 3, -19, -19, -64, -100, -19, -3, -17, -10, -38, -28, -66, 5, -126, 7, -19, -10, -26, -102, -22, -21, 7, -20, 8, -9, -19, 18, -24, -21, 95, -76, -75, -15, -27, -30, 5, 3, -40, -1, 12, 4, -13, -26, -88, 10, -7, -6, 5}
, {-30, 20, -36, -6, -34, 10, -3, -49, 61, -54, -8, -42, -12, -12, 34, 13, -19, 4, -12, -24, 17, 4, -59, -52, -34, -1, -1, 3, -115, 51, -25, -18, -86, 19, 44, 5, 25, -26, 4, -13, -19, 3, -50, -2, -44, 6, 10, -30, 13, 23, 4, 35, -147, 1, -14, -51, -25, 20, 40, 46, 5, 13, -37, 5, 12, -9, 25, 27, -9, 22, 105, -24, 71, -31, -76, -24, 4, -51, -31, 53, 8, -8, 24, 64, -27, -1, -15, 0, -7, 4, 3, -3, 49, 41, -13, -104, 15, 1, -40, -168, 87, 6, 31, 21, 38, -14, -18, 6, -20, -10, 20, -10, 29, -4, 14, 16, -42, 35, 18, -11, -16, -42, 37, 9, -107, 12, 13, 14}
, {7, -14, -16, -6, -45, -12, -10, -23, 9, 6, -23, -9, -9, -25, -24, -25, -23, 13, -11, -11, -11, -24, -47, -4, 8, -9, -18, 8, 11, -2, 27, -30, 1, -2, -9, 17, -11, -18, 1, -17, -10, 12, -25, -6, -5, 13, -11, 6, -6, -4, -18, -37, -15, -13, -3, -32, -6, 14, -33, -28, -28, -17, -14, 2, -28, -14, -39, 18, 7, -12, -1, -4, -16, 14, -2, -27, -4, -36, 5, -44, -23, 1, -32, -26, -14, -3, -26, -21, 14, -19, 13, -8, -5, 5, -11, -15, -12, 4, -51, -20, -28, -34, 3, -7, -23, 7, -15, 11, -33, -11, -21, -20, -14, 0, -8, 6, 11, -42, -14, 6, -26, -4, 12, -27, 25, 4, -29, -40}
, {-24, 13, -42, -8, -56, -6, -26, -11, -32, -11, 12, 11, -29, -22, -54, -11, -49, -30, -65, -60, 10, -12, 21, -17, -57, -8, -95, 12, 10, 22, 2, -2, -24, -40, -17, -12, -4, -39, -30, -11, 7, -7, -22, 7, -39, -40, -6, -12, 15, -11, -14, -9, -47, 14, -26, -60, -73, 10, 4, -42, -8, -29, -63, -23, -51, -6, -1, -21, 13, -3, -34, -7, 3, -14, -29, 3, -29, -122, -36, -98, -40, -28, -54, -31, -38, 9, -29, -25, 12, -23, -27, -47, -28, -95, -19, -33, 1, 6, -92, -102, 12, -30, -2, 4, -10, -4, -29, -1, -38, -16, -14, 1, -2, -25, -11, -17, -38, 8, -4, 4, 13, -18, -13, -26, 7, 8, -4, 6}
, {4, 5, 22, 11, -58, 26, 12, -16, 19, 76, -9, -78, 18, -114, 36, 24, -134, -4, -25, 4, 1, -11, 10, 38, -45, 14, -3, 22, -30, 15, -13, 12, -104, 4, -107, 27, 16, -31, 5, 4, 65, 14, -13, -1, -28, 95, -4, 21, -25, 37, 28, -52, -163, 1, 15, -125, -12, -10, -28, 29, 4, -31, -40, -10, -33, 12, -55, -40, 3, 15, 67, -73, 31, -6, -13, -15, -9, 15, -43, 43, -30, -1, -24, 51, -11, 5, 23, 27, 26, -42, 14, -5, 56, 32, 15, 10, 10, -50, 1, -103, -63, -8, -25, -12, -18, -19, -6, 0, -5, 13, -35, -69, -2, -3, -7, -1, 0, 19, -144, 14, -12, -81, 1, -28, -129, 25, 6, 19}
, {-15, 57, 64, 65, -15, -15, 32, 51, -19, -138, -5, -13, 32, -162, -66, 25, -34, -19, -58, 1, -14, 24, -23, 36, 10, 8, 41, 11, 7, 6, 9, -8, -5, 42, 67, 20, 9, -32, 20, 17, 68, -10, -23, -10, 61, 64, 19, 15, -1, 27, -5, -4, -38, -13, 15, 4, -42, 27, 3, 79, 29, 25, -15, 26, 3, -20, 7, 80, -7, 22, 56, -49, -58, -9, 41, -67, 9, 19, -76, -53, -68, 16, 8, -37, -65, 20, -15, 20, -15, -28, 9, 3, -6, -18, -1, -1, 11, 28, -36, 4, -48, -32, -3, -19, 8, 5, -71, -6, -57, -15, -2, -31, -2, -18, 7, -8, -95, -21, -73, -17, 14, -68, 22, 11, -45, -15, 23, 44}
, {4, 11, -30, -18, -27, -1, 4, -10, -13, -8, -11, -24, 7, 7, -29, -25, -36, -20, -20, -24, -16, -1, -22, -3, -19, -3, -11, 3, 4, -32, 1, -62, 17, -8, -4, 11, 10, -27, -19, -15, -7, -24, -8, -27, -3, 2, 10, 1, -8, -22, 18, 12, -34, 4, -16, -44, -21, -6, -26, -40, -4, 16, -55, -4, -62, 22, -46, 5, -19, 20, -13, 6, 12, -3, 1, -18, 2, -22, -15, -18, -18, -3, -41, -68, -40, 11, 19, -28, 0, 1, 9, -19, -28, -23, -1, -4, -19, -5, -32, -42, -14, -15, -22, 0, -4, 17, -44, 1, -17, -17, -10, 3, -13, -27, 20, 21, -27, -29, -12, 2, -4, -24, 14, -24, -19, -20, -15, 4}
, {12, -22, -3, 7, -10, 17, -2, -24, -18, -7, -10, -11, -1, 10, -25, 20, -12, 17, -42, -27, -19, 17, -48, -17, -6, 21, -8, -23, 2, -36, -13, -23, -15, -11, -10, -8, -15, -29, -7, -10, 15, -10, 24, 13, -35, -19, -10, 3, -18, 2, 20, -3, -13, -7, -12, -27, -6, 14, -32, 7, 5, 13, -5, -5, -50, -14, -14, -17, -21, -1, -3, -55, -18, 14, -25, 17, -11, -7, 12, -18, -18, 0, -32, -19, -34, 8, -5, -18, -20, -33, -14, -35, -20, -8, -22, -18, 7, 7, -9, -12, -6, -10, 8, 20, -3, -1, -23, 21, -15, -59, -47, 10, 12, -30, -17, -5, 4, -10, 11, 12, -19, -1, 18, -21, -6, -19, -27, -17}
, {38, -16, 24, 3, -2, 18, 27, -132, -15, -31, 22, -19, -34, -94, -23, -4, 20, 15, -34, -56, 25, 8, -63, -71, -6, -37, -49, 19, 48, 38, -10, -68, -38, 38, 26, 14, 29, -22, -14, -5, -22, -7, -55, 7, -20, -89, -14, 8, -2, -43, -11, 50, 13, 10, -13, 16, -11, 21, 15, 7, 17, 13, -56, 2, -84, -6, 8, -47, 10, 22, 5, -19, -86, 22, -15, -113, 17, 1, -70, -28, -37, -6, 7, -74, -89, 18, 3, 7, 14, 12, 11, -107, -15, -88, -4, 27, 17, 8, -50, -49, -7, -4, 5, 34, -9, -19, -52, 24, -69, -18, -89, -70, -2, -31, 3, 0, -69, -100, -24, 20, 8, -11, 30, -9, -38, 21, -9, 1}
, {0, 30, -28, -51, -12, -64, -9, -111, -77, -43, -7, 87, 2, -43, 7, -20, 23, -13, 19, -57, -27, -28, 5, 17, -51, -17, -37, 9, -32, 53, 36, 25, 80, 17, -116, -11, -57, -125, 17, -55, -18, 8, 18, -26, -24, 12, 4, 18, 18, -43, 10, -44, 13, 0, 5, -83, -31, -15, 41, -87, -19, 14, -35, 1, -22, 1, -1, 7, -8, -27, -26, -34, 35, -26, -36, -7, 7, -99, -40, -70, -91, -6, 0, 13, 8, 12, 9, 7, -23, 2, -20, -18, 79, 47, 3, 42, 3, -55, 25, 46, -25, 11, 27, 14, 15, 6, 68, 9, -25, 6, -131, 97, 3, -53, -11, -27, -98, -132, 20, -10, -9, -81, -93, -11, -77, -4, -9, -56}
, {-5, -12, -70, -17, -48, -7, 18, 18, -35, -11, 18, -64, -27, -3, -33, -26, -18, 5, -16, 9, -9, -1, -44, 17, -80, -3, -73, 8, 27, -71, -2, -21, -15, -30, 9, -26, -16, -59, 16, -8, -83, -11, -5, -11, -49, -18, 0, -8, 21, -50, -1, -14, -9, -3, 8, -40, -18, -15, -100, -48, -10, 16, -14, -3, -35, 8, -86, 36, -30, -20, 28, 10, -20, -17, 19, -24, 4, -21, -16, -53, -46, 18, -27, -88, -7, -16, -12, -8, -16, -89, -12, -4, -5, -45, -3, -28, -26, 15, -28, -34, -16, -7, -24, -20, 12, -6, -42, -6, -33, -49, -61, -11, 24, -41, 26, 6, 21, -49, 9, -17, 7, -137, -26, -85, 1, 4, -9, -17}
, {-6, -11, 12, -46, 64, -25, -18, -43, -36, 10, 9, -103, 8, -47, -65, -6, 49, -43, -64, 11, 12, -25, -79, -59, -9, -10, -51, 0, 2, 3, -6, 26, -63, -38, -25, 0, -73, -93, -27, -7, -14, -13, -21, -18, -51, -59, 7, -14, 9, -33, -30, 41, 1, -30, 18, -84, -33, -30, -27, -37, 4, -2, -85, -3, 29, -8, -47, -40, -13, -14, 32, -87, 40, -1, -38, -103, 4, -53, -20, -51, -52, 2, -40, 19, -46, -22, 0, -21, -27, 38, -9, -32, -30, -39, -5, 17, 14, 24, -64, 29, -30, -15, 7, 19, -20, -20, -68, 7, -88, -85, 23, 74, -29, -30, -20, -21, -42, -45, -106, 3, -20, -118, 60, -55, 35, -28, -14, -67}
, {-13, -6, 7, -30, -13, 21, 23, -116, -35, -5, -7, 14, -41, 0, -13, 18, 42, -13, -4, -19, 19, 6, 11, 51, 53, 41, 40, 6, 3, 24, -28, -58, 8, -24, -75, 14, -17, 19, -19, -36, 5, 16, 9, 17, 13, -130, -22, -53, -2, -78, 17, 84, -15, -15, 13, -31, 82, -2, 69, 21, -6, -24, -64, 8, -15, 3, -22, 7, 7, -13, 41, -93, -8, 9, 13, -13, -13, -60, 29, 52, 36, 2, 23, -48, 6, 11, 4, 16, 21, -4, 7, 34, 49, -50, -39, -7, 8, 29, -44, -67, 119, 6, -26, 10, 30, -5, -3, -19, -147, -61, -6, 22, -6, 19, 29, -6, 62, -45, -143, -4, 12, 51, 80, 2, -14, -1, 9, 51}
, {69, -16, -28, 14, 1, -164, -18, -22, -26, 91, 8, 61, -60, -17, 34, 12, -64, -1, 20, 26, -28, -28, 64, 28, 38, -38, -67, 18, 114, 12, -32, -61, 2, 31, -58, 1, 5, -31, 19, -5, 0, 13, 22, -24, -1, -23, 10, -6, 14, -1, 16, 44, 31, -8, 15, -57, 35, 6, -19, -14, 10, 14, -63, -10, -14, 10, -114, -74, -4, -7, -50, 18, -18, 3, -117, -35, -2, -21, -77, 52, -4, -20, 52, -50, -5, -21, -1, -23, 24, -68, 4, -84, -50, 24, -17, -3, -27, 16, -7, 57, -55, -24, -15, 38, 7, -12, 17, -1, -46, 18, -50, 59, -14, 19, 49, -23, 52, -49, -69, 4, -8, -38, 58, -8, 81, -25, -11, 13}
, {-16, -3, -30, 25, -118, -34, 13, -25, -35, -52, 6, -38, -19, -22, 14, -23, -40, -8, -81, -41, 9, 11, -49, -33, 70, -22, 18, -4, -46, 62, 3, -8, 47, -39, -81, 10, -47, -29, -20, 13, -74, 4, -56, -19, -16, -27, 3, 3, -8, -53, -20, -97, -43, -14, 11, -68, 131, -23, -24, -11, -3, -12, -77, -10, -91, 19, -11, -15, -2, -25, -19, 8, -95, -8, -3, -103, 3, -38, -56, -112, -78, 8, -16, 13, -138, -9, 16, -28, 8, -3, -32, -77, 4, -7, 8, -34, -3, -15, -63, 3, 14, 6, 5, 6, 16, 16, -57, -8, -159, -65, -10, 14, -5, 8, -1, -12, -35, -29, -22, 6, -2, -54, -7, 8, -105, -3, -30, -68}
, {-7, 32, 68, -37, -69, 14, -8, 19, 82, 40, 26, -18, -5, 58, 22, 17, -15, -19, 10, -46, 9, 29, -45, -133, 40, -4, -38, -6, 127, -98, -29, 71, -16, -45, -103, 21, 23, 7, -16, -1, -17, 12, 78, -3, 21, -113, -8, -17, -1, -96, 22, -52, 26, 1, -34, -84, -51, -11, 40, 18, -15, 78, -7, 18, 5, -10, 26, -4, 8, -16, 4, -51, -48, -17, 10, 28, -7, -20, -50, 16, 40, -11, -3, -142, 37, 7, 1, -14, -6, -68, -3, -65, -23, -1, 4, -60, 23, -28, 9, -5, -59, 8, -20, 9, 10, -28, 45, -2, -55, -27, -83, -16, 3, -64, -14, 2, -54, 18, -118, -7, -21, -33, 17, 34, 72, 0, 16, 0}
, {-10, 21, -32, -79, -63, -23, -8, -36, -69, -101, 17, -90, 15, -94, 70, -18, -35, -12, -50, 11, -26, 2, 27, 6, 12, 25, -34, -27, -66, 43, 35, 33, -61, -29, 97, -29, 89, 20, -7, 7, -41, -4, -30, -22, 11, -39, -11, 14, -4, 5, 10, 84, -60, 10, -10, 10, -56, -22, -25, -50, -4, 23, -1, -23, -3, 9, -8, 42, -5, 26, -55, 1, 5, 24, 64, 20, -21, 30, -32, 60, 31, 8, 16, -35, 7, -19, 3, -15, -23, -42, -30, 19, -11, 11, -21, -56, -29, -48, 3, -7, 66, -4, -26, -18, -1, -10, -57, 3, 26, 8, -29, -92, 10, -3, -38, 10, 68, -22, 58, -8, -14, -31, 27, 12, -10, -21, -6, 68}
, {-21, -20, 7, 10, -84, -11, 12, -16, -67, 86, -5, -157, 7, -93, 35, -26, -61, -6, -19, -2, -21, -19, 10, 5, -24, -19, -11, -2, -1, -55, -2, -45, 39, -69, -11, -9, -14, -112, -30, 40, -42, -23, -28, -22, -72, 0, 3, -24, 5, 2, 7, -19, -61, 2, -3, -29, 41, -10, 6, 4, -17, -24, 13, -24, -90, 13, -72, -25, 7, -31, -21, -20, 7, -13, -37, -89, 4, -93, -47, -55, -40, -1, -57, -84, -76, -16, 15, -13, -19, 5, -25, -94, -19, -11, 9, 39, 7, -22, -25, -96, -8, -19, 3, 10, -24, -24, -49, -2, -57, -12, -61, -130, -17, -49, -24, -22, -35, 1, 12, 5, -18, -57, 12, -46, -54, -24, -11, -42}
, {-21, 17, -94, -23, -52, 8, 0, -14, -33, 111, 6, -78, -16, -29, -25, -17, -78, -14, -115, 8, -1, -4, -58, -46, -19, -3, 29, 7, -48, 67, 14, -7, -68, 9, -32, 12, 31, -45, -3, -1, -17, -27, -38, 7, 5, -21, -4, -14, -3, -58, 1, -75, -105, -8, 9, -54, 61, 14, -57, -10, 4, -2, -53, -8, -65, 12, -34, -23, -18, 14, -66, 21, -12, 22, 15, -94, -11, -19, -61, 10, -128, 14, -2, -11, -56, -2, 8, -14, -9, 35, -27, 44, -19, -86, -26, 41, -15, 11, -84, -13, 12, -21, 5, -12, -1, -4, -98, -18, -23, -25, -84, -20, 3, -42, -3, -7, -47, -15, -61, -23, -13, -10, -28, -44, -48, -10, 6, -123}
, {54, -51, -30, -15, -25, 23, -31, -24, -26, 23, -18, -51, 3, -34, 15, 4, 32, 20, -14, -5, 25, 14, 21, 12, 2, 4, -70, -12, -67, 56, -4, -19, 96, 3, -19, -17, 34, 17, -14, -23, 34, 2, 73, 1, -22, 23, -1, 7, -56, -59, -21, -42, 22, 14, 1, -51, 9, 19, -55, 49, -8, 22, -26, -7, -8, 17, 39, -139, -10, -16, -7, -189, 57, -1, -69, -62, -30, -39, -53, 52, 12, -11, 19, 49, -160, 16, 10, 10, -3, 6, -4, 81, -36, 35, -5, -27, 12, 34, 14, -5, 102, 13, 95, 35, 4, 15, 12, 12, -82, -21, -30, 37, -8, -85, 13, 11, 84, -45, -65, 30, 10, -24, -110, -17, -4, 22, 8, -27}
, {0, -37, -181, 75, 39, -43, 50, 69, -43, -2, 18, -5, 37, -55, 34, -2, -50, 37, 33, -44, 26, 22, -49, 51, -74, -38, 61, 14, 24, -5, 5, 21, -90, -101, -7, -27, -1, -6, -4, 22, -114, 10, 0, -8, -45, 47, 5, -40, -11, 26, 24, 89, -56, 24, 3, -46, -24, -3, 19, 28, 19, -7, 12, 29, -65, 4, -55, -32, 10, 13, -89, -10, 5, 0, -14, 10, 27, 12, 4, -29, 27, -3, -29, 23, 0, -17, 16, 24, 5, 50, 20, -78, 8, 44, 15, -66, 25, 14, 22, 29, -4, 5, -5, 13, 14, 3, 16, 5, 0, 67, 41, -86, 13, -60, -21, -6, -145, 33, 9, 9, -6, 25, 165, 50, 148, -12, 21, 21}
, {2, 4, -43, -7, -58, -13, 9, -25, -1, 0, -20, -54, 14, -36, 12, -1, -57, -4, -112, -38, -11, -17, -4, -23, 11, 6, -4, -16, -4, -10, -13, 20, 13, -26, -7, -14, 13, 18, -22, 22, 20, 8, -61, -6, -18, -20, -8, -24, -7, -24, -28, -27, -14, -23, 14, -45, 138, -13, 3, 16, -20, -12, -48, 4, -24, 1, 0, -28, 6, 12, -39, -34, -22, 3, -3, 46, -26, -62, -41, -15, -4, -24, 76, -10, -17, -5, -8, 3, 7, -8, -13, -77, -7, -13, -18, -30, -9, -44, -54, -16, -2, -11, -48, 14, -13, 13, -19, -27, -44, -89, -74, -84, -6, -14, -26, -15, -46, -78, -52, -7, 0, 42, -19, 14, -14, 4, 8, 0}
, {16, -4, -53, 31, 7, -24, 27, 13, 6, -2, -13, 54, -25, -67, 55, 1, -84, 7, 41, -42, 16, -6, -10, 41, -36, -2, -48, 4, 54, -23, -7, 54, 21, -41, -50, -21, -8, 19, 30, 6, -94, 6, 38, 21, -84, 62, -12, -15, -4, -6, -12, 78, 56, 22, -2, -18, -60, 19, -8, 54, -12, 15, -38, 20, -18, -4, -27, 22, -10, 14, 42, -52, -23, 5, -54, 23, 1, 11, -52, -4, 21, 3, 76, -2, 7, -1, 18, 1, 25, -75, 2, -61, 51, 21, 13, 44, -4, 30, -16, 12, 27, 4, 18, 30, 51, 9, 14, 1, 2, 48, -42, 3, 27, -97, -2, 26, -35, -9, -51, 17, 19, -99, 61, -1, 145, 17, 10, 28}
, {40, 14, 63, -88, -73, -40, 5, -100, 25, 48, 6, -142, 22, -75, 21, 8, -46, -5, 6, -17, 0, 14, 57, -67, 18, -5, 23, -13, 93, -79, -8, 10, -8, 29, 27, -7, 107, -46, 13, -11, 61, 5, 46, 26, 31, 5, -4, 5, 9, -7, -3, 20, 8, -7, 13, -97, -41, 10, -1, -48, 6, -4, -27, -17, -25, 22, -11, 41, -22, 11, -89, -74, 56, -1, 3, -36, -25, -34, -113, 25, -13, -25, -21, -23, -9, 26, -6, -21, -11, -93, -20, -65, 32, 21, 3, -38, 8, -44, -4, -19, -84, 4, -19, -3, 7, 4, 85, -4, -76, -9, -27, -15, -17, -31, -1, 1, -2, -32, -102, 19, 5, -36, 3, 56, 9, 11, 7, 28}
}
;

#undef INPUT_SAMPLES
#undef FC_UNITS
/**
  ******************************************************************************
  * @file    fc.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 january 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _DENSE_29_H_
#define _DENSE_29_H_

#ifndef SINGLE_FILE
#include "number.h"
#include <stdint.h>
#endif

#define INPUT_SAMPLES 64
#define FC_UNITS 10

typedef int16_t dense_29_output_type[FC_UNITS];

#if 0
void dense_29(
  const number_t input[INPUT_SAMPLES], 			      // IN
	const number_t kernel[FC_UNITS][INPUT_SAMPLES],  // IN

	const number_t bias[FC_UNITS],			              // IN

	number_t output[FC_UNITS]); 			                // OUT
#endif

#undef INPUT_SAMPLES
#undef FC_UNITS

#endif//_DENSE_29_H_
/**
  ******************************************************************************
  * @file    fc.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "dense_29.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#elif defined(WITH_NMSIS_NN)
#include "riscv_nnfunctions.h"
#endif

#define INPUT_SAMPLES 64
#define FC_UNITS 10
#define ACTIVATION_LINEAR

// For fixed point quantization
#define WEIGHTS_SCALE_FACTOR 7
#define BIASES_SCALE_FACTOR 7
#define TMP_SCALE_FACTOR 7
#define INPUT_SCALE_FACTOR 7
#define OUTPUT_SCALE_FACTOR 7
#define OUTPUT_ROUND_MODE ROUND_MODE_FLOOR
#define NUMBER_T int16_t
#define LONG_NUMBER_T int32_t


static inline void dense_29(
  const NUMBER_T input[INPUT_SAMPLES], 			      // IN
	const NUMBER_T kernel[FC_UNITS][INPUT_SAMPLES],  // IN

	const NUMBER_T bias[FC_UNITS],			              // IN

	NUMBER_T output[FC_UNITS]) {			                // OUT

#if !defined(WITH_CMSIS_NN) && !defined(WITH_NMSIS_NN)
  unsigned short k, z; 
  LONG_NUMBER_T output_acc;

  for (k = 0; k < FC_UNITS; k++) { 
    output_acc = 0;
    for (z = 0; z < INPUT_SAMPLES; z++) 
      output_acc = output_acc + ((LONG_NUMBER_T)kernel[k][z] * (LONG_NUMBER_T)input[z]);

    output_acc = scale(NUMBER_T, output_acc, WEIGHTS_SCALE_FACTOR - TMP_SCALE_FACTOR, OUTPUT_ROUND_MODE);

    output_acc += scale(NUMBER_T, (LONG_NUMBER_T)bias[k], BIASES_SCALE_FACTOR - TMP_SCALE_FACTOR - INPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);


    // Activation function
#ifdef ACTIVATION_LINEAR
    // Linear (MEANS NONE)
    output[k] = scale_and_clamp_to(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
#elif defined(ACTIVATION_RELU) || defined(ACTIVATION_RELU6)
    // ReLU
    if (output_acc < 0) {
      output[k] = 0;
    } else {
#if defined(ACTIVATION_RELU6)
      if (output_acc > scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE)) {
        output_acc = scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE);
      }
#endif
      output[k] = scale_and_clamp_to(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
    }
#else
#error "Unsupported activation function"
#endif
  }
#else

#if BIASES_SCALE_FACTOR > WEIGHTS_SCALE_FACTOR
#error "CMSIS-NN does not support BIASES_SCALE_FACTOR larger than WEIGHTS_SCALE_FACTOR"
#endif

  static q15_t bufferA[INPUT_SAMPLES];
#ifdef WITH_CMSIS_NN
  arm_fully_connected_q15(
#elif defined(WITH_NMSIS_NN)
  riscv_fully_connected_q15(
#endif
                             (q15_t*)input,
                             (q15_t*)kernel,
                             INPUT_SAMPLES,
                             FC_UNITS,
                             INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - BIASES_SCALE_FACTOR,
                             INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR,
                             (q15_t*)bias,
                             (q15_t*)output,
                             (q15_t*)bufferA);
#ifdef ACTIVATION_RELU
#ifdef WITH_CMSIS_NN
  arm_relu_q15((q15_t*)output, FC_UNITS);
#elif defined(WITH_NMSIS_NN)
  riscv_relu_q15((q15_t*)output, FC_UNITS);
#endif
#elif !defined(ACTIVATION_LINEAR)
#error "Unsupported activation with CMSIS-NN"
#endif


#endif
}

#undef INPUT_SAMPLES
#undef FC_UNITS
#undef ACTIVATION_LINEAR
#undef WEIGHTS_SCALE_FACTOR
#undef BIASES_SCALE_FACTOR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
#undef NUMBER_T
#undef LONG_NUMBER_T
/**
  ******************************************************************************
  * @file    weights/fc.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

#define INPUT_SAMPLES 64
#define FC_UNITS 10


const int16_t dense_29_bias[FC_UNITS] = {4, -20, -15, -31, 21, 29, 9, 62, -53, -10}
;

const int16_t dense_29_kernel[FC_UNITS][INPUT_SAMPLES] = {{51, 1, -29, -81, -50, 21, 26, 19, 13, 33, -26, 52, -36, 29, 8, 10, -1, 27, -13, 3, 10, -7, 22, 17, 29, -8, -72, -36, 105, -20, -14, -29, -18, -27, -30, 1, -6, -1, -44, 1, -61, -18, 19, -73, 48, 14, -44, -11, -44, -147, 7, -38, 4, 62, -19, -49, 28, 55, -47, 26, -47, -18, -31, 82}
, {14, -21, 26, 30, 11, -31, -5, -13, -14, 47, 10, 3, 24, 0, 13, -5, 23, -13, 10, -18, 6, 40, 18, -3, 21, -14, -13, -26, -5, -6, 21, -7, -8, -15, -23, -83, -12, 11, 14, -25, -39, 21, -2, -10, 30, 31, -72, -18, 61, -24, -55, -25, 14, 8, 4, -22, 28, -109, 0, -30, 16, -17, 19, -41}
, {-15, -19, 39, -2, 21, 0, 29, -24, -32, 7, 17, -25, -50, -16, -10, -24, -25, -37, 9, 10, -61, -8, -60, 28, -13, -59, 34, -44, -101, -21, 18, -20, -41, 38, -36, 19, 8, 3, 4, 28, -109, -45, 4, -22, -94, -55, 1, 12, 12, 28, 13, -44, 1, 31, -7, 39, -30, -73, -4, 4, 13, -10, 31, 42}
, {26, 20, 31, 44, 48, -27, -39, 3, 0, -58, 23, -10, -66, 26, -1, -13, -26, 4, -39, -51, 17, 1, -4, -38, 9, -25, -36, 0, -45, 20, 4, -21, 66, 8, 26, 8, 72, 12, -5, 4, -44, 22, -8, -21, 26, 7, -16, -64, 2, -48, -17, -4, -51, -42, -11, 6, -21, -103, 2, 43, -22, 7, -45, 35}
, {-14, -68, -24, -14, 28, 15, -53, -18, 35, -14, -87, 5, 12, 39, 20, -44, 29, -36, -45, 8, -60, 27, -10, -83, -12, -26, 8, 37, 19, 23, -50, 11, -20, -51, 4, -12, 32, -16, -18, 13, -11, 36, -34, -69, 8, -16, -74, -36, -16, 11, -63, -2, -38, 7, -31, -19, 25, 27, -54, 19, 9, 10, 20, 13}
, {-17, 43, 1, -5, -26, -31, 17, -20, -20, -22, 21, -13, 6, 11, -83, -55, -3, 16, 19, 20, 30, -25, 12, -45, 14, 61, 14, 18, -69, 30, 9, 27, 27, -33, 24, 6, -21, -76, 1, 25, -94, 13, -3, 9, -7, -59, -2, -52, -22, -31, 12, -10, -96, -2, -20, -37, -10, -25, 17, -49, 32, -19, 10, -39}
, {10, -70, -28, 25, -18, -4, 1, 24, -39, -7, 39, 12, 25, -93, -38, -18, 8, 12, -28, -22, 31, 13, -26, -23, 11, -48, 2, 34, -58, 6, 34, 14, 18, -35, -51, -9, 36, -11, 0, -13, 29, -40, -13, 39, -82, -53, -23, -10, -45, 24, 4, 84, 2, 38, 27, -90, 6, -63, 64, 30, -55, 75, 2, -58}
, {-61, -34, -96, -35, -75, -29, 61, 14, -21, 7, 19, 4, -90, -94, 29, 1, -77, -9, 3, 2, -7, 13, -84, -60, 13, 13, -38, -19, -48, -1, -9, -15, -78, 19, 14, 100, -15, -2, 23, 8, -59, 4, 7, 6, -61, -42, -85, -20, 19, 7, 14, -74, -23, -24, -25, 3, 64, -79, -29, 4, -33, 22, -5, 5}
, {39, 20, 58, -22, -63, -13, 56, -6, 1, 8, -55, -67, 27, 18, -55, 13, -84, -26, -18, -30, 52, -13, 6, 51, -152, -11, 5, 12, -64, -53, 32, 48, 3, -75, -94, -71, -128, 1, 7, -51, -64, -5, -22, -7, -76, -16, -46, -15, 34, -81, -20, 42, 32, -55, -15, -23, -6, 71, -52, 24, -25, -67, -95, -116}
, {-4, -1, 26, 0, -20, 23, -46, -11, -14, 5, 30, 63, 53, 27, 17, -27, -12, 3, 10, -40, -30, -7, 25, -7, -22, 14, -10, 32, 15, 32, -31, -70, -9, 13, -4, 11, -15, 59, -35, -51, -6, -73, -17, 4, -13, 36, -29, -9, -47, 41, -20, -28, -54, -26, 59, -16, 6, -113, 13, -37, -57, 13, -42, 22}
}
;

#undef INPUT_SAMPLES
#undef FC_UNITS
/**
  ******************************************************************************
  * @file    model.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    08 july 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */


#ifdef __cplusplus
extern "C" {
#endif

#ifndef __MODEL_H__
#define __MODEL_H__

#ifndef SINGLE_FILE
#include "number.h"

 // InputLayer is excluded
#include "conv1d_36.h" // InputLayer is excluded
#include "max_pooling1d_27.h" // InputLayer is excluded
#include "conv1d_37.h" // InputLayer is excluded
#include "max_pooling1d_28.h" // InputLayer is excluded
#include "conv1d_38.h" // InputLayer is excluded
#include "max_pooling1d_29.h" // InputLayer is excluded
#include "conv1d_39.h" // InputLayer is excluded
#include "average_pooling1d_9.h" // InputLayer is excluded
#include "flatten_9.h" // InputLayer is excluded
#include "dense_27.h" // InputLayer is excluded
#include "dense_28.h" // InputLayer is excluded
#include "dense_29.h"
#endif


#define MODEL_INPUT_DIM_0 16000
#define MODEL_INPUT_DIM_1 1
#define MODEL_INPUT_DIMS 16000 * 1

#define MODEL_OUTPUT_SAMPLES 10

#define MODEL_INPUT_SCALE_FACTOR 7 // scale factor of InputLayer
#define MODEL_INPUT_ROUND_MODE ROUND_MODE_FLOOR
#define MODEL_INPUT_NUMBER_T int16_t
#define MODEL_INPUT_LONG_NUMBER_T int32_t

// node 0 is InputLayer so use its output shape as input shape of the model
// typedef  input_t[16000][1];
typedef int16_t input_t[16000][1];
typedef dense_29_output_type output_t;


void cnn(
  const input_t input,
  output_t output);

void reset(void);

#endif//__MODEL_H__


#ifdef __cplusplus
} // extern "C"
#endif
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
