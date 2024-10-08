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
#define FC_UNITS 3


const int16_t dense_bias[FC_UNITS] = {-22, 18, 7}
;

const int16_t dense_kernel[FC_UNITS][INPUT_SAMPLES] = {{-5, -2, 5, -4, -25, 18, 2, -9, 7, 3, -25, 6, 16, -15, 21, 8, -4, -9, 2, -16, -25, -15, 11, -7, -6, -3, -1, 6, -19, 0, -34, 2, 14, 10, -8, -11, -2, -10, 4, 11, -2, 4, -4, -8, -1, -11, -10, -7, 0, -3, -11, -4, -1, -7, -6, -9, 3, -6, -4, 21, -27, 14, -3, -4, 1, -18, -5, 22, 22, 0, 18, 7, 11, 22, -4, 23, 4, -17, 8, -17, 7, -20, 10, -3, -2, 10, -6, 2, 23, 3, 17, -1, -11, -6, -15, 11, 6, 16, -6, -14, 9, -2, 0, 6, 18, 0, -2, 0, 2, 20, -4, 17, 5, 6, 5, -5, -8, 14, 7, 15, 28, 8, 17, -4, 5, 13, 10, 20, 4, -22, 15, -17, -8, 10, -12, -18, 11, -1, -7, 6, 14, -17, -10, 5, 7, -11, -17, -1, 9, 6, 16, 6, 0, 7, 17, -3, -8, 9, -7, -2, -8, 1, 12, 7, -2, 7, -6, 7, 5, 12, -14, -10, 3, -6, 2, 11, 2, 16, -15, -16, 19, 2, 11, 18, -1, -1, -3, 22, -5, 6, 2, 11, -4, -11, 11, -2, 4, -3, 7, 2, -23, 8, -10, -9, 11, -11, 0, -10, 5, -18, -13, 11, 9, 3, -16, 3, 11, -14, -10, 20, -24, 13, 4, 12, -13, -6, 13, 15, -2, 15, -13, -1, -1, 12, -23, 14, -3, -9, 10, -12, -8, 1, -24, -14, -8, -4, 15, 11, -11, -7, 17, -3, -21, -2, -5, -2, -6, -6, 12, -14, 12, -4, 3, 17, 1, -8, -1, -5, 14, 10, -8, 1, 21, 2, 20, 2, -5, 6, -4, 2, 13, -15, -1, 1, -4, 6, -2, 16, 16, 18, 22, -11, 16, 11, -3, 14, 15, 22, 18, 0, 6, 14, 17, -11, -6, 20, -2, 2, 1, 7, -6, 13, 13, 1, 18, 17, -21, 20, 6, 15}
, {5, -1, -11, 8, 4, -24, -15, -18, -4, -15, 11, -26, -17, 18, 8, -2, -10, 21, -18, -19, -7, 17, 2, -11, -7, 8, 4, -8, 9, 8, 9, 5, 16, 0, 2, -15, 2, 1, -7, -2, 8, -11, 7, 8, 12, 7, 2, 8, -14, -24, 21, 13, 1, -10, 31, -11, -21, -20, 8, -18, 20, -22, -5, -20, -12, 29, 4, -12, -8, 6, 10, 22, 10, -11, -1, 5, -4, -19, -19, -3, -4, 34, 6, 9, -10, -10, 10, -3, -2, -2, 11, 0, 22, 8, -15, 7, 13, 24, -15, 0, 3, 19, -8, 6, -1, 7, 14, 3, -5, -6, -3, -15, -16, -19, -2, 20, -10, -20, 3, -25, 7, 9, -9, 3, 21, 7, 0, 2, -3, 29, -4, -15, -20, -8, -24, 2, 11, -8, -11, 0, -19, -6, -24, 7, 1, 0, 0, 3, -6, -25, 22, 0, 0, 14, -8, -24, 15, 5, 25, -27, 0, -31, 1, -24, -10, -21, 23, 0, -3, 4, -10, -6, 1, -3, -25, -22, -18, 8, 34, 2, -2, -4, -22, 9, -30, 8, -8, 2, 4, 0, -11, 4, 8, 22, -27, -14, -7, -17, 4, -5, 0, 7, -6, -2, 6, -14, -12, 20, 9, 41, 35, 6, 11, -17, -39, -14, 6, -8, 3, -12, 35, -15, -32, -18, -14, 9, 4, 7, -23, -14, -4, -8, 3, 4, -12, -20, 9, -18, -8, -16, 5, -3, -18, 5, -32, -3, 6, 8, -5, -8, -2, -1, 24, 5, -13, 11, -14, 16, -24, -5, -13, 2, -19, -24, 6, -22, 4, -21, 2, -26, -4, 18, -1, 28, 23, -1, -27, -24, -17, -2, -8, -8, -23, 5, 8, -6, 3, -25, -14, -12, -19, 0, -28, -24, 11, 1, -24, 0, 5, 4, -6, 6, -10, -34, 9, -3, 29, -27, -5, 6, -8, 4, -16, 2, 10, 7, 25, -6, -11, -18}
, {20, -4, 10, 3, 24, -19, -14, -18, -8, -27, 11, 24, -12, 7, -24, -3, -13, 4, -36, 6, 18, 19, -14, -16, 24, 0, -13, -18, 5, -13, 19, 15, -12, -15, -15, -1, 2, -6, -16, 22, 3, -24, -4, 5, 10, -16, 2, 7, -14, 6, -8, -9, 0, -9, -3, 7, -3, -6, -30, -5, -14, -16, -13, -4, -21, -14, -9, -7, 0, -23, -29, -2, -19, -4, 5, -14, -32, 5, -4, -7, -22, -26, -39, -1, -11, -4, -6, -6, -13, -2, -35, -25, -3, -4, 24, -3, -17, -3, -23, 0, 3, -24, 18, -17, -33, -21, -28, -12, 18, 11, -7, -5, 25, -9, -2, 17, -4, 17, -21, -4, -8, -2, 3, -14, -16, -13, -22, -8, -10, 8, 7, 11, 15, 19, 12, 16, 4, 10, -9, -6, -10, 24, 17, -9, 12, -11, 11, 22, -3, 13, 11, -15, 18, -14, -6, -14, 0, 1, -7, 16, 3, 3, 0, 11, 8, 0, -5, 18, -2, 6, -10, 20, -3, -1, -15, -6, 0, -10, -24, 15, 13, 17, 13, 21, 12, 6, 20, 22, -16, 11, -12, -7, 4, -26, 10, 11, -2, 21, 9, 2, -8, 1, -14, -3, 13, 15, -3, -11, 0, -34, -21, -2, -5, 10, 13, -10, 5, -2, 2, -16, 4, 23, 14, -4, -8, -8, -5, 4, 6, 6, 1, 9, -13, -1, -9, 16, 12, 21, 13, 3, -13, 13, 21, 24, 24, -10, -21, -3, 12, 24, 1, -4, -7, 11, -14, 3, 15, -25, 24, 2, -6, 19, -6, -21, 17, -4, -13, 13, -6, 9, 3, -11, -7, -22, -33, 8, 3, 14, 17, -9, -7, -8, 5, 2, -4, -5, 18, 1, -10, -3, 4, 2, 7, 21, -14, 4, 13, 5, -2, 10, 2, 14, 15, 18, -5, -10, -5, 17, 21, 24, 5, 7, -14, 5, -11, -10, -4, -9, 17, -4}
}
;

#undef INPUT_SAMPLES
#undef FC_UNITS