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


const int16_t  conv1d_50_bias[CONV_FILTERS] = {4, 8, 2, -20, 7, -1, -1, 5, 1, 10, 0, -3, 5, 5, 13, -2}
;

const int16_t  conv1d_50_kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS / CONV_GROUPS] = {{{27, 15, -12, 15, 5, -15, 25, -15}
, {17, 20, -12, 25, -20, -6, -8, 7}
, {10, 0, 5, 11, 8, 1, -2, 4}
, {6, 18, -19, -18, 10, -31, 27, 30}
, {10, -10, -4, 24, -24, -4, 7, 18}
, {-2, -12, 9, -8, -13, -23, -7, 8}
, {-14, 15, -12, 27, -24, 11, -5, -2}
, {-13, 14, -1, 14, -26, 10, 21, -17}
}
, {{-9, -2, -9, -20, 5, -17, -4, 25}
, {-1, 12, -24, -21, 16, 17, -27, -3}
, {6, 16, -24, -15, 22, -12, 0, -2}
, {15, -19, -2, -5, 25, 31, 0, -9}
, {24, -21, -14, -16, -19, 27, 21, -14}
, {-8, -6, 10, 6, 10, -9, 5, -20}
, {-15, 12, 13, -2, 1, 5, -22, 7}
, {-17, 12, -9, 32, -1, -3, 0, 7}
}
, {{17, 19, -14, 7, -4, 15, 14, 16}
, {14, -10, 17, -13, -7, 0, -19, 1}
, {-17, 9, -14, -13, 23, 12, -10, -15}
, {28, 19, -11, -15, 17, 15, -11, 0}
, {8, 8, -20, -2, 19, 11, 24, -4}
, {5, -2, 19, -5, -3, -7, 25, 4}
, {25, -10, -21, -16, -13, -25, 1, 11}
, {-7, -20, -12, 1, -3, 8, 17, 19}
}
, {{10, 11, 1, 9, 10, 12, -5, 24}
, {26, 9, 14, -5, 13, 1, -10, 0}
, {12, 3, -14, 8, 7, 14, -15, 24}
, {11, 17, 8, -16, 0, -17, -14, 25}
, {-10, -6, -5, 13, 0, 8, 15, 28}
, {21, -15, 10, -21, -19, 4, 31, -2}
, {17, 2, 15, -24, -18, 6, -10, 3}
, {8, -16, -6, -14, -8, 22, 18, 12}
}
, {{-4, 10, -3, 14, -8, 5, -9, 15}
, {-19, -16, -28, 16, -1, -18, -16, -23}
, {-14, -29, -16, -8, 2, 3, 31, -18}
, {13, -19, 17, -26, 17, 25, -13, 32}
, {-10, -18, -30, 8, 28, -6, 3, 20}
, {-26, -3, -13, 7, 44, -2, 18, 9}
, {-7, 21, 4, 1, 31, -11, 9, 26}
, {-3, -3, -9, 5, -3, 14, -18, 5}
}
, {{10, 3, -16, 14, 24, -19, -26, 19}
, {8, 22, 19, 11, 23, -2, 11, 12}
, {19, 21, 27, 28, 0, 16, 7, -2}
, {8, 31, 20, -13, -20, -12, -3, 18}
, {25, -10, 7, 12, 16, 22, -17, -19}
, {-10, -2, 16, 5, 24, 16, 21, -18}
, {0, 4, 17, 10, 7, -2, -14, -26}
, {18, 19, 14, 16, 8, 11, -23, 14}
}
, {{22, 28, -8, -3, -16, -7, 12, 10}
, {-17, -8, 27, -2, -5, -13, 0, 1}
, {21, 25, 26, 8, 2, 2, 11, 6}
, {21, 10, 11, -10, 20, 5, 2, -28}
, {-1, 29, -17, -4, -6, 9, 13, -11}
, {14, 6, -10, 17, -1, 14, -24, -14}
, {21, 8, 25, -16, 20, 13, -24, -4}
, {-10, 10, 20, -19, 16, 4, -9, -8}
}
, {{3, 18, 26, -21, 22, 11, -7, -17}
, {9, 23, 17, -19, 3, 5, -1, 21}
, {-11, 8, -4, -24, 26, -6, 23, 5}
, {16, 13, 27, -21, 29, -10, 0, -14}
, {-16, -6, 15, -16, -3, -18, 19, 13}
, {13, 17, 11, -2, 6, -23, 0, 11}
, {11, -18, 23, -19, 26, -17, 26, 15}
, {15, 3, -20, -2, -7, 8, 22, 11}
}
, {{2, -7, 18, 21, 3, -18, 13, -4}
, {-14, 1, 13, 9, 11, -16, -7, -15}
, {-8, 14, 13, -8, 17, 8, -5, -10}
, {7, 1, -21, -35, 18, 14, -4, -5}
, {-12, 3, -2, -18, 28, -4, -16, 10}
, {16, -23, 22, 10, -7, 0, -24, 4}
, {19, -1, -23, -25, 25, 17, -20, 29}
, {7, -15, -17, 12, 8, 21, 16, 23}
}
, {{19, -10, -23, -3, -13, -9, 27, 27}
, {12, 16, 2, 11, 24, 6, 10, 22}
, {-19, -2, -21, -20, -21, -14, 3, -3}
, {-4, -16, -20, -13, 8, -14, -23, 7}
, {4, 24, -13, 25, 10, 26, 21, -8}
, {-3, 3, -21, 14, -7, -11, 23, 1}
, {-3, -14, -17, -15, -17, 21, 24, 3}
, {2, -11, 16, 2, -21, -13, 0, -16}
}
, {{2, -12, -12, -14, 12, 14, 18, -20}
, {-9, -2, -22, 22, 11, -17, 7, 28}
, {-9, -10, -15, 26, -14, 3, 17, 1}
, {19, -21, 19, 21, -16, 8, -18, -5}
, {11, -20, 18, 22, -6, 13, -19, 25}
, {11, -6, -13, 0, 3, -28, 27, 21}
, {-12, 10, 4, 23, 7, 0, 5, 24}
, {-17, -10, -1, 10, -22, -3, 25, 9}
}
, {{16, -14, -3, -6, 31, 9, -7, 24}
, {28, -4, 22, 17, 30, 14, -3, 11}
, {-11, 14, 13, 9, 25, 1, -28, 9}
, {22, -13, -16, -19, 11, 13, 5, -4}
, {-13, 26, 20, 23, 11, -7, -20, 0}
, {21, -17, 22, -17, -4, 14, -15, 5}
, {-2, 21, -5, 29, 22, 30, -17, 17}
, {1, 17, 20, -6, 11, 23, 0, -2}
}
, {{-1, -17, -15, 18, -23, 10, -15, -10}
, {-21, 17, 26, 19, -17, -9, -22, -18}
, {16, -17, -18, 26, -6, -8, -8, -2}
, {-11, 9, 10, 4, -34, 11, -8, -19}
, {-18, 0, -5, 8, 2, 29, 14, -16}
, {-14, 0, 30, 20, -21, -8, 11, 5}
, {13, 1, 20, -9, 7, 12, -12, -1}
, {-2, 29, 10, 19, -8, 22, -11, 19}
}
, {{6, 19, 21, -3, -12, 7, 20, 24}
, {-20, 20, -9, -10, -15, -15, 21, 1}
, {-13, -12, -17, 16, -22, 6, -10, -9}
, {-5, 23, 14, -3, -20, -14, 11, 2}
, {-16, -17, 0, 16, -16, -2, -14, -11}
, {5, 3, 21, -14, -30, -5, -17, -9}
, {-9, 23, 26, 3, 10, -17, -8, 9}
, {-4, -22, 12, -14, 6, -2, 19, 19}
}
, {{-5, -7, -8, -2, 11, 16, 10, 13}
, {-5, 7, -10, 4, -1, 38, -14, 5}
, {22, -1, -19, 19, -19, 7, -17, 14}
, {-1, -20, -14, -16, -16, 18, -1, 6}
, {22, 0, -5, -9, -4, 27, -17, 18}
, {11, -12, -8, 12, 6, 19, -32, -16}
, {-6, -29, -2, -6, -11, 14, 1, -8}
, {30, -9, -26, -16, -5, -14, -14, -8}
}
, {{13, -9, 14, -7, -23, 28, 19, 3}
, {-1, -4, -6, 9, -14, 6, 24, 27}
, {-16, 9, 17, -6, -17, 23, -8, 16}
, {3, 9, -7, 9, -7, 17, 1, 28}
, {-9, 2, -21, 13, 1, 18, 7, -1}
, {-3, 12, -8, 14, 1, 5, 7, -12}
, {-8, -11, -10, -21, -21, -9, 24, 12}
, {-4, 21, -31, -27, 16, -21, -12, 13}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_GROUPS