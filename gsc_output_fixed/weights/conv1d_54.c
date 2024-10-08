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


const int16_t  conv1d_54_bias[CONV_FILTERS] = {-2, 7, 2, 1, 13, 10, 6, 7, 1, 4, 3, -1, -1, 0, 2, 6}
;

const int16_t  conv1d_54_kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS / CONV_GROUPS] = {{{-16, -9, -22, -3, 27, 23, -3, 20}
, {6, 21, -1, -5, 21, 4, -7, -1}
, {15, 21, 3, 3, 19, 5, 6, 10}
, {6, 25, -17, -29, 9, 5, -17, -9}
, {-9, 31, 24, -3, -16, -8, 0, 26}
, {-19, 21, 24, -19, 3, 15, -15, 16}
, {25, -4, 2, 2, 29, 6, 12, -13}
, {7, -2, 0, -5, -1, 24, -20, 26}
}
, {{10, -14, 14, -10, 16, 2, 3, 30}
, {-19, 25, 20, 0, 25, 25, -26, 21}
, {-16, -14, -18, -11, -12, 13, -12, 4}
, {-24, 5, -6, 19, 18, 0, -23, 22}
, {-24, 21, 19, 19, -8, -12, -26, 28}
, {12, -11, 14, -17, 18, -18, -25, -5}
, {-15, 16, -16, -5, -10, 2, -18, 27}
, {8, -16, -9, 12, 11, -15, -6, -8}
}
, {{-4, -2, 24, 12, 12, 4, -9, -7}
, {-5, -5, -24, -17, 9, 24, -19, -2}
, {-23, 13, -13, 10, 6, 5, 6, 3}
, {21, -28, -3, 13, 26, 4, 7, -3}
, {-14, -27, 12, -1, 29, 15, 9, 18}
, {-15, -23, -7, 17, -12, -26, -26, -14}
, {-9, 12, -11, -4, -23, 9, -1, 16}
, {-5, -17, 18, -19, 10, -3, 18, -13}
}
, {{-18, -9, -9, 13, -28, 7, -17, 19}
, {0, -10, 14, 6, -12, -10, 31, 1}
, {-11, 14, 15, -19, 12, -8, 19, -30}
, {4, 1, -20, -23, 21, -20, 23, -17}
, {21, -8, -5, 5, 9, -16, 0, -5}
, {-8, -1, -7, 4, 17, -7, 8, 6}
, {-23, -13, 19, 23, 2, 17, -1, 11}
, {-4, 2, -5, -6, -17, -17, 21, -2}
}
, {{8, -8, 22, 7, 19, 20, -4, 26}
, {-5, 8, 21, 5, 3, 4, -1, 4}
, {18, 10, 29, 17, -11, -7, -21, -20}
, {-7, 2, -19, 1, 4, -1, -16, -21}
, {21, -26, -7, -26, -22, 4, -23, -24}
, {-10, -19, -34, 13, -16, -28, -13, -41}
, {-11, -21, -21, 3, -22, -21, -10, -1}
, {25, -7, 16, -5, -3, -8, 21, -7}
}
, {{-11, 3, -11, -24, -10, 11, 6, -17}
, {15, 12, 2, -1, 24, 11, 10, -10}
, {-9, 5, 16, -7, -8, -12, 3, -4}
, {-21, 3, -5, 10, 6, -11, 28, 15}
, {23, 10, -14, 5, -3, -16, 22, 3}
, {-12, 30, 15, -5, 21, 13, 21, -10}
, {-13, 23, -3, 0, -5, 2, 22, 4}
, {5, -16, -13, -14, -22, -25, -1, 8}
}
, {{-5, -3, 7, 15, -11, 6, -25, -6}
, {-4, 22, 0, -14, 15, 4, -21, -1}
, {-7, 19, 12, 20, -13, -6, -4, -4}
, {11, 26, -11, 9, 7, 0, -24, 22}
, {13, 1, 19, -5, 7, 3, 20, 4}
, {19, 2, -13, -8, 4, -14, 8, -24}
, {20, -16, -14, 16, -8, 1, -22, -4}
, {13, -11, -24, -23, -10, -26, 18, -14}
}
, {{4, -15, 15, -17, -17, 13, 7, -13}
, {-24, 16, -3, 0, 23, 12, -16, -1}
, {-16, -7, -9, -24, 6, 16, 10, -13}
, {-20, -15, -13, -15, 5, 27, -15, -8}
, {0, -4, 20, 19, 18, -8, 6, 7}
, {-18, -2, 28, 7, 2, 29, -17, 13}
, {-17, -18, -19, 13, -1, 1, 26, 10}
, {-19, -21, -3, -1, -9, 26, 7, -18}
}
, {{2, 13, -7, -15, 1, -4, 10, -20}
, {13, -19, 21, 9, -16, -11, -22, 17}
, {12, -1, 19, -11, -13, 9, -10, -16}
, {2, 0, 10, -2, 18, -6, -12, -7}
, {22, -13, -3, 19, 6, 6, 1, -11}
, {8, 24, 17, 14, -17, 4, 4, -18}
, {-4, 18, -7, 20, 4, -17, 29, -5}
, {14, 18, -17, 4, 3, -4, 3, 7}
}
, {{3, 21, 18, -6, -1, -27, -24, -13}
, {13, 28, -8, 5, 2, 18, 15, 3}
, {6, -16, 25, -28, 5, -7, -21, -22}
, {-11, 3, -5, -3, 18, -13, -22, -5}
, {-25, -5, 15, -1, -21, 22, -13, -21}
, {7, 21, 20, 26, -10, 5, -15, -18}
, {-18, -13, 27, -12, 8, 20, 2, -26}
, {-8, -30, -17, 12, -16, -25, -5, -12}
}
, {{-14, -25, -19, 5, 15, 25, 18, 1}
, {-12, -12, 6, -13, -11, -12, 1, -20}
, {-15, 1, 19, 23, 1, 11, 12, -18}
, {-11, -4, 13, 1, -11, 23, 23, -17}
, {-23, -2, 0, -5, 0, 5, 25, -16}
, {11, -19, -2, 7, 10, 6, -11, -8}
, {12, 4, 12, -7, 3, 8, -15, -20}
, {-15, 3, 10, 8, -9, 24, 5, 9}
}
, {{-5, -10, -13, 7, -11, -34, -4, -6}
, {24, -5, 17, -24, -4, -9, -9, -10}
, {25, 10, 20, -6, -4, -4, 17, -8}
, {-12, -18, 0, 18, 13, -12, 20, -5}
, {5, 2, -8, -6, 9, -4, -12, 5}
, {-25, -2, 14, -3, 14, 22, -14, 20}
, {-22, 19, -18, -7, 1, 9, 12, 5}
, {-19, 23, 12, -5, -7, -13, -27, 6}
}
, {{18, 28, 12, 7, -11, -12, 16, 8}
, {8, 15, -24, 8, 19, -24, -20, 4}
, {-4, 17, 17, -21, -22, -14, 5, 1}
, {28, 21, 1, -22, 7, 11, -1, -7}
, {23, 19, 5, -6, -2, -5, -10, 16}
, {-12, 22, -22, -7, -18, -9, -5, -4}
, {-21, 21, 10, 9, 22, 17, -6, -15}
, {8, 10, 20, -17, 21, -2, -21, 9}
}
, {{10, 15, -8, 5, 16, 20, -2, -1}
, {19, -17, -18, -8, 14, -5, 7, 18}
, {18, -15, 16, 0, -14, 15, 19, 12}
, {13, -10, -11, 5, -3, -11, -2, -4}
, {-15, 20, -8, 21, 5, 25, 25, 15}
, {4, -6, -8, 20, 10, 3, 22, 4}
, {-20, -24, 25, 6, -3, 21, -9, 4}
, {14, -14, -3, 28, 12, 17, 28, -6}
}
, {{13, 1, -5, -11, 20, 9, 2, -9}
, {-23, 9, 4, 20, 0, 14, -20, 19}
, {-4, -5, -14, -20, 12, 2, -27, -12}
, {12, 0, 28, -9, 10, 23, 12, -10}
, {5, -12, -13, -13, 27, 24, -13, -13}
, {3, -16, -6, -1, 18, 13, -19, 20}
, {11, -18, -5, -9, -10, 30, -17, -2}
, {18, -1, -22, -21, 19, -12, -10, 28}
}
, {{20, -6, -20, -9, 15, -15, -14, -26}
, {12, -23, -14, -10, 11, -19, -21, -14}
, {-1, -26, 24, -1, -2, -10, -8, -12}
, {-2, 9, 9, -20, -9, 25, -1, -16}
, {3, 5, 16, 17, -15, -11, 16, -21}
, {-14, 18, 26, -18, 15, -19, -14, -21}
, {-12, -5, -2, 23, 21, 13, 16, -16}
, {29, 23, 14, 24, 13, 18, -8, -1}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_GROUPS