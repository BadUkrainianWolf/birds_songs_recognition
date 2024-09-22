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


const int16_t  conv1d_1_bias[CONV_FILTERS] = {1, 13, 3, -3, -14, 5, 4, -10, 5, 4, -1, -9, 3, 0, -11, 8}
;

const int16_t  conv1d_1_kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS / CONV_GROUPS] = {{{7, -19, 16, 20, -20, 3, 11, 21}
, {16, -8, -6, -9, -34, 15, -23, -22}
, {-23, 11, -9, 17, -25, -29, -5, 24}
, {-2, 19, 18, -4, -8, 4, 8, -8}
, {-8, -34, 8, -17, -6, -19, -13, -8}
, {-20, 4, 9, -13, 12, 8, -1, 12}
, {14, 5, -3, 17, -14, 16, 11, 4}
, {-20, -23, -23, -9, 8, -18, -28, -5}
}
, {{5, -21, -30, 13, 3, -19, -13, 4}
, {-39, 16, -11, -8, 11, 25, -7, -34}
, {-1, -7, -7, -22, -3, -3, 8, -28}
, {7, 11, -12, -13, -8, 11, 11, -10}
, {-24, -18, -13, 0, -35, -25, -2, 35}
, {-4, -3, -22, 7, -31, -11, -6, -9}
, {-23, 11, -15, 17, 2, 17, 23, 9}
, {-20, -24, -39, 9, -24, 6, 18, 1}
}
, {{23, 1, 32, -5, 22, -21, 14, -12}
, {1, 17, 25, -28, -4, 5, 15, 20}
, {0, 13, 6, -10, -14, -12, -8, 0}
, {-17, 15, 1, -14, -9, -10, 9, 16}
, {4, -22, 9, -25, 18, -15, -23, -11}
, {-4, -25, 18, 19, -13, 14, 20, 23}
, {4, -6, -4, -15, -19, -28, -19, 9}
, {7, -8, 32, -15, 11, 23, -20, -1}
}
, {{-4, 23, 16, 3, -16, 2, -16, -13}
, {-9, 12, 19, 25, 6, 1, -17, 4}
, {-15, -14, 8, -9, 26, 8, -3, 15}
, {-24, 16, 9, 10, 26, -14, 23, -15}
, {-23, -23, -16, -22, -5, 9, 8, 7}
, {10, 20, -8, 9, 3, -12, 16, -13}
, {-18, -13, 3, -3, -4, -1, -2, 10}
, {8, 0, 17, 12, -6, -13, 10, -31}
}
, {{-19, -8, 6, -5, -13, 14, 9, -18}
, {-21, -25, -16, 0, 6, -7, 9, 14}
, {8, 6, 14, -20, -12, 8, -20, 8}
, {12, -20, 0, 10, 11, 18, 14, -17}
, {-9, 10, -1, -27, -9, 22, -5, -10}
, {-25, -10, 13, 12, -26, -19, 3, 0}
, {11, -6, 0, 15, -1, -2, -18, -8}
, {6, 8, 6, -24, 14, -3, -25, -17}
}
, {{-3, -11, -25, 10, -22, -11, 10, 7}
, {0, -23, -22, -35, -8, -15, -9, 12}
, {22, 17, -15, -11, 4, 8, -19, 11}
, {-11, -24, -19, 10, -14, 1, 20, -32}
, {-2, 2, -26, 2, -37, -16, -19, 17}
, {7, 12, -5, 22, 7, 21, 22, 18}
, {13, 1, -26, -6, -20, -3, 22, -27}
, {-5, -31, -3, -25, -17, -18, 3, 22}
}
, {{-19, 14, -24, -4, -11, 11, 11, 13}
, {-35, -3, -5, 4, -22, 21, -1, -20}
, {-26, -2, -5, -22, -5, -8, -8, -9}
, {-22, -3, 6, -19, 9, -2, 3, 3}
, {-26, 20, -21, 4, -10, 16, 11, 0}
, {-34, 8, 5, 5, -16, 7, 24, -41}
, {-21, -21, -22, -13, -9, -1, 18, -29}
, {11, -25, -7, 0, -13, 12, -23, 30}
}
, {{-31, -28, -28, 8, -28, 0, -29, -12}
, {-24, 8, -22, 10, 12, 7, 7, -28}
, {6, -29, -16, -24, -13, -7, -31, -14}
, {-4, -24, -2, 6, 13, -27, -11, -29}
, {10, -5, -8, 6, -18, -19, 5, -9}
, {-10, -27, 3, 3, 7, 1, 2, -16}
, {0, 14, 10, 8, -24, -19, -7, -1}
, {8, -11, 7, -26, 5, 13, 9, -2}
}
, {{-22, -25, 19, -16, 21, -19, 9, 18}
, {5, -14, 16, 15, 8, -16, 21, -5}
, {-15, -31, 16, 7, -7, -29, 6, 20}
, {25, 9, 17, -12, -3, -14, -5, 21}
, {18, -25, -19, -17, -6, 19, -15, -37}
, {7, -5, -14, 10, 13, -27, -31, 22}
, {18, 3, 15, 12, 9, -1, 17, -28}
, {16, 8, -13, -30, 7, -32, -12, 13}
}
, {{-10, 11, -16, 4, -8, 10, 5, 20}
, {-7, -20, -16, -10, -18, 16, 20, 14}
, {-10, -8, -6, -23, -2, -8, 12, -6}
, {-16, -27, 8, -4, -10, -25, -32, -3}
, {2, 24, 12, 24, 11, -5, 15, 22}
, {-24, -14, -19, -8, -17, 21, 29, -34}
, {-27, -18, -24, -36, -20, -2, -4, -15}
, {-16, -24, -10, 6, -5, -5, -17, 23}
}
, {{-24, 5, -6, -26, -1, 0, 7, 24}
, {16, -10, 9, 11, -22, -5, 3, 25}
, {12, 0, -22, 0, -15, -9, -2, -3}
, {17, -11, 1, 8, -26, 15, 9, 19}
, {-20, -5, -7, -26, -7, 25, 26, -27}
, {24, -10, 2, -8, 7, -7, -30, 21}
, {-1, 22, -13, 32, -32, 31, 4, -2}
, {-2, -2, -19, -3, -21, -19, -15, -13}
}
, {{-19, 13, -28, -15, -30, -16, -22, -1}
, {-23, -17, 1, -9, 11, -17, -21, -10}
, {-22, -25, -3, -18, -14, 0, -22, -29}
, {4, -17, -5, 14, -13, 2, -4, 3}
, {10, -13, -26, 6, 12, -16, -20, -18}
, {-10, -10, -30, 10, -3, -3, -24, -6}
, {-31, -5, -23, 3, -27, -29, 13, -11}
, {-19, -21, -27, -30, 0, -30, -28, -8}
}
, {{-7, -11, 7, -14, -4, 10, 24, 13}
, {18, -3, 17, -6, 10, -5, -3, 23}
, {-1, 5, 13, 2, -16, -1, -15, -10}
, {-8, 5, 3, 3, -13, -4, 6, -17}
, {9, -17, 1, 2, 19, -13, -30, 14}
, {25, 17, -3, 9, -7, -7, 3, 10}
, {15, -9, 11, -17, -6, 11, -2, 9}
, {11, 10, -15, 25, 24, -20, -10, 6}
}
, {{4, 5, 14, 32, -4, 29, 15, -27}
, {20, -8, 8, -4, 6, -12, -17, -13}
, {-7, 33, 11, 36, -9, -19, 11, 25}
, {-13, -11, 17, 5, 9, 19, 6, -14}
, {33, -13, -9, -22, -6, -23, -10, 18}
, {22, 3, -12, 8, 3, -2, 1, -19}
, {-4, -13, 5, -16, 26, -2, -25, -2}
, {24, 3, 19, 19, -1, -12, -8, -6}
}
, {{-16, -12, -32, 11, -12, -15, -20, -3}
, {-32, 4, 8, 7, -8, -15, -6, -19}
, {-5, 3, -9, -28, -23, 1, -9, 14}
, {6, 9, 11, -4, 3, -16, -26, -1}
, {2, 2, 2, 11, 12, -2, 2, -25}
, {7, 12, -19, -23, 2, 4, 2, -6}
, {-9, -9, -30, -14, -33, -28, -24, 12}
, {-28, 14, -7, 10, -23, -15, 11, 3}
}
, {{14, -6, -1, 10, -9, -22, -21, -16}
, {-4, -10, 12, 4, -12, 11, 4, 8}
, {9, 22, 37, 3, -12, 5, -1, 22}
, {21, -19, 22, 13, 5, -17, -12, -10}
, {23, -3, 21, -4, 21, 18, 6, 13}
, {26, 20, 17, 3, 13, 8, 15, 20}
, {26, 4, 4, 15, 10, -11, 3, -16}
, {25, 5, 29, -2, 3, -20, 7, 18}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_GROUPS