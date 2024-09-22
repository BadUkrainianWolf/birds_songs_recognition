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


const int16_t  conv1d_bias[CONV_FILTERS] = {-1, -17, 2, -8, -1, -9, 3, 4}
;

const int16_t  conv1d_kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS / CONV_GROUPS] = {{{-19}
, {-4}
, {21}
, {-22}
, {3}
, {16}
, {12}
, {-10}
, {13}
, {19}
, {-3}
, {-21}
, {8}
, {-27}
, {6}
, {-19}
, {27}
, {-20}
, {8}
, {-6}
}
, {{20}
, {-19}
, {-6}
, {5}
, {-5}
, {19}
, {-16}
, {3}
, {3}
, {14}
, {-3}
, {14}
, {2}
, {4}
, {1}
, {24}
, {19}
, {-7}
, {11}
, {19}
}
, {{9}
, {19}
, {-26}
, {-19}
, {-16}
, {12}
, {-27}
, {-6}
, {1}
, {10}
, {-12}
, {15}
, {38}
, {23}
, {16}
, {-10}
, {24}
, {-12}
, {-26}
, {-15}
}
, {{-18}
, {-9}
, {-3}
, {-9}
, {-22}
, {2}
, {2}
, {-14}
, {11}
, {2}
, {19}
, {-1}
, {6}
, {12}
, {3}
, {24}
, {-4}
, {12}
, {33}
, {1}
}
, {{-9}
, {0}
, {10}
, {10}
, {-14}
, {-9}
, {-2}
, {-8}
, {11}
, {20}
, {-6}
, {-20}
, {-36}
, {3}
, {9}
, {-15}
, {8}
, {2}
, {20}
, {15}
}
, {{-18}
, {9}
, {22}
, {18}
, {3}
, {-2}
, {16}
, {1}
, {17}
, {32}
, {-10}
, {7}
, {11}
, {-1}
, {-9}
, {21}
, {28}
, {13}
, {6}
, {-1}
}
, {{9}
, {10}
, {0}
, {4}
, {19}
, {22}
, {-5}
, {30}
, {17}
, {1}
, {25}
, {6}
, {2}
, {2}
, {9}
, {20}
, {21}
, {-3}
, {16}
, {-7}
}
, {{-16}
, {-3}
, {-24}
, {-11}
, {-34}
, {-2}
, {4}
, {-19}
, {-25}
, {4}
, {-2}
, {5}
, {-17}
, {-20}
, {-31}
, {16}
, {-21}
, {31}
, {-1}
, {17}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_GROUPS