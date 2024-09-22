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


const int16_t  conv1d_49_bias[CONV_FILTERS] = {-8, -5, -1, -1, -3, -3, 0, -4}
;

const int16_t  conv1d_49_kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS / CONV_GROUPS] = {{{-6}
, {13}
, {22}
, {17}
, {-16}
, {11}
, {19}
, {11}
, {24}
, {27}
, {18}
, {-1}
, {-33}
, {-5}
, {0}
, {-16}
, {-17}
, {39}
, {4}
, {23}
}
, {{28}
, {-19}
, {-30}
, {-10}
, {21}
, {14}
, {32}
, {-6}
, {14}
, {7}
, {-28}
, {-8}
, {14}
, {25}
, {19}
, {-13}
, {8}
, {-4}
, {-28}
, {-12}
}
, {{-30}
, {1}
, {-14}
, {-21}
, {32}
, {26}
, {17}
, {-3}
, {-32}
, {-30}
, {24}
, {-19}
, {1}
, {34}
, {-2}
, {-13}
, {-23}
, {13}
, {5}
, {0}
}
, {{-5}
, {-2}
, {-2}
, {25}
, {33}
, {18}
, {-7}
, {-30}
, {-17}
, {0}
, {8}
, {22}
, {-10}
, {-21}
, {-34}
, {-5}
, {30}
, {11}
, {1}
, {-28}
}
, {{13}
, {-7}
, {24}
, {7}
, {7}
, {-3}
, {-26}
, {15}
, {-27}
, {-19}
, {19}
, {-1}
, {-11}
, {26}
, {7}
, {-10}
, {5}
, {-17}
, {2}
, {-15}
}
, {{13}
, {5}
, {0}
, {-17}
, {-19}
, {-6}
, {6}
, {-9}
, {-8}
, {11}
, {6}
, {-13}
, {-20}
, {-32}
, {-34}
, {-4}
, {14}
, {24}
, {-2}
, {23}
}
, {{-26}
, {14}
, {19}
, {-29}
, {-32}
, {3}
, {18}
, {-35}
, {-18}
, {3}
, {8}
, {7}
, {-13}
, {1}
, {25}
, {23}
, {-2}
, {-3}
, {11}
, {22}
}
, {{22}
, {-36}
, {-30}
, {24}
, {33}
, {4}
, {-26}
, {-19}
, {30}
, {8}
, {-7}
, {-26}
, {-6}
, {-1}
, {-2}
, {6}
, {-12}
, {11}
, {12}
, {17}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_GROUPS