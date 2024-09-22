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


const float  conv1d_50_bias[CONV_FILTERS] = {0x1.0aa1a60000000p-5, 0x1.085d300000000p-4, 0x1.492cea0000000p-6, -0x1.3cae9a0000000p-3, 0x1.eedade0000000p-5, -0x1.67e4a00000000p-8, -0x1.8257d60000000p-8, 0x1.4787940000000p-5, 0x1.8b539a0000000p-7, 0x1.545a080000000p-4, 0x1.bd44cc0000000p-8, -0x1.52ad640000000p-6, 0x1.7fa2460000000p-5, 0x1.52293c0000000p-5, 0x1.b481880000000p-4, -0x1.8118e20000000p-7}
;

const float  conv1d_50_kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS / CONV_GROUPS] = {{{0x1.bdaf820000000p-3, 0x1.f195a00000000p-4, -0x1.7d38b00000000p-4, 0x1.e7aa140000000p-4, 0x1.6344ce0000000p-5, -0x1.d415600000000p-4, 0x1.91ec180000000p-3, -0x1.c54f5c0000000p-4}
, {0x1.1a4f9a0000000p-3, 0x1.4ba8c20000000p-3, -0x1.7b1d380000000p-4, 0x1.912d300000000p-3, -0x1.3c28300000000p-3, -0x1.45b6800000000p-5, -0x1.c4cf080000000p-5, 0x1.e7ebd60000000p-5}
, {0x1.5efd6a0000000p-4, 0x1.74bda80000000p-8, 0x1.661ae20000000p-5, 0x1.64857e0000000p-4, 0x1.1226800000000p-4, 0x1.4710100000000p-7, -0x1.10d1ba0000000p-7, 0x1.04ec720000000p-5}
, {0x1.99aac60000000p-5, 0x1.259d720000000p-3, -0x1.2852620000000p-3, -0x1.1d20600000000p-3, 0x1.48d6420000000p-4, -0x1.efac140000000p-3, 0x1.ba44900000000p-3, 0x1.e575900000000p-3}
, {0x1.574fb20000000p-4, -0x1.36164c0000000p-4, -0x1.e9540e0000000p-6, 0x1.813f040000000p-3, -0x1.773ee00000000p-3, -0x1.90eafe0000000p-6, 0x1.ccd6300000000p-5, 0x1.2676b40000000p-3}
, {-0x1.7b20a00000000p-7, -0x1.61d4d20000000p-4, 0x1.3075de0000000p-4, -0x1.c035da0000000p-5, -0x1.9cdc760000000p-4, -0x1.6bd5e60000000p-3, -0x1.adffc60000000p-5, 0x1.1017120000000p-4}
, {-0x1.b60d120000000p-4, 0x1.e462b80000000p-4, -0x1.6e3b440000000p-4, 0x1.b58b180000000p-3, -0x1.700eda0000000p-3, 0x1.7651ea0000000p-4, -0x1.3015a60000000p-5, -0x1.44ee2a0000000p-7}
, {-0x1.9161300000000p-4, 0x1.db746a0000000p-4, -0x1.6634c20000000p-11, 0x1.d2a1220000000p-4, -0x1.94e6360000000p-3, 0x1.4634220000000p-4, 0x1.5fe7fe0000000p-3, -0x1.0b42880000000p-3}
}
, {{-0x1.09cf860000000p-4, -0x1.4685fe0000000p-7, -0x1.1d18fe0000000p-4, -0x1.359e660000000p-3, 0x1.7893fa0000000p-5, -0x1.04e1500000000p-3, -0x1.9ddd0a0000000p-6, 0x1.9336560000000p-3}
, {-0x1.16ceee0000000p-9, 0x1.9583ae0000000p-4, -0x1.70dad20000000p-3, -0x1.4b54b20000000p-3, 0x1.0e4dc40000000p-3, 0x1.17a8520000000p-3, -0x1.abb74e0000000p-3, -0x1.7dcf700000000p-6}
, {0x1.bce19e0000000p-5, 0x1.0b16980000000p-3, -0x1.74de500000000p-3, -0x1.cc15ae0000000p-4, 0x1.60be2a0000000p-3, -0x1.61512c0000000p-4, 0x1.b759560000000p-8, -0x1.3896f80000000p-7}
, {0x1.ef83e20000000p-4, -0x1.2e43f00000000p-3, -0x1.cfe6f60000000p-7, -0x1.2fbbf80000000p-5, 0x1.965f4a0000000p-3, 0x1.f38a580000000p-3, 0x1.fe63f80000000p-9, -0x1.0b71180000000p-4}
, {0x1.8ae8720000000p-3, -0x1.4139f00000000p-3, -0x1.a85cdc0000000p-4, -0x1.fe67300000000p-4, -0x1.2422d60000000p-3, 0x1.b7cf0a0000000p-3, 0x1.5c382e0000000p-3, -0x1.a180a60000000p-4}
, {-0x1.cc8a720000000p-5, -0x1.6d1d060000000p-5, 0x1.534a5c0000000p-4, 0x1.8506760000000p-5, 0x1.4ede640000000p-4, -0x1.01ad480000000p-4, 0x1.5dc81e0000000p-5, -0x1.3a23c20000000p-3}
, {-0x1.c19b600000000p-4, 0x1.9b189e0000000p-4, 0x1.ab13940000000p-4, -0x1.8e241e0000000p-7, 0x1.4e330a0000000p-7, 0x1.5670be0000000p-5, -0x1.542fe60000000p-3, 0x1.e086bc0000000p-5}
, {-0x1.0aaf420000000p-3, 0x1.9560ac0000000p-4, -0x1.0b330e0000000p-4, 0x1.02ebae0000000p-2, -0x1.f0e6820000000p-10, -0x1.44a0dc0000000p-6, 0x1.02e6ec0000000p-10, 0x1.c66d500000000p-5}
}
, {{0x1.10b0b80000000p-3, 0x1.3c18140000000p-3, -0x1.b57d460000000p-4, 0x1.e8032a0000000p-5, -0x1.8a87020000000p-6, 0x1.e329600000000p-4, 0x1.cae2340000000p-4, 0x1.035a140000000p-3}
, {0x1.c25aa60000000p-4, -0x1.3bb5d60000000p-4, 0x1.132b240000000p-3, -0x1.924c8c0000000p-4, -0x1.a186ce0000000p-5, 0x1.df1ae00000000p-9, -0x1.2c98980000000p-3, 0x1.c546a80000000p-7}
, {-0x1.0e36c20000000p-3, 0x1.3896d00000000p-4, -0x1.ad5c8a0000000p-4, -0x1.8960ee0000000p-4, 0x1.70b0060000000p-3, 0x1.9ea85a0000000p-4, -0x1.315a760000000p-4, -0x1.cbb9680000000p-4}
, {0x1.c60c720000000p-3, 0x1.3916f80000000p-3, -0x1.48612e0000000p-4, -0x1.c7bf340000000p-4, 0x1.1654520000000p-3, 0x1.e85ac80000000p-4, -0x1.512d220000000p-4, 0x1.a807f80000000p-8}
, {0x1.1c94d60000000p-4, 0x1.10e28e0000000p-4, -0x1.3a6ce80000000p-3, -0x1.71c5b80000000p-7, 0x1.38fb940000000p-3, 0x1.735af40000000p-4, 0x1.8f16ec0000000p-3, -0x1.9df7fa0000000p-6}
, {0x1.51b1520000000p-5, -0x1.358e6e0000000p-7, 0x1.3674000000000p-3, -0x1.0e36f20000000p-5, -0x1.62656c0000000p-6, -0x1.8803b40000000p-5, 0x1.9b0c480000000p-3, 0x1.2c52ac0000000p-5}
, {0x1.90babc0000000p-3, -0x1.2aef800000000p-4, -0x1.46267c0000000p-3, -0x1.ef71b20000000p-4, -0x1.9cd2320000000p-4, -0x1.8399ea0000000p-3, 0x1.0c06d60000000p-7, 0x1.7788bc0000000p-4}
, {-0x1.8768100000000p-5, -0x1.356de60000000p-3, -0x1.7305a00000000p-4, 0x1.813b7a0000000p-7, -0x1.3ab45e0000000p-6, 0x1.1e77100000000p-4, 0x1.15c0920000000p-3, 0x1.35078a0000000p-3}
}
, {{0x1.5ea1e00000000p-4, 0x1.6c42120000000p-4, 0x1.942c4a0000000p-7, 0x1.20182e0000000p-4, 0x1.4f25cc0000000p-4, 0x1.96dcc40000000p-4, -0x1.1d54340000000p-5, 0x1.8dd0020000000p-3}
, {0x1.a312420000000p-3, 0x1.21d0720000000p-4, 0x1.d1964c0000000p-4, -0x1.2d1ca20000000p-5, 0x1.b346ce0000000p-4, 0x1.11980c0000000p-7, -0x1.3643c20000000p-4, 0x1.fbd5640000000p-10}
, {0x1.99b6ca0000000p-4, 0x1.ffb9a00000000p-6, -0x1.bacf3e0000000p-4, 0x1.0bf97e0000000p-4, 0x1.dcb4600000000p-5, 0x1.c643a20000000p-4, -0x1.d9a8ba0000000p-4, 0x1.86761c0000000p-3}
, {0x1.645e400000000p-4, 0x1.179f960000000p-3, 0x1.0520b00000000p-4, -0x1.eeaef80000000p-4, 0x1.d890f60000000p-12, -0x1.0d1fac0000000p-3, -0x1.b3c8400000000p-4, 0x1.956b5e0000000p-3}
, {-0x1.28d5880000000p-4, -0x1.7ccf020000000p-5, -0x1.2d0dc00000000p-5, 0x1.be6b580000000p-4, 0x1.9a2a660000000p-11, 0x1.1a19fe0000000p-4, 0x1.f6aae40000000p-4, 0x1.c20fee0000000p-3}
, {0x1.5aee460000000p-3, -0x1.d0cb9a0000000p-4, 0x1.44b8b00000000p-4, -0x1.428ce20000000p-3, -0x1.232d5a0000000p-3, 0x1.3f5cec0000000p-5, 0x1.f28fc00000000p-3, -0x1.1757300000000p-7}
, {0x1.10de7a0000000p-3, 0x1.0e0b580000000p-6, 0x1.ea34d20000000p-4, -0x1.7c3c640000000p-3, -0x1.1d04d20000000p-3, 0x1.878f5a0000000p-5, -0x1.3266860000000p-4, 0x1.be59960000000p-6}
, {0x1.16fca80000000p-4, -0x1.f9c1ca0000000p-4, -0x1.6cc1620000000p-5, -0x1.af22ba0000000p-4, -0x1.c8444a0000000p-5, 0x1.6599860000000p-3, 0x1.2ea7260000000p-3, 0x1.879a240000000p-4}
}
, {{-0x1.f687920000000p-6, 0x1.583b0c0000000p-4, -0x1.3abf840000000p-6, 0x1.c37dbe0000000p-4, -0x1.d3cff40000000p-5, 0x1.4736300000000p-5, -0x1.02b7cc0000000p-4, 0x1.fffa2c0000000p-4}
, {-0x1.23a7bc0000000p-3, -0x1.f52c7e0000000p-4, -0x1.b145800000000p-3, 0x1.021cc60000000p-3, -0x1.c00a620000000p-8, -0x1.1ae9000000000p-3, -0x1.eeccda0000000p-4, -0x1.69b7bc0000000p-3}
, {-0x1.bba0240000000p-4, -0x1.cf5aa00000000p-3, -0x1.e3f69c0000000p-4, -0x1.c28b2c0000000p-5, 0x1.56dc700000000p-6, 0x1.ebc08e0000000p-6, 0x1.f46a200000000p-3, -0x1.1688020000000p-3}
, {0x1.a2b18a0000000p-4, -0x1.250ae40000000p-3, 0x1.19df5c0000000p-3, -0x1.9d475e0000000p-3, 0x1.1ffdf20000000p-3, 0x1.9859d00000000p-3, -0x1.8ebd960000000p-4, 0x1.0237900000000p-2}
, {-0x1.2fe3160000000p-4, -0x1.15f26e0000000p-3, -0x1.d5f7240000000p-3, 0x1.064be40000000p-4, 0x1.c17b760000000p-3, -0x1.7801ba0000000p-5, 0x1.d335860000000p-6, 0x1.4915b20000000p-3}
, {-0x1.97c07a0000000p-3, -0x1.30126c0000000p-6, -0x1.8a19be0000000p-4, 0x1.ead1240000000p-5, 0x1.6336940000000p-2, -0x1.d99c140000000p-7, 0x1.22659a0000000p-3, 0x1.2b033a0000000p-4}
, {-0x1.8a720e0000000p-5, 0x1.573d900000000p-3, 0x1.09295c0000000p-5, 0x1.0ab82c0000000p-7, 0x1.fa60580000000p-3, -0x1.46da1e0000000p-4, 0x1.3c00e20000000p-4, 0x1.ade9f80000000p-3}
, {-0x1.710ce60000000p-6, -0x1.41862a0000000p-6, -0x1.1f425a0000000p-4, 0x1.73358c0000000p-5, -0x1.5a96f20000000p-6, 0x1.dbdf5c0000000p-4, -0x1.1f1b2e0000000p-3, 0x1.7db75a0000000p-5}
}
, {{0x1.5023c00000000p-4, 0x1.c2e9c20000000p-6, -0x1.e32e980000000p-4, 0x1.cc8ef20000000p-4, 0x1.8d16a80000000p-3, -0x1.2436b00000000p-3, -0x1.9c280c0000000p-3, 0x1.3c1f780000000p-3}
, {0x1.058d800000000p-4, 0x1.6bda720000000p-3, 0x1.310d840000000p-3, 0x1.7e9bd60000000p-4, 0x1.74cfbc0000000p-3, -0x1.668c5c0000000p-7, 0x1.7b14e00000000p-4, 0x1.9a1ae60000000p-4}
, {0x1.3915d40000000p-3, 0x1.52b5b40000000p-3, 0x1.bb8f780000000p-3, 0x1.c9feea0000000p-3, 0x1.cdefda0000000p-9, 0x1.0deb2a0000000p-3, 0x1.e5f4e00000000p-5, -0x1.84f8ca0000000p-7}
, {0x1.0f2d2c0000000p-4, 0x1.f5cc400000000p-3, 0x1.4c089c0000000p-3, -0x1.96de520000000p-4, -0x1.3859b00000000p-3, -0x1.73b5f40000000p-4, -0x1.2322180000000p-6, 0x1.2066860000000p-3}
, {0x1.90d5ac0000000p-3, -0x1.2271c60000000p-4, 0x1.fbbb520000000p-5, 0x1.8660040000000p-4, 0x1.0009ac0000000p-3, 0x1.6b986c0000000p-3, -0x1.06aeb00000000p-3, -0x1.277a4c0000000p-3}
, {-0x1.2f01c60000000p-4, -0x1.02b7ce0000000p-7, 0x1.022b8e0000000p-3, 0x1.7e5cf80000000p-5, 0x1.8426800000000p-3, 0x1.005a160000000p-3, 0x1.5dba760000000p-3, -0x1.11538c0000000p-3}
, {0x1.9451f40000000p-14, 0x1.2dd5fa0000000p-5, 0x1.1937620000000p-3, 0x1.5b2a600000000p-4, 0x1.f577600000000p-5, -0x1.a7f6680000000p-7, -0x1.a0fe460000000p-4, -0x1.9e97940000000p-3}
, {0x1.24c5b40000000p-3, 0x1.3d0c540000000p-3, 0x1.c9215e0000000p-4, 0x1.01de160000000p-3, 0x1.175ef20000000p-4, 0x1.67f0aa0000000p-4, -0x1.6a2c0c0000000p-3, 0x1.c087460000000p-4}
}
, {{0x1.64694c0000000p-3, 0x1.c3a0a00000000p-3, -0x1.e85ac20000000p-5, -0x1.704ad20000000p-6, -0x1.e4f3540000000p-4, -0x1.95609a0000000p-5, 0x1.91e8cc0000000p-4, 0x1.590cb20000000p-4}
, {-0x1.002ce00000000p-3, -0x1.ce82ac0000000p-5, 0x1.b0bba40000000p-3, -0x1.82ba040000000p-7, -0x1.0c13160000000p-5, -0x1.8e84160000000p-4, 0x1.21fab60000000p-8, 0x1.1808ce0000000p-7}
, {0x1.515bde0000000p-3, 0x1.912ef60000000p-3, 0x1.abd7680000000p-3, 0x1.04a84c0000000p-4, 0x1.145c420000000p-6, 0x1.636c960000000p-6, 0x1.753dd40000000p-4, 0x1.acc9c60000000p-5}
, {0x1.524d320000000p-3, 0x1.430f4c0000000p-4, 0x1.6403540000000p-4, -0x1.3e3ac80000000p-4, 0x1.4bdc060000000p-3, 0x1.511b0e0000000p-5, 0x1.545d120000000p-6, -0x1.b1a35e0000000p-3}
, {-0x1.059a5c0000000p-9, 0x1.d1923e0000000p-3, -0x1.0f30900000000p-3, -0x1.a7feac0000000p-6, -0x1.5351400000000p-5, 0x1.361c840000000p-4, 0x1.a5680e0000000p-4, -0x1.5a0f440000000p-4}
, {0x1.d743840000000p-4, 0x1.8e73a40000000p-5, -0x1.314a5c0000000p-4, 0x1.14c7fa0000000p-3, -0x1.2c32f20000000p-11, 0x1.c1883c0000000p-4, -0x1.79b7ac0000000p-3, -0x1.ae8bca0000000p-4}
, {0x1.5576c60000000p-3, 0x1.07bc960000000p-4, 0x1.9f44020000000p-3, -0x1.e3a1d20000000p-4, 0x1.4d985a0000000p-3, 0x1.a0638e0000000p-4, -0x1.77b06c0000000p-3, -0x1.d45f3c0000000p-6}
, {-0x1.37a4f60000000p-4, 0x1.500a580000000p-4, 0x1.40f13c0000000p-3, -0x1.28bc620000000p-3, 0x1.034c200000000p-3, 0x1.102a540000000p-5, -0x1.0e85de0000000p-4, -0x1.d94f1a0000000p-5}
}
, {{0x1.9c72900000000p-6, 0x1.2e08a60000000p-3, 0x1.a5f5240000000p-3, -0x1.4535300000000p-3, 0x1.64b6140000000p-3, 0x1.7b77d40000000p-4, -0x1.b5e9940000000p-5, -0x1.022d9a0000000p-3}
, {0x1.3aa3aa0000000p-4, 0x1.7dee180000000p-3, 0x1.1c9c6e0000000p-3, -0x1.25f4dc0000000p-3, 0x1.ff92f20000000p-6, 0x1.7e347e0000000p-5, -0x1.e3a3640000000p-9, 0x1.5bf6b00000000p-3}
, {-0x1.42cda80000000p-4, 0x1.1a3dec0000000p-4, -0x1.e627240000000p-6, -0x1.7a7e860000000p-3, 0x1.a045760000000p-3, -0x1.7269b00000000p-5, 0x1.794bc60000000p-3, 0x1.7848340000000p-5}
, {0x1.06171a0000000p-3, 0x1.a143800000000p-4, 0x1.bb4bd60000000p-3, -0x1.4268ea0000000p-3, 0x1.d5c27a0000000p-3, -0x1.3ab73a0000000p-4, 0x1.5ab8d20000000p-10, -0x1.b83a0c0000000p-4}
, {-0x1.ee449e0000000p-4, -0x1.74a7d80000000p-5, 0x1.eb40520000000p-4, -0x1.e3038c0000000p-4, -0x1.6a89600000000p-6, -0x1.173f160000000p-3, 0x1.3f697c0000000p-3, 0x1.a5e9400000000p-4}
, {0x1.ae20120000000p-4, 0x1.13acb40000000p-3, 0x1.7dc3e00000000p-4, -0x1.96867c0000000p-7, 0x1.b511400000000p-5, -0x1.6bec300000000p-3, 0x1.fcc84a0000000p-12, 0x1.66e6aa0000000p-4}
, {0x1.6d36a60000000p-4, -0x1.1cea980000000p-3, 0x1.7cc77e0000000p-3, -0x1.2878140000000p-3, 0x1.a49e620000000p-3, -0x1.04476a0000000p-3, 0x1.a3775c0000000p-3, 0x1.f5f5260000000p-4}
, {0x1.fef1540000000p-4, 0x1.c377140000000p-6, -0x1.33cefa0000000p-3, -0x1.7d0da80000000p-7, -0x1.b7b1f00000000p-5, 0x1.1579380000000p-4, 0x1.64faea0000000p-3, 0x1.60d04a0000000p-4}
}
, {{0x1.03d6ba0000000p-6, -0x1.b27f300000000p-5, 0x1.26c7bc0000000p-3, 0x1.555e180000000p-3, 0x1.d8dc6c0000000p-6, -0x1.1ef3c20000000p-3, 0x1.ab045e0000000p-4, -0x1.858d120000000p-6}
, {-0x1.bf6b840000000p-4, 0x1.5d48260000000p-7, 0x1.b3cb0e0000000p-4, 0x1.2316c00000000p-4, 0x1.6715900000000p-4, -0x1.e329ba0000000p-4, -0x1.8fe2780000000p-5, -0x1.d0ab6a0000000p-4}
, {-0x1.d5c4b40000000p-5, 0x1.d9d02a0000000p-4, 0x1.ac5bdc0000000p-4, -0x1.fe53f00000000p-5, 0x1.1fc1ba0000000p-3, 0x1.163a860000000p-4, -0x1.3eee280000000p-5, -0x1.3a7cf00000000p-4}
, {0x1.d8921e0000000p-5, 0x1.ebce9c0000000p-7, -0x1.4c95ec0000000p-3, -0x1.11850a0000000p-2, 0x1.23359a0000000p-3, 0x1.dfcfba0000000p-4, -0x1.f1ae6a0000000p-6, -0x1.02b5820000000p-5}
, {-0x1.6585580000000p-4, 0x1.9825cc0000000p-6, -0x1.8508340000000p-7, -0x1.1d77ac0000000p-3, 0x1.caafce0000000p-3, -0x1.cce77c0000000p-6, -0x1.f058fc0000000p-4, 0x1.55d68a0000000p-4}
, {0x1.00ff0c0000000p-3, -0x1.6799c20000000p-3, 0x1.6e3b460000000p-3, 0x1.465ffc0000000p-4, -0x1.8b818a0000000p-5, 0x1.469bf00000000p-8, -0x1.7978060000000p-3, 0x1.0cbb300000000p-5}
, {0x1.3b55800000000p-3, -0x1.3510600000000p-9, -0x1.62f7f00000000p-3, -0x1.8613640000000p-3, 0x1.91329c0000000p-3, 0x1.1cf3c60000000p-3, -0x1.3e82c60000000p-3, 0x1.d514740000000p-3}
, {0x1.da1ff20000000p-5, -0x1.ce7af60000000p-4, -0x1.0a1a700000000p-3, 0x1.8356240000000p-4, 0x1.170a080000000p-4, 0x1.547f0a0000000p-3, 0x1.093ce00000000p-3, 0x1.7837c00000000p-3}
}
, {{0x1.3f27de0000000p-3, -0x1.2ee8740000000p-4, -0x1.691d640000000p-3, -0x1.3aa6100000000p-6, -0x1.8447ee0000000p-4, -0x1.165ddc0000000p-4, 0x1.b5fdee0000000p-3, 0x1.baa9300000000p-3}
, {0x1.9798320000000p-4, 0x1.0a8e600000000p-3, 0x1.5112320000000p-6, 0x1.7c3ba00000000p-4, 0x1.8663fc0000000p-3, 0x1.8d17e40000000p-5, 0x1.4e24c60000000p-4, 0x1.64537e0000000p-3}
, {-0x1.21b6cc0000000p-3, -0x1.4918480000000p-7, -0x1.4fe4080000000p-3, -0x1.3b435a0000000p-3, -0x1.43708c0000000p-3, -0x1.b0a5ba0000000p-4, 0x1.e296320000000p-6, -0x1.65d98c0000000p-6}
, {-0x1.8ca89c0000000p-6, -0x1.f6e82c0000000p-4, -0x1.3984480000000p-3, -0x1.9261be0000000p-4, 0x1.1375ac0000000p-4, -0x1.a66cc40000000p-4, -0x1.614bec0000000p-3, 0x1.f2b48a0000000p-5}
, {0x1.3a3fcc0000000p-5, 0x1.8f02e00000000p-3, -0x1.87655e0000000p-4, 0x1.92ffe60000000p-3, 0x1.5e04400000000p-4, 0x1.a88bc80000000p-3, 0x1.5bc3ce0000000p-3, -0x1.e274dc0000000p-5}
, {-0x1.4bbfac0000000p-6, 0x1.9789aa0000000p-6, -0x1.46414c0000000p-3, 0x1.c05f940000000p-4, -0x1.84d5220000000p-5, -0x1.4645ae0000000p-4, 0x1.7077300000000p-3, 0x1.23cc460000000p-7}
, {-0x1.427e080000000p-6, -0x1.a206c20000000p-4, -0x1.0483860000000p-3, -0x1.c170d60000000p-4, -0x1.008e320000000p-3, 0x1.5025ce0000000p-3, 0x1.818b940000000p-3, 0x1.de7c660000000p-6}
, {0x1.7b806c0000000p-6, -0x1.50a6b60000000p-4, 0x1.00ffd00000000p-3, 0x1.00a0320000000p-6, -0x1.4bfd140000000p-3, -0x1.83d0e40000000p-4, 0x1.66340c0000000p-8, -0x1.ec7f380000000p-4}
}
, {{0x1.55abca0000000p-6, -0x1.7fdb260000000p-4, -0x1.6ed8b40000000p-4, -0x1.a9cdf60000000p-4, 0x1.9924e80000000p-4, 0x1.d1feba0000000p-4, 0x1.272ffc0000000p-3, -0x1.396a940000000p-3}
, {-0x1.0f37200000000p-4, -0x1.f421b40000000p-7, -0x1.54259e0000000p-3, 0x1.67a7cc0000000p-3, 0x1.7f30560000000p-4, -0x1.0b3ebc0000000p-3, 0x1.e63b540000000p-5, 0x1.c4ceb40000000p-3}
, {-0x1.00cf380000000p-4, -0x1.377bb40000000p-4, -0x1.de333e0000000p-4, 0x1.a14ce80000000p-3, -0x1.a626280000000p-4, 0x1.fcbb340000000p-6, 0x1.1e03be0000000p-3, 0x1.30e73e0000000p-7}
, {0x1.33a3ec0000000p-3, -0x1.40105c0000000p-3, 0x1.3bb1ba0000000p-3, 0x1.5fd7740000000p-3, -0x1.f9a0dc0000000p-4, 0x1.1d1da00000000p-4, -0x1.1f79ea0000000p-3, -0x1.1a46f80000000p-5}
, {0x1.631b100000000p-4, -0x1.349c080000000p-3, 0x1.2323180000000p-3, 0x1.6f59700000000p-3, -0x1.599bdc0000000p-5, 0x1.be79fe0000000p-4, -0x1.2b6f4c0000000p-3, 0x1.925aa00000000p-3}
, {0x1.61f0f60000000p-4, -0x1.61e46e0000000p-5, -0x1.94866c0000000p-4, 0x1.f0f2380000000p-10, 0x1.9daf380000000p-6, -0x1.bfa6f20000000p-3, 0x1.b3546a0000000p-3, 0x1.59a8460000000p-3}
, {-0x1.73f3d20000000p-4, 0x1.4f518a0000000p-4, 0x1.01d3820000000p-5, 0x1.794f660000000p-3, 0x1.f238000000000p-5, 0x1.6ccf2c0000000p-12, 0x1.4672940000000p-5, 0x1.8aba180000000p-3}
, {-0x1.09496c0000000p-3, -0x1.2117d40000000p-4, -0x1.c680840000000p-8, 0x1.50a8ec0000000p-4, -0x1.5ea6a60000000p-3, -0x1.6196120000000p-6, 0x1.95388e0000000p-3, 0x1.3c93d40000000p-4}
}
, {{0x1.0168b20000000p-3, -0x1.b4ccb20000000p-4, -0x1.164c200000000p-6, -0x1.74288c0000000p-5, 0x1.f0b3000000000p-3, 0x1.2689480000000p-4, -0x1.b26b2c0000000p-5, 0x1.88c6080000000p-3}
, {0x1.c810280000000p-3, -0x1.85d8280000000p-6, 0x1.6b8ea20000000p-3, 0x1.1702d40000000p-3, 0x1.eab5720000000p-3, 0x1.ded9e00000000p-4, -0x1.282ebc0000000p-6, 0x1.75d4e60000000p-4}
, {-0x1.5666960000000p-4, 0x1.de58620000000p-4, 0x1.b94c8c0000000p-4, 0x1.224e8e0000000p-4, 0x1.94463a0000000p-3, 0x1.d363ec0000000p-7, -0x1.bf4bc60000000p-3, 0x1.2ff3920000000p-4}
, {0x1.67c2340000000p-3, -0x1.964c440000000p-4, -0x1.f18ba00000000p-4, -0x1.20021a0000000p-3, 0x1.7181960000000p-4, 0x1.a4a3b60000000p-4, 0x1.4abec00000000p-5, -0x1.8e6fe80000000p-6}
, {-0x1.88b8d40000000p-4, 0x1.ae6b960000000p-3, 0x1.4dc03c0000000p-3, 0x1.76d93c0000000p-3, 0x1.6b15a20000000p-4, -0x1.858e040000000p-5, -0x1.38027a0000000p-3, 0x1.3ba98e0000000p-10}
, {0x1.5273440000000p-3, -0x1.01f4700000000p-3, 0x1.6d7d680000000p-3, -0x1.0a1cf00000000p-3, -0x1.ffb4980000000p-6, 0x1.d992220000000p-4, -0x1.dcd8500000000p-4, 0x1.4260b40000000p-5}
, {-0x1.2861b40000000p-7, 0x1.5c4a520000000p-3, -0x1.03d7b60000000p-5, 0x1.d4be960000000p-3, 0x1.6408e40000000p-3, 0x1.e7f51c0000000p-3, -0x1.03e9080000000p-3, 0x1.1e924a0000000p-3}
, {0x1.ca9c580000000p-7, 0x1.1b59000000000p-3, 0x1.43cc200000000p-3, -0x1.6ef26a0000000p-5, 0x1.7e79f40000000p-4, 0x1.75c99e0000000p-3, 0x1.abbbc00000000p-8, -0x1.577c060000000p-7}
}
, {{-0x1.c103320000000p-9, -0x1.067d7a0000000p-3, -0x1.dc33b60000000p-4, 0x1.26d8620000000p-3, -0x1.62076e0000000p-3, 0x1.43e6c20000000p-4, -0x1.de07860000000p-4, -0x1.3b7cb40000000p-4}
, {-0x1.47a6fc0000000p-3, 0x1.1d3c800000000p-3, 0x1.a6974e0000000p-3, 0x1.3fc21e0000000p-3, -0x1.0c297e0000000p-3, -0x1.0ce9120000000p-4, -0x1.50eb840000000p-3, -0x1.13d67e0000000p-3}
, {0x1.0dfd120000000p-3, -0x1.0bdc1c0000000p-3, -0x1.1ac83c0000000p-3, 0x1.a8b94c0000000p-3, -0x1.4fc0240000000p-5, -0x1.db877a0000000p-5, -0x1.f808ca0000000p-5, -0x1.b30ca80000000p-7}
, {-0x1.4686880000000p-4, 0x1.321a320000000p-4, 0x1.5315ac0000000p-4, 0x1.26157e0000000p-5, -0x1.0914600000000p-2, 0x1.7193880000000p-4, -0x1.e5c2880000000p-5, -0x1.213ee60000000p-3}
, {-0x1.1cfd400000000p-3, 0x1.03417c0000000p-8, -0x1.1c02f80000000p-5, 0x1.0e68dc0000000p-4, 0x1.59b6440000000p-6, 0x1.da45a60000000p-3, 0x1.c6a1780000000p-4, -0x1.ee00180000000p-4}
, {-0x1.abfc920000000p-4, 0x1.254a8c0000000p-11, 0x1.e2cdbe0000000p-3, 0x1.416af80000000p-3, -0x1.4b17ce0000000p-3, -0x1.e6dde40000000p-5, 0x1.615a640000000p-4, 0x1.4c5f580000000p-5}
, {0x1.aedae60000000p-4, 0x1.b453c60000000p-7, 0x1.407e480000000p-3, -0x1.1869600000000p-4, 0x1.c9d0e20000000p-5, 0x1.97c25c0000000p-4, -0x1.6fbce20000000p-4, -0x1.59702a0000000p-12}
, {-0x1.68884c0000000p-7, 0x1.d0bc020000000p-3, 0x1.5394820000000p-4, 0x1.31b0120000000p-3, -0x1.d669040000000p-5, 0x1.64e8aa0000000p-3, -0x1.5235aa0000000p-4, 0x1.3e25ba0000000p-3}
}
, {{0x1.88042a0000000p-5, 0x1.3974940000000p-3, 0x1.5d1ec80000000p-3, -0x1.3e5b1a0000000p-6, -0x1.6529520000000p-4, 0x1.d4affe0000000p-5, 0x1.4d67b60000000p-3, 0x1.86a6580000000p-3}
, {-0x1.367f0a0000000p-3, 0x1.4bd0c60000000p-3, -0x1.05ffea0000000p-4, -0x1.37d1a40000000p-4, -0x1.d222080000000p-4, -0x1.c8524c0000000p-4, 0x1.5f0e7e0000000p-3, 0x1.9b2df80000000p-7}
, {-0x1.950f7a0000000p-4, -0x1.6f82100000000p-4, -0x1.053bc00000000p-3, 0x1.0832fe0000000p-3, -0x1.5441220000000p-3, 0x1.948e6a0000000p-5, -0x1.2a3b6c0000000p-4, -0x1.1cdcde0000000p-4}
, {-0x1.15657c0000000p-5, 0x1.7fad8a0000000p-3, 0x1.cb3f040000000p-4, -0x1.7376240000000p-6, -0x1.36958e0000000p-3, -0x1.bc85aa0000000p-4, 0x1.70e87c0000000p-4, 0x1.0669920000000p-6}
, {-0x1.ee58ae0000000p-4, -0x1.01a5800000000p-3, 0x1.f5e69e0000000p-9, 0x1.0af3c00000000p-3, -0x1.e0c46e0000000p-4, -0x1.7026c80000000p-7, -0x1.b3d8660000000p-4, -0x1.4b623a0000000p-4}
, {0x1.45387c0000000p-5, 0x1.b02ae60000000p-6, 0x1.545a140000000p-3, -0x1.a3d3060000000p-4, -0x1.d913840000000p-3, -0x1.03f2540000000p-5, -0x1.0579100000000p-3, -0x1.035d760000000p-4}
, {-0x1.105ab60000000p-4, 0x1.73094a0000000p-3, 0x1.addf760000000p-3, 0x1.eef6100000000p-6, 0x1.4f4a480000000p-4, -0x1.0125e20000000p-3, -0x1.e92abe0000000p-5, 0x1.38f7ce0000000p-4}
, {-0x1.c168e20000000p-6, -0x1.54585e0000000p-3, 0x1.8251940000000p-4, -0x1.bad46a0000000p-4, 0x1.aa64720000000p-5, -0x1.a343920000000p-7, 0x1.3a4e900000000p-3, 0x1.3255ba0000000p-3}
}
, {{-0x1.185b200000000p-5, -0x1.93663a0000000p-5, -0x1.d740f00000000p-5, -0x1.e623ec0000000p-7, 0x1.6eb7f00000000p-4, 0x1.0d35760000000p-3, 0x1.50102c0000000p-4, 0x1.a209200000000p-4}
, {-0x1.2161be0000000p-5, 0x1.d5a5b00000000p-5, -0x1.2641300000000p-4, 0x1.3272e80000000p-5, -0x1.58ecc40000000p-8, 0x1.3109ea0000000p-2, -0x1.b2e02c0000000p-4, 0x1.4acee40000000p-5}
, {0x1.6955cc0000000p-3, -0x1.fca59e0000000p-9, -0x1.2483ac0000000p-3, 0x1.32e47e0000000p-3, -0x1.2400140000000p-3, 0x1.d682660000000p-5, -0x1.099ea00000000p-3, 0x1.d9804c0000000p-4}
, {-0x1.ce1c820000000p-8, -0x1.34322a0000000p-3, -0x1.bd17880000000p-4, -0x1.e28c460000000p-4, -0x1.ff7f120000000p-4, 0x1.2c12020000000p-3, -0x1.43f1900000000p-9, 0x1.9cbb580000000p-5}
, {0x1.6052460000000p-3, 0x1.ef1f4e0000000p-8, -0x1.39dc100000000p-5, -0x1.07977c0000000p-4, -0x1.dac11c0000000p-6, 0x1.bf6c6e0000000p-3, -0x1.0f0bea0000000p-3, 0x1.285b8c0000000p-3}
, {0x1.77f5b60000000p-4, -0x1.6fa70e0000000p-4, -0x1.e722520000000p-5, 0x1.93fefc0000000p-4, 0x1.ac38600000000p-5, 0x1.3dcce60000000p-3, -0x1.f582700000000p-3, -0x1.ea92320000000p-4}
, {-0x1.74ebc20000000p-5, -0x1.c716f20000000p-3, -0x1.a707a60000000p-7, -0x1.5cb54c0000000p-5, -0x1.5130e60000000p-4, 0x1.c4c2520000000p-4, 0x1.6c19ca0000000p-7, -0x1.f4f01a0000000p-5}
, {0x1.e5fb5e0000000p-3, -0x1.0989a60000000p-4, -0x1.94f2f00000000p-3, -0x1.f1adbc0000000p-4, -0x1.2fc9860000000p-5, -0x1.a7782e0000000p-4, -0x1.be5b9a0000000p-4, -0x1.d04d340000000p-5}
}
, {{0x1.a0222c0000000p-4, -0x1.00c9fa0000000p-4, 0x1.d1b7020000000p-4, -0x1.8c14a00000000p-5, -0x1.6eda160000000p-3, 0x1.c435780000000p-3, 0x1.3995720000000p-3, 0x1.c1bb180000000p-6}
, {-0x1.58e5420000000p-10, -0x1.e77eac0000000p-6, -0x1.440a800000000p-5, 0x1.2b597e0000000p-4, -0x1.acd0ac0000000p-4, 0x1.94eff60000000p-5, 0x1.861c400000000p-3, 0x1.b9ee4a0000000p-3}
, {-0x1.f7b4960000000p-4, 0x1.33c0740000000p-4, 0x1.180cf60000000p-3, -0x1.5d33fc0000000p-5, -0x1.0ccbf40000000p-3, 0x1.727fb80000000p-3, -0x1.e047a80000000p-5, 0x1.03ccf40000000p-3}
, {0x1.90c0260000000p-6, 0x1.2d0c180000000p-4, -0x1.aa9bfe0000000p-5, 0x1.27f50a0000000p-4, -0x1.946f660000000p-5, 0x1.1093cc0000000p-3, 0x1.cc69060000000p-7, 0x1.c0160a0000000p-3}
, {-0x1.17847a0000000p-4, 0x1.3f8b460000000p-6, -0x1.4c09160000000p-3, 0x1.bcb92e0000000p-4, 0x1.34e25a0000000p-7, 0x1.23ce920000000p-3, 0x1.ec8ada0000000p-5, -0x1.cd1eae0000000p-8}
, {-0x1.0b9ca40000000p-6, 0x1.81366c0000000p-4, -0x1.e4ba220000000p-5, 0x1.c87d220000000p-4, 0x1.4bca7a0000000p-7, 0x1.65514a0000000p-5, 0x1.fd56bc0000000p-5, -0x1.61f5840000000p-4}
, {-0x1.c3f2d00000000p-5, -0x1.5e062e0000000p-4, -0x1.3362700000000p-4, -0x1.4df4460000000p-3, -0x1.4bb3ec0000000p-3, -0x1.06e1360000000p-4, 0x1.895eee0000000p-3, 0x1.82af820000000p-4}
, {-0x1.fd09540000000p-6, 0x1.573a320000000p-3, -0x1.edca220000000p-3, -0x1.ac5dd80000000p-3, 0x1.0483de0000000p-3, -0x1.48ecfe0000000p-3, -0x1.690fbe0000000p-4, 0x1.b5b4d80000000p-4}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_GROUPS