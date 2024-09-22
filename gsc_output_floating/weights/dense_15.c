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
#define FC_UNITS 10


const float dense_15_bias[FC_UNITS] = {-0x1.2d18c40000000p-7, 0x1.171e2a0000000p-7, 0x1.efbbe20000000p-7, 0x1.4f1c580000000p-8, -0x1.e318b20000000p-7, -0x1.f2142c0000000p-6, 0x1.eaa2680000000p-6, 0x1.136c6e0000000p-7, -0x1.695ab60000000p-6, 0x1.032bf60000000p-8}
;

const float dense_15_kernel[FC_UNITS][INPUT_SAMPLES] = {{-0x1.4d5f640000000p-5, 0x1.0535240000000p-3, 0x1.95e9b00000000p-7, -0x1.34f98e0000000p-3, -0x1.6207980000000p-3, -0x1.2e1e100000000p-5, -0x1.5c769c0000000p-3, 0x1.04bf5e0000000p-3, 0x1.3b83a00000000p-3, -0x1.13e1bc0000000p-4, -0x1.b7dd9a0000000p-4, -0x1.e9c04c0000000p-4, 0x1.99e6820000000p-3, -0x1.40db820000000p-4, -0x1.10468e0000000p-4, -0x1.c584620000000p-4, 0x1.3123d00000000p-6, 0x1.8b90d60000000p-3, 0x1.5506360000000p-3, 0x1.49aa560000000p-5, -0x1.953bdc0000000p-3, 0x1.6c8fb40000000p-3, -0x1.97661c0000000p-3, 0x1.9deb140000000p-4, -0x1.f6bf160000000p-7, -0x1.5c20b60000000p-4, -0x1.6a75320000000p-4, 0x1.2279e20000000p-10, 0x1.d18d240000000p-4, -0x1.d226020000000p-7, -0x1.8380ac0000000p-5, -0x1.64a95a0000000p-3, -0x1.165d6c0000000p-4, -0x1.b3c1be0000000p-6, 0x1.18eb220000000p-6, 0x1.9300600000000p-4, -0x1.61404e0000000p-3, -0x1.d777fe0000000p-4, 0x1.b09d8c0000000p-3, 0x1.2ff0b20000000p-5, 0x1.4790940000000p-4, -0x1.27a6d40000000p-3, -0x1.bd52820000000p-5, -0x1.aada3e0000000p-5, -0x1.ef12ce0000000p-4, -0x1.7242080000000p-4, 0x1.4d78ba0000000p-7, -0x1.541dd20000000p-11, 0x1.1fbdec0000000p-3, 0x1.6f741c0000000p-4, -0x1.785cac0000000p-3, 0x1.7143740000000p-7, -0x1.6983360000000p-3, -0x1.26719e0000000p-3, 0x1.19e35a0000000p-7, -0x1.7ac8800000000p-4, -0x1.ba7c1a0000000p-4, -0x1.169cb80000000p-4, -0x1.6671e60000000p-4, 0x1.1ec9380000000p-6, 0x1.2c9e500000000p-5, -0x1.a565a80000000p-4, 0x1.2a59d00000000p-3, 0x1.0230720000000p-4, 0x1.5e81ce0000000p-3, 0x1.217f320000000p-3, -0x1.38e1600000000p-3, 0x1.9665120000000p-5, 0x1.ca44860000000p-7, 0x1.7792640000000p-7, 0x1.cab1320000000p-6, 0x1.0f1c480000000p-7, -0x1.a009d60000000p-4, -0x1.4c6f0c0000000p-3, -0x1.33b8080000000p-6, -0x1.6172460000000p-5, -0x1.5032b60000000p-3, -0x1.9c38320000000p-3, -0x1.2c345a0000000p-5, -0x1.1d45c40000000p-3, 0x1.8096cc0000000p-3, 0x1.14c0a00000000p-3, 0x1.990aa80000000p-6, -0x1.ce94280000000p-4, -0x1.24f5400000000p-3, -0x1.0067460000000p-4, 0x1.2de9d60000000p-3, 0x1.1de8460000000p-3, 0x1.3053f60000000p-5, 0x1.843e8c0000000p-6, -0x1.34703a0000000p-3, -0x1.997c320000000p-7, -0x1.5cb1060000000p-4, -0x1.c7d4500000000p-6, -0x1.2323a40000000p-3, 0x1.a34e9c0000000p-5, 0x1.52c4c20000000p-4, 0x1.4384c80000000p-3, 0x1.046e040000000p-3, -0x1.317b460000000p-4, 0x1.569dc40000000p-8, 0x1.37324a0000000p-5, -0x1.a9f8c20000000p-3, 0x1.dddaec0000000p-4, 0x1.426d940000000p-5, 0x1.44e2820000000p-4, -0x1.624fbc0000000p-4, -0x1.7428b80000000p-3, 0x1.4d14820000000p-3, 0x1.997c600000000p-3, -0x1.6add800000000p-4, 0x1.fc0d5a0000000p-5, 0x1.9ed3cc0000000p-4, -0x1.75ffd60000000p-4, 0x1.748ed80000000p-3, 0x1.84c3b80000000p-3, 0x1.908a4a0000000p-3, 0x1.70cfda0000000p-4, -0x1.c6038e0000000p-4, -0x1.07f72a0000000p-3, -0x1.7d5fbc0000000p-4, -0x1.1a31a20000000p-3, -0x1.c300e80000000p-4, -0x1.d45b680000000p-4, -0x1.37adaa0000000p-5, -0x1.574a400000000p-4, 0x1.fb8d420000000p-4, 0x1.c932540000000p-4}
, {0x1.5fe2ac0000000p-4, -0x1.a6062a0000000p-4, -0x1.7431fc0000000p-9, -0x1.029d960000000p-3, 0x1.97163c0000000p-3, 0x1.3590080000000p-3, -0x1.f721c20000000p-4, -0x1.3914c20000000p-6, -0x1.7cff4a0000000p-3, -0x1.98549c0000000p-3, -0x1.98aeca0000000p-7, -0x1.f2ba540000000p-4, 0x1.89debe0000000p-5, -0x1.58f0d40000000p-4, -0x1.4b3c3c0000000p-3, -0x1.9ac4180000000p-5, 0x1.4d751a0000000p-3, 0x1.6137620000000p-5, 0x1.1467fe0000000p-4, -0x1.e0fd9a0000000p-5, -0x1.9aa0c40000000p-4, -0x1.6b63bc0000000p-5, 0x1.b2086e0000000p-4, -0x1.f7a0da0000000p-4, -0x1.ae94440000000p-3, -0x1.0892d00000000p-3, -0x1.ab1ef40000000p-4, 0x1.a7272c0000000p-3, 0x1.2c4a900000000p-3, 0x1.3ff7e20000000p-4, -0x1.ddc36c0000000p-8, 0x1.8448140000000p-3, 0x1.ab93d80000000p-4, 0x1.64cfbe0000000p-4, 0x1.500f2e0000000p-3, 0x1.30a4260000000p-3, -0x1.ef6a4a0000000p-4, 0x1.ea69a00000000p-4, 0x1.af9cd60000000p-4, 0x1.9657d60000000p-5, 0x1.5691140000000p-3, 0x1.c25af80000000p-5, -0x1.f304620000000p-4, -0x1.6e7f7e0000000p-6, 0x1.77dd100000000p-3, 0x1.ac41440000000p-6, -0x1.e7164c0000000p-5, -0x1.7b00dc0000000p-5, -0x1.eb88a60000000p-5, -0x1.4f3cfc0000000p-3, -0x1.8958e00000000p-6, -0x1.432c220000000p-3, -0x1.4687100000000p-4, 0x1.726b8e0000000p-4, -0x1.0eb2120000000p-3, -0x1.cf409a0000000p-5, 0x1.0d9bca0000000p-4, -0x1.ed3e380000000p-4, 0x1.5088dc0000000p-3, 0x1.5588aa0000000p-4, 0x1.ded6280000000p-6, 0x1.7d2e7e0000000p-3, -0x1.5d64c60000000p-3, 0x1.09ad300000000p-3, -0x1.a8ca4a0000000p-4, -0x1.384fc60000000p-4, 0x1.5870220000000p-3, 0x1.60501e0000000p-5, -0x1.fa83ee0000000p-4, 0x1.2b1c680000000p-4, 0x1.2f03ae0000000p-4, 0x1.54ffc00000000p-7, 0x1.564f380000000p-5, 0x1.54ec860000000p-5, -0x1.6670760000000p-5, -0x1.0adad00000000p-3, -0x1.380b9a0000000p-4, -0x1.1f663c0000000p-7, 0x1.13898a0000000p-3, 0x1.893b360000000p-4, -0x1.91efdc0000000p-4, -0x1.cb8b500000000p-6, -0x1.95151a0000000p-7, -0x1.5583780000000p-6, -0x1.a43b9a0000000p-4, 0x1.9573240000000p-4, 0x1.8394f20000000p-3, 0x1.9a87a80000000p-3, -0x1.72431c0000000p-4, 0x1.a4fe300000000p-3, 0x1.d7ef5c0000000p-4, -0x1.d480c20000000p-4, 0x1.ada0e80000000p-3, 0x1.8ff6340000000p-5, -0x1.1649fe0000000p-3, -0x1.6423600000000p-3, -0x1.668f220000000p-4, 0x1.e677fa0000000p-4, -0x1.75ba380000000p-4, -0x1.512c880000000p-5, 0x1.af96ec0000000p-8, -0x1.7e28b20000000p-4, -0x1.35747c0000000p-3, 0x1.07d02c0000000p-3, -0x1.10601c0000000p-4, -0x1.7b6e260000000p-3, -0x1.ea7a420000000p-6, 0x1.834dfa0000000p-3, 0x1.8a5c360000000p-3, 0x1.0c71b20000000p-5, -0x1.9955700000000p-3, -0x1.a309180000000p-4, -0x1.38290a0000000p-3, 0x1.e644760000000p-5, 0x1.99bfca0000000p-5, 0x1.9bfc6c0000000p-3, -0x1.737dac0000000p-4, -0x1.6dd6900000000p-4, 0x1.13f34c0000000p-3, -0x1.5ce4b40000000p-7, -0x1.f89cc40000000p-6, 0x1.29b8fc0000000p-5, 0x1.c20ba20000000p-5, -0x1.45d6600000000p-4, -0x1.3982360000000p-3, -0x1.a44b100000000p-3, -0x1.3e72fa0000000p-4, -0x1.13a5640000000p-3}
, {-0x1.7023440000000p-4, -0x1.97f7680000000p-6, 0x1.6855b20000000p-3, 0x1.4295000000000p-5, 0x1.cb4d300000000p-5, -0x1.36dcf60000000p-4, -0x1.e4c4340000000p-4, -0x1.c700fc0000000p-6, 0x1.7a324a0000000p-5, -0x1.4c28ae0000000p-5, 0x1.0f199c0000000p-8, -0x1.76a3a20000000p-7, 0x1.b63e400000000p-3, 0x1.ced2e00000000p-5, 0x1.4ba4a60000000p-3, 0x1.073dae0000000p-5, -0x1.632be60000000p-3, 0x1.17a2260000000p-4, -0x1.3240200000000p-5, 0x1.b119740000000p-4, -0x1.5feb6a0000000p-7, -0x1.1537d40000000p-5, -0x1.edca2c0000000p-4, 0x1.211d980000000p-3, -0x1.8affb60000000p-6, -0x1.ea19ae0000000p-4, 0x1.9108400000000p-3, -0x1.ab2da20000000p-3, -0x1.f1a1da0000000p-4, -0x1.358eae0000000p-3, 0x1.6c33140000000p-3, -0x1.6d87b60000000p-3, 0x1.635f0c0000000p-4, 0x1.28a4480000000p-3, -0x1.33e00c0000000p-3, 0x1.617aac0000000p-5, -0x1.d39e980000000p-6, 0x1.1fbbf40000000p-4, -0x1.04a3860000000p-3, -0x1.ffb32c0000000p-9, -0x1.a0681c0000000p-5, -0x1.afde6a0000000p-4, 0x1.dc26140000000p-4, -0x1.9fd8560000000p-3, -0x1.74a85e0000000p-3, 0x1.8c2a4e0000000p-4, 0x1.9d606a0000000p-9, -0x1.3bfd900000000p-3, 0x1.44a2060000000p-4, -0x1.5cb64c0000000p-9, 0x1.cc4f320000000p-4, -0x1.34f00e0000000p-5, 0x1.76bbb20000000p-3, 0x1.b946720000000p-3, -0x1.f8dd060000000p-4, 0x1.3fbdea0000000p-3, -0x1.2135f40000000p-4, -0x1.7f9e640000000p-4, -0x1.1e6ab60000000p-3, -0x1.0092920000000p-3, 0x1.01a2ee0000000p-4, 0x1.18d6b40000000p-5, -0x1.14626e0000000p-3, 0x1.5b772a0000000p-3, -0x1.e34cde0000000p-6, 0x1.9897880000000p-5, 0x1.3d16ea0000000p-5, 0x1.4f93900000000p-3, 0x1.5451ee0000000p-4, 0x1.3ceb440000000p-3, 0x1.4d37260000000p-3, -0x1.294f240000000p-4, -0x1.54acdc0000000p-4, 0x1.61dc6a0000000p-8, 0x1.890af40000000p-3, 0x1.2a9bea0000000p-3, -0x1.8c5db80000000p-3, -0x1.a319cc0000000p-4, -0x1.260c4c0000000p-4, -0x1.b0dcb40000000p-4, 0x1.0b7d840000000p-3, 0x1.fba80c0000000p-4, 0x1.9afc7e0000000p-3, 0x1.58775c0000000p-5, -0x1.069dec0000000p-3, 0x1.bbff680000000p-6, -0x1.23e04e0000000p-5, 0x1.adb0600000000p-4, -0x1.5044280000000p-6, -0x1.8733e00000000p-4, 0x1.fc3eba0000000p-4, 0x1.2bcbc00000000p-6, -0x1.c4993c0000000p-9, -0x1.01fc440000000p-3, -0x1.74d3f40000000p-3, -0x1.c5d6040000000p-5, 0x1.b64b3a0000000p-3, -0x1.6fa5a40000000p-3, 0x1.8bc1060000000p-3, -0x1.b7a2600000000p-3, 0x1.a19be60000000p-5, -0x1.79a5580000000p-5, -0x1.0b43720000000p-3, -0x1.f461780000000p-5, 0x1.0ba11c0000000p-3, -0x1.b477a60000000p-4, 0x1.74d2b80000000p-4, 0x1.f758a60000000p-4, -0x1.94682e0000000p-3, 0x1.deb7b20000000p-5, -0x1.749ab80000000p-7, -0x1.163a8a0000000p-3, 0x1.61c10c0000000p-5, 0x1.8b0efe0000000p-3, 0x1.674cb60000000p-4, 0x1.acf80e0000000p-6, -0x1.6c310a0000000p-4, -0x1.0ccd580000000p-6, -0x1.adcfc60000000p-4, 0x1.0eae980000000p-3, -0x1.0d7b560000000p-3, -0x1.b500520000000p-5, -0x1.8221de0000000p-4, 0x1.46900a0000000p-4, 0x1.52c4d80000000p-3, -0x1.5304d60000000p-4, -0x1.413aee0000000p-6, -0x1.55be9a0000000p-5}
, {0x1.737a9a0000000p-3, 0x1.251baa0000000p-6, 0x1.88fbd80000000p-4, 0x1.b12a340000000p-7, -0x1.8826f20000000p-4, 0x1.9b291c0000000p-3, -0x1.5681840000000p-3, 0x1.02a3e00000000p-4, -0x1.2928500000000p-3, -0x1.7678cc0000000p-3, -0x1.9d9ebe0000000p-5, -0x1.2f38f00000000p-4, 0x1.621f1e0000000p-3, 0x1.72db7e0000000p-3, -0x1.b3a6160000000p-6, -0x1.2efc420000000p-3, 0x1.0fc53a0000000p-3, -0x1.4bcb100000000p-4, -0x1.68d6760000000p-3, -0x1.ef6a6a0000000p-4, 0x1.9ac3a60000000p-3, -0x1.53ba640000000p-4, 0x1.527d5a0000000p-4, 0x1.6343580000000p-4, 0x1.271af80000000p-8, 0x1.889cb40000000p-4, 0x1.94abb40000000p-4, 0x1.5d3e9c0000000p-4, 0x1.e6062c0000000p-5, 0x1.7fcda40000000p-6, 0x1.b8e79c0000000p-5, 0x1.b7a1700000000p-8, 0x1.fa44d60000000p-6, 0x1.88c1f20000000p-3, 0x1.dbb5b80000000p-6, -0x1.2964220000000p-3, 0x1.4bd44c0000000p-3, 0x1.38a8fe0000000p-3, 0x1.e35ece0000000p-4, -0x1.454ea40000000p-3, 0x1.5f5dde0000000p-6, -0x1.2f47ea0000000p-4, 0x1.dde0740000000p-8, 0x1.25183a0000000p-4, -0x1.8ed4fc0000000p-3, -0x1.a5c25a0000000p-3, -0x1.2109780000000p-4, 0x1.cc6d820000000p-10, 0x1.766d760000000p-10, -0x1.ac547c0000000p-4, 0x1.9fb22a0000000p-4, 0x1.348bfa0000000p-3, 0x1.dae9620000000p-5, -0x1.5a4c3a0000000p-4, -0x1.b171a00000000p-3, 0x1.48f2d20000000p-3, -0x1.c516a20000000p-4, 0x1.d178720000000p-5, 0x1.ecf11e0000000p-5, -0x1.d14ec00000000p-4, 0x1.c0db5a0000000p-4, 0x1.cfb2360000000p-5, 0x1.c94e5a0000000p-4, -0x1.8937860000000p-5, 0x1.8805280000000p-6, -0x1.5496a20000000p-5, 0x1.86958e0000000p-4, 0x1.8c29600000000p-4, -0x1.aa7b700000000p-3, 0x1.0810280000000p-5, 0x1.0e9bb00000000p-3, -0x1.18d3ca0000000p-5, -0x1.3ff2e20000000p-3, 0x1.22dba40000000p-4, 0x1.c543160000000p-4, 0x1.b598f60000000p-5, -0x1.31d9300000000p-3, -0x1.1a7eae0000000p-5, 0x1.f7fb700000000p-7, 0x1.f1d8d60000000p-4, 0x1.b735420000000p-8, -0x1.a2d6b40000000p-3, -0x1.7562160000000p-5, -0x1.793eda0000000p-4, 0x1.f5d16c0000000p-5, -0x1.51e3cc0000000p-5, -0x1.3468940000000p-5, -0x1.c14a0c0000000p-5, -0x1.d279d60000000p-4, -0x1.033e360000000p-6, -0x1.0e5aae0000000p-5, -0x1.c8258e0000000p-5, 0x1.ac309e0000000p-5, 0x1.42ca120000000p-5, -0x1.ab99e60000000p-4, -0x1.0db9c60000000p-4, 0x1.6f17560000000p-5, -0x1.2b75f20000000p-3, 0x1.0af1ea0000000p-3, -0x1.3dc12a0000000p-4, -0x1.6121700000000p-5, 0x1.bd8ca40000000p-3, 0x1.6d892e0000000p-4, -0x1.214c420000000p-4, -0x1.5f76220000000p-3, 0x1.5839f20000000p-6, 0x1.83a9d00000000p-3, -0x1.2c56960000000p-3, -0x1.40b04e0000000p-7, 0x1.4704240000000p-3, 0x1.66c0c00000000p-3, 0x1.50a9ea0000000p-4, 0x1.2ef8740000000p-4, -0x1.e6acf80000000p-6, 0x1.a416d20000000p-3, -0x1.c1c54e0000000p-4, -0x1.0759440000000p-4, 0x1.a151000000000p-4, -0x1.14e4ac0000000p-3, -0x1.679b7c0000000p-3, 0x1.f3db040000000p-4, -0x1.68f98c0000000p-3, 0x1.94a6140000000p-3, -0x1.56849c0000000p-9, -0x1.7c228c0000000p-9, -0x1.0d1a3e0000000p-4, 0x1.7000560000000p-4, 0x1.f770040000000p-5}
, {-0x1.100bf60000000p-4, -0x1.bfbe0c0000000p-7, 0x1.09fe1c0000000p-4, -0x1.3514c20000000p-3, 0x1.3039700000000p-3, -0x1.6771640000000p-3, 0x1.9852aa0000000p-3, -0x1.f9e2380000000p-4, 0x1.6f3cb80000000p-14, -0x1.abb45e0000000p-5, 0x1.4243020000000p-5, -0x1.298cb20000000p-3, -0x1.0c5ef60000000p-3, 0x1.5f85c60000000p-3, 0x1.7622800000000p-3, 0x1.512b120000000p-5, 0x1.5845fe0000000p-3, -0x1.5e666c0000000p-4, 0x1.8e16660000000p-3, -0x1.7106a80000000p-3, 0x1.36949a0000000p-4, 0x1.d8c1320000000p-4, -0x1.edf69e0000000p-4, 0x1.930d300000000p-4, 0x1.7a39060000000p-3, -0x1.86d2d00000000p-3, -0x1.6f51e20000000p-3, -0x1.f17b860000000p-4, 0x1.3af98a0000000p-3, -0x1.db39580000000p-5, -0x1.2863040000000p-3, 0x1.c4be640000000p-4, 0x1.db58940000000p-4, 0x1.f20d260000000p-4, -0x1.e944660000000p-4, 0x1.5383d80000000p-3, 0x1.ee9a000000000p-4, -0x1.cebbf80000000p-4, 0x1.42bff00000000p-4, 0x1.0cbc660000000p-5, 0x1.9a631a0000000p-3, -0x1.36ac980000000p-3, 0x1.447d680000000p-4, 0x1.b1dc320000000p-6, -0x1.b073520000000p-4, 0x1.0ec2f60000000p-3, -0x1.8292740000000p-3, -0x1.3d85b60000000p-3, -0x1.5cf4b40000000p-3, -0x1.87cdba0000000p-3, -0x1.b3c3700000000p-4, -0x1.6fd5380000000p-4, -0x1.48a6120000000p-3, -0x1.1f6b2e0000000p-3, 0x1.71b2dc0000000p-6, 0x1.0587940000000p-6, -0x1.984daa0000000p-3, -0x1.1c4cea0000000p-3, 0x1.ba16560000000p-7, 0x1.ecd2fe0000000p-4, 0x1.88035c0000000p-4, 0x1.ab0fd60000000p-4, -0x1.99a8820000000p-3, -0x1.d6718c0000000p-4, -0x1.89ee8e0000000p-3, 0x1.074dba0000000p-5, -0x1.6a72180000000p-3, -0x1.a46c340000000p-3, 0x1.9ffac80000000p-4, 0x1.c377820000000p-4, 0x1.6b45d80000000p-3, -0x1.6c7e280000000p-4, 0x1.7193b60000000p-4, -0x1.344ed00000000p-4, -0x1.8689660000000p-5, -0x1.1af9900000000p-6, 0x1.a0a46e0000000p-6, 0x1.ecdd4e0000000p-4, 0x1.4095f40000000p-3, 0x1.8bfdd40000000p-3, 0x1.77d7040000000p-4, 0x1.d0c35e0000000p-4, -0x1.69bd6a0000000p-3, 0x1.9a2ddc0000000p-3, 0x1.d0d2220000000p-4, 0x1.a77fae0000000p-5, 0x1.eed1700000000p-4, 0x1.5d6a2e0000000p-5, -0x1.33482e0000000p-3, -0x1.1a1c400000000p-3, -0x1.32a8e00000000p-4, 0x1.7681760000000p-3, 0x1.5d76480000000p-3, 0x1.22d7b40000000p-4, 0x1.c8c7220000000p-4, -0x1.e0238c0000000p-6, -0x1.374fb80000000p-3, 0x1.350e5c0000000p-3, -0x1.5721840000000p-3, -0x1.f0737c0000000p-4, 0x1.8a8b320000000p-3, 0x1.29d7840000000p-3, 0x1.dd2db60000000p-5, 0x1.1ec2060000000p-3, 0x1.2f22ec0000000p-4, 0x1.f567600000000p-4, 0x1.5c37f40000000p-4, 0x1.fb47800000000p-4, -0x1.495d6a0000000p-3, -0x1.702b1e0000000p-4, 0x1.9ad1aa0000000p-4, -0x1.4951760000000p-3, 0x1.7b80440000000p-3, -0x1.8eab000000000p-3, -0x1.9e2cde0000000p-3, 0x1.3b73360000000p-5, 0x1.b115b20000000p-5, -0x1.ac00ca0000000p-6, -0x1.d368c00000000p-4, 0x1.bf4e6c0000000p-4, -0x1.181e860000000p-4, 0x1.6054ca0000000p-4, -0x1.ada3320000000p-3, -0x1.13e8f00000000p-4, -0x1.4bae7a0000000p-3, 0x1.4060700000000p-4, -0x1.7597cc0000000p-3, -0x1.14ec120000000p-5}
, {0x1.9d69ce0000000p-9, 0x1.3362000000000p-3, -0x1.1684d80000000p-7, 0x1.c1903a0000000p-5, 0x1.99ea0c0000000p-3, 0x1.dbf2980000000p-4, -0x1.8ea65e0000000p-3, -0x1.90c3380000000p-7, 0x1.d9774a0000000p-4, 0x1.33d5600000000p-3, 0x1.8fb3fc0000000p-3, -0x1.abe3340000000p-5, -0x1.3525ee0000000p-9, -0x1.6a0ae40000000p-3, 0x1.8813880000000p-7, -0x1.3b205a0000000p-5, -0x1.2730fa0000000p-3, -0x1.293b720000000p-4, 0x1.b2de320000000p-3, -0x1.9031940000000p-3, 0x1.abb1700000000p-7, -0x1.e8cb100000000p-9, 0x1.46437c0000000p-4, -0x1.0efaea0000000p-5, -0x1.84408a0000000p-3, -0x1.819a900000000p-3, 0x1.1ae9de0000000p-3, 0x1.4285800000000p-3, 0x1.10be540000000p-4, 0x1.28d3120000000p-3, 0x1.74e2180000000p-6, 0x1.9654da0000000p-4, -0x1.8379760000000p-5, -0x1.18f56c0000000p-3, 0x1.978f980000000p-3, -0x1.6ef43c0000000p-3, 0x1.7caf360000000p-7, -0x1.87e1480000000p-3, 0x1.799a840000000p-4, 0x1.1b19aa0000000p-8, -0x1.fc464a0000000p-4, 0x1.ef35300000000p-4, 0x1.2533c20000000p-4, 0x1.37bafa0000000p-10, 0x1.dcd61c0000000p-10, -0x1.add1900000000p-3, 0x1.66eaca0000000p-8, 0x1.f6554a0000000p-6, 0x1.2fb9440000000p-5, 0x1.9e05d40000000p-5, -0x1.f111020000000p-5, -0x1.9eb8cc0000000p-3, 0x1.3bef4e0000000p-5, -0x1.67c69a0000000p-6, 0x1.c503220000000p-5, 0x1.5eaeae0000000p-3, 0x1.e7b3ca0000000p-4, -0x1.1399600000000p-3, 0x1.8599ba0000000p-4, 0x1.743b780000000p-3, -0x1.ab448c0000000p-3, 0x1.ba35400000000p-3, 0x1.74fdbe0000000p-3, 0x1.00f5400000000p-4, -0x1.4246c40000000p-3, -0x1.8db91a0000000p-3, -0x1.2e828e0000000p-3, 0x1.4695820000000p-5, -0x1.ca541a0000000p-4, 0x1.1e376a0000000p-3, 0x1.e5683c0000000p-4, -0x1.5c2af40000000p-8, 0x1.0646320000000p-3, 0x1.aef91c0000000p-4, 0x1.66e7b20000000p-3, 0x1.2aa2f80000000p-3, 0x1.270e520000000p-3, -0x1.83bdfc0000000p-5, -0x1.1157340000000p-5, 0x1.6b471a0000000p-3, 0x1.cad8740000000p-4, -0x1.f49f7e0000000p-4, 0x1.aec71c0000000p-4, 0x1.14213a0000000p-3, 0x1.61110a0000000p-3, -0x1.3ea91a0000000p-3, -0x1.71b7d40000000p-3, -0x1.2873100000000p-4, 0x1.e4710a0000000p-4, 0x1.b0c5e20000000p-5, -0x1.4d86700000000p-4, 0x1.1592a20000000p-3, 0x1.12c0940000000p-5, 0x1.65a89c0000000p-3, -0x1.998d5a0000000p-3, -0x1.b9c2c80000000p-4, 0x1.5f96140000000p-5, 0x1.3bc9620000000p-4, 0x1.2556860000000p-6, -0x1.a862a80000000p-3, 0x1.6142f00000000p-4, 0x1.d651320000000p-4, 0x1.97b7fa0000000p-3, 0x1.d51c360000000p-5, -0x1.7c873c0000000p-4, 0x1.1ea1cc0000000p-6, -0x1.ea1d380000000p-5, -0x1.41fcb60000000p-5, -0x1.f4146a0000000p-4, -0x1.e6664e0000000p-4, -0x1.a134720000000p-4, -0x1.17398a0000000p-3, -0x1.1bded40000000p-4, -0x1.9e0f320000000p-3, -0x1.905e500000000p-6, -0x1.173d380000000p-3, 0x1.7c1c1a0000000p-4, 0x1.e56a7a0000000p-7, -0x1.3c371c0000000p-6, 0x1.a9c5ce0000000p-3, -0x1.a4fc2a0000000p-3, -0x1.982a840000000p-7, 0x1.553d9c0000000p-7, -0x1.1c925e0000000p-3, 0x1.a15c7c0000000p-3, 0x1.88a5b20000000p-4, 0x1.1b11860000000p-4, 0x1.313a120000000p-3}
, {-0x1.d3983c0000000p-4, 0x1.ea4ab00000000p-4, -0x1.9ffc3a0000000p-3, -0x1.36fdd80000000p-4, 0x1.35ae400000000p-3, -0x1.e5a3ae0000000p-6, -0x1.90912a0000000p-4, 0x1.4965000000000p-4, 0x1.32f28a0000000p-6, -0x1.71caea0000000p-4, 0x1.2fedde0000000p-4, 0x1.fed1860000000p-4, 0x1.a402940000000p-5, -0x1.5ae8ac0000000p-3, 0x1.6e17880000000p-4, 0x1.3536600000000p-4, -0x1.f951d80000000p-9, -0x1.20fcac0000000p-4, -0x1.80b0de0000000p-4, 0x1.66530e0000000p-3, -0x1.a617960000000p-3, 0x1.b10a9a0000000p-3, -0x1.82d6f20000000p-3, -0x1.b8ae9c0000000p-4, 0x1.31f3c40000000p-3, -0x1.6f98260000000p-3, 0x1.8fa6580000000p-5, -0x1.64f1ae0000000p-3, -0x1.45f0ac0000000p-5, -0x1.b69e000000000p-5, 0x1.6a92d60000000p-3, -0x1.0d20de0000000p-3, -0x1.19f4f80000000p-5, -0x1.bbad8c0000000p-3, -0x1.008e600000000p-3, -0x1.0a2ca40000000p-3, -0x1.51b2340000000p-4, 0x1.f5cad00000000p-4, -0x1.7e06500000000p-5, 0x1.5a972e0000000p-4, 0x1.708ec60000000p-4, -0x1.0867540000000p-3, -0x1.300fd40000000p-4, 0x1.ae81540000000p-3, -0x1.af6cd60000000p-3, 0x1.5a7fa40000000p-8, -0x1.d7996a0000000p-3, 0x1.36851a0000000p-3, 0x1.4f959c0000000p-4, -0x1.9841b40000000p-3, 0x1.8b1c420000000p-3, 0x1.bca52a0000000p-3, 0x1.585c0e0000000p-3, 0x1.460aee0000000p-3, 0x1.99f13a0000000p-3, 0x1.a382d20000000p-5, 0x1.0ccda40000000p-3, 0x1.b7ff440000000p-5, 0x1.9f5a520000000p-5, 0x1.aeb9620000000p-5, -0x1.9e04760000000p-8, 0x1.5923dc0000000p-3, -0x1.c6dd260000000p-4, 0x1.a396360000000p-8, 0x1.0447860000000p-3, -0x1.2196860000000p-4, -0x1.61c2fc0000000p-3, -0x1.2bc5540000000p-4, -0x1.dece720000000p-4, 0x1.19c2a60000000p-3, -0x1.6b38640000000p-3, 0x1.5ecdc60000000p-3, 0x1.53381a0000000p-8, 0x1.a8b77c0000000p-6, 0x1.27b7b20000000p-3, 0x1.a94c1a0000000p-4, -0x1.3543120000000p-5, -0x1.a3f3da0000000p-3, -0x1.fc543a0000000p-6, -0x1.77fa5c0000000p-4, -0x1.319ee80000000p-8, 0x1.9b98ac0000000p-7, 0x1.04b9040000000p-5, -0x1.e1b9c60000000p-4, 0x1.69b9100000000p-3, -0x1.73623a0000000p-7, -0x1.2599f40000000p-3, 0x1.e674760000000p-4, -0x1.4b8c220000000p-3, 0x1.2479de0000000p-3, 0x1.5bea5c0000000p-5, 0x1.a6d2900000000p-3, 0x1.7852580000000p-3, -0x1.014c3a0000000p-5, 0x1.d4983e0000000p-8, -0x1.1dc3ec0000000p-4, -0x1.14e8a20000000p-3, -0x1.bbed7c0000000p-4, -0x1.112fae0000000p-4, -0x1.24e8520000000p-5, -0x1.0a13c60000000p-5, 0x1.4f40120000000p-3, 0x1.01cf060000000p-4, -0x1.3cfcb00000000p-3, 0x1.3a81740000000p-3, 0x1.d0cdc60000000p-5, -0x1.209ea20000000p-4, -0x1.960a3e0000000p-5, -0x1.51d9260000000p-7, -0x1.edb66a0000000p-4, 0x1.712e940000000p-3, -0x1.94a02e0000000p-5, -0x1.ba17a60000000p-5, -0x1.37df9c0000000p-3, 0x1.51dfdc0000000p-5, 0x1.12cdf80000000p-5, 0x1.d6c21a0000000p-6, -0x1.483a560000000p-7, 0x1.f7b2120000000p-4, 0x1.0964c00000000p-3, 0x1.62f87c0000000p-3, -0x1.fe15f40000000p-4, 0x1.7cd74e0000000p-3, 0x1.bd966a0000000p-3, -0x1.73a0ba0000000p-3, 0x1.c3dc9e0000000p-3, 0x1.88aaf60000000p-3, 0x1.15bd2c0000000p-3}
, {-0x1.4fb0440000000p-3, -0x1.cf23ee0000000p-4, -0x1.d9bf600000000p-10, 0x1.8057fc0000000p-4, -0x1.a737420000000p-3, -0x1.697db00000000p-5, 0x1.bd23920000000p-8, 0x1.4ac6b40000000p-3, 0x1.373c3e0000000p-3, -0x1.e244100000000p-4, -0x1.6ec3580000000p-3, -0x1.53e8ae0000000p-5, 0x1.e4447e0000000p-7, -0x1.f6a3ec0000000p-4, -0x1.d7e13c0000000p-7, 0x1.1dff340000000p-3, -0x1.0e96ec0000000p-4, 0x1.4ea8a00000000p-6, 0x1.2b559c0000000p-4, -0x1.dced840000000p-5, 0x1.f266ec0000000p-4, -0x1.97f0040000000p-4, 0x1.a0b56a0000000p-3, 0x1.d79ff20000000p-4, -0x1.3f601e0000000p-7, -0x1.9934b40000000p-3, 0x1.98efae0000000p-3, 0x1.c81a8e0000000p-4, -0x1.701f160000000p-4, 0x1.54018e0000000p-3, 0x1.5d1ece0000000p-3, -0x1.f0b2f80000000p-6, -0x1.7a02f20000000p-3, 0x1.b8c9d20000000p-4, 0x1.341be40000000p-4, 0x1.4594ae0000000p-5, -0x1.e343a60000000p-6, -0x1.9675e20000000p-5, -0x1.0dd8bc0000000p-3, 0x1.09f3e60000000p-9, 0x1.237f480000000p-3, 0x1.4237f80000000p-3, 0x1.b353b60000000p-5, 0x1.c624580000000p-3, 0x1.e1e4bc0000000p-4, -0x1.6ddf2a0000000p-4, 0x1.3b20ba0000000p-3, 0x1.fa932a0000000p-5, -0x1.b6ad7e0000000p-4, -0x1.387b880000000p-4, -0x1.66c39e0000000p-7, 0x1.91d7700000000p-4, -0x1.3eb90a0000000p-7, -0x1.c8c72a0000000p-3, 0x1.b194e20000000p-6, 0x1.31a6b20000000p-3, 0x1.43897e0000000p-3, 0x1.6f71da0000000p-6, 0x1.5378be0000000p-5, -0x1.ce65ba0000000p-5, 0x1.2dd7140000000p-6, -0x1.6bd3dc0000000p-4, -0x1.105f3c0000000p-3, -0x1.b7e7140000000p-5, -0x1.3951bc0000000p-4, -0x1.289efe0000000p-3, 0x1.04f7700000000p-3, -0x1.9600360000000p-3, -0x1.86469c0000000p-3, 0x1.a465e00000000p-10, -0x1.81e20c0000000p-4, 0x1.1106200000000p-3, -0x1.00b79c0000000p-4, -0x1.14a7540000000p-6, 0x1.cb60040000000p-5, 0x1.2c85320000000p-4, 0x1.3378540000000p-7, -0x1.4a7d6c0000000p-6, 0x1.9ad45c0000000p-4, 0x1.73b92e0000000p-4, -0x1.5152ba0000000p-4, -0x1.6410b20000000p-3, -0x1.2a5ba40000000p-3, -0x1.409c4c0000000p-3, -0x1.d21ea20000000p-9, 0x1.cc2a780000000p-6, 0x1.fa00160000000p-10, 0x1.3c56200000000p-6, -0x1.d0c74a0000000p-4, 0x1.1f56140000000p-3, 0x1.888a700000000p-3, -0x1.031a980000000p-3, -0x1.2d96d20000000p-5, -0x1.2120980000000p-3, -0x1.9c35ca0000000p-3, -0x1.e275de0000000p-4, -0x1.42998a0000000p-3, -0x1.20740c0000000p-7, 0x1.d3c4920000000p-5, -0x1.9de6860000000p-3, -0x1.85111c0000000p-3, -0x1.62b5100000000p-5, 0x1.43f26e0000000p-9, 0x1.ab7a520000000p-4, 0x1.668eba0000000p-3, -0x1.539da80000000p-4, 0x1.5b142c0000000p-7, -0x1.b7cf3a0000000p-4, 0x1.a1bfe40000000p-6, 0x1.ce99220000000p-6, -0x1.8355640000000p-4, 0x1.44fb220000000p-3, 0x1.27b8780000000p-3, 0x1.fb0b400000000p-6, 0x1.6c8f940000000p-5, 0x1.871c300000000p-4, -0x1.b009c80000000p-4, 0x1.e5d1120000000p-4, -0x1.2b84920000000p-3, -0x1.e77bae0000000p-5, -0x1.ad51c40000000p-3, -0x1.3cae1e0000000p-3, -0x1.7fb99a0000000p-3, 0x1.96fd4c0000000p-3, -0x1.6e259e0000000p-3, 0x1.b7cade0000000p-4, -0x1.9eddf80000000p-3, -0x1.3b50ee0000000p-3}
, {0x1.2506400000000p-3, -0x1.4926f00000000p-3, -0x1.a3a3880000000p-3, 0x1.3c20340000000p-3, 0x1.aa94f20000000p-3, 0x1.ffbb700000000p-5, -0x1.54283c0000000p-3, 0x1.bcebfa0000000p-4, 0x1.4535880000000p-3, -0x1.2ad1060000000p-3, 0x1.c876360000000p-4, 0x1.3f4dbc0000000p-4, -0x1.4cd5280000000p-3, 0x1.c96a980000000p-4, -0x1.5e598e0000000p-3, 0x1.880b360000000p-5, -0x1.57b11a0000000p-5, 0x1.75a7900000000p-4, -0x1.8d28a40000000p-6, 0x1.c370340000000p-3, -0x1.7a7d080000000p-3, 0x1.2681400000000p-4, -0x1.a85cf40000000p-3, 0x1.40836c0000000p-3, -0x1.3e43dc0000000p-3, 0x1.2f22d40000000p-3, -0x1.8184960000000p-4, 0x1.b5603a0000000p-5, 0x1.8346200000000p-3, -0x1.3dac7c0000000p-3, -0x1.444ae60000000p-7, 0x1.72a0160000000p-6, -0x1.5b1c460000000p-4, -0x1.1c802c0000000p-3, -0x1.3197d60000000p-3, 0x1.6c9e520000000p-3, 0x1.cc12e80000000p-6, -0x1.7249780000000p-4, 0x1.7689a60000000p-3, -0x1.8327e00000000p-3, 0x1.96d6a00000000p-3, 0x1.002ac20000000p-3, -0x1.7e56140000000p-3, -0x1.46d90a0000000p-3, -0x1.47539e0000000p-3, 0x1.1d13580000000p-3, -0x1.a215a60000000p-3, 0x1.2cb9880000000p-4, -0x1.7deef60000000p-3, -0x1.798a540000000p-8, -0x1.9f79520000000p-4, -0x1.ad3ff60000000p-4, -0x1.9dc3ca0000000p-3, -0x1.05a0a40000000p-5, 0x1.9bebac0000000p-3, 0x1.63f6560000000p-4, -0x1.2f7c060000000p-6, -0x1.cdb4aa0000000p-5, 0x1.8c068a0000000p-6, -0x1.be29d80000000p-3, -0x1.3cdd460000000p-3, 0x1.146a3c0000000p-3, -0x1.b6d1f40000000p-6, -0x1.20ea6a0000000p-4, 0x1.4b64220000000p-3, -0x1.a25b6c0000000p-4, -0x1.be53d00000000p-4, 0x1.a1623a0000000p-3, 0x1.7c775e0000000p-3, -0x1.4f8fa40000000p-3, 0x1.2c11620000000p-5, -0x1.0b86e40000000p-4, -0x1.c36a5a0000000p-4, 0x1.04efd80000000p-3, -0x1.3397980000000p-4, -0x1.0396d20000000p-3, 0x1.88feb80000000p-6, 0x1.d8329e0000000p-5, 0x1.fc7d160000000p-4, -0x1.45bd8a0000000p-4, 0x1.0924ec0000000p-3, 0x1.960f400000000p-3, -0x1.0b69160000000p-6, -0x1.e63ef00000000p-4, -0x1.1b154c0000000p-7, -0x1.4f8aa00000000p-3, 0x1.7aebb00000000p-3, -0x1.4dcfaa0000000p-3, 0x1.e3f02c0000000p-5, -0x1.ccfa440000000p-4, 0x1.9ec4420000000p-3, -0x1.7547520000000p-13, -0x1.0e80de0000000p-4, 0x1.78f0220000000p-4, -0x1.88b8fc0000000p-3, -0x1.ea268a0000000p-6, -0x1.a6caf00000000p-3, -0x1.cad8140000000p-3, 0x1.d7035a0000000p-7, 0x1.3e17660000000p-3, 0x1.5672f40000000p-3, 0x1.2c5e040000000p-4, -0x1.0d53e40000000p-3, 0x1.c34f380000000p-5, -0x1.3457880000000p-5, 0x1.64c2840000000p-3, 0x1.8209820000000p-4, 0x1.df6e420000000p-4, -0x1.7401d20000000p-3, -0x1.a851dc0000000p-3, 0x1.40a9180000000p-4, -0x1.3498520000000p-3, -0x1.6018b40000000p-3, -0x1.84a30a0000000p-6, 0x1.52885a0000000p-5, -0x1.3ea4780000000p-6, 0x1.4410880000000p-4, 0x1.becdbc0000000p-4, 0x1.50838e0000000p-6, -0x1.6aa8560000000p-4, 0x1.af76fe0000000p-5, -0x1.d1dc960000000p-4, -0x1.3449680000000p-4, -0x1.06fce60000000p-3, 0x1.6fc3940000000p-4, -0x1.ceb66c0000000p-7, -0x1.e9fe9c0000000p-5, 0x1.0179a20000000p-5}
, {0x1.7d4cfe0000000p-4, 0x1.37b5440000000p-4, 0x1.f9ddcc0000000p-4, 0x1.7707ee0000000p-3, 0x1.f5303e0000000p-4, 0x1.642cee0000000p-3, 0x1.ac24f80000000p-3, 0x1.88d63a0000000p-5, -0x1.ab72940000000p-3, 0x1.0c81a60000000p-9, -0x1.25768a0000000p-3, -0x1.923a460000000p-4, -0x1.40e0ae0000000p-3, 0x1.8623b00000000p-3, -0x1.02d9d60000000p-4, -0x1.219fe00000000p-3, -0x1.c1f6a40000000p-5, 0x1.3b01fa0000000p-5, 0x1.9a460c0000000p-5, 0x1.a1c6d00000000p-4, 0x1.2521800000000p-4, 0x1.0872720000000p-4, 0x1.e267400000000p-6, 0x1.41a7900000000p-4, -0x1.14635a0000000p-3, 0x1.3526cc0000000p-3, -0x1.ac7e0a0000000p-6, -0x1.67fd180000000p-3, -0x1.3ff4fe0000000p-7, 0x1.3d01220000000p-3, 0x1.c4c4e60000000p-4, 0x1.807e000000000p-3, 0x1.f86f6a0000000p-5, -0x1.68391e0000000p-4, -0x1.ae1a120000000p-5, -0x1.62dcdc0000000p-3, -0x1.2d34b20000000p-4, -0x1.78d4a80000000p-3, -0x1.6711620000000p-5, -0x1.9b4cf80000000p-5, 0x1.1fd3e40000000p-4, -0x1.7045260000000p-5, 0x1.3120280000000p-4, -0x1.96e44a0000000p-3, 0x1.c57c160000000p-4, -0x1.5585940000000p-3, -0x1.6fec3c0000000p-4, 0x1.ef89160000000p-4, -0x1.517ace0000000p-3, -0x1.e444d60000000p-4, -0x1.1c76f40000000p-8, -0x1.0c3c620000000p-3, -0x1.d366f80000000p-7, -0x1.376bd40000000p-4, 0x1.205dfe0000000p-3, -0x1.8603b60000000p-5, -0x1.2857940000000p-5, 0x1.0de2c80000000p-4, -0x1.0a2bba0000000p-3, -0x1.2cecc00000000p-5, 0x1.374e600000000p-3, 0x1.92372c0000000p-3, -0x1.75117c0000000p-4, 0x1.59d8ce0000000p-4, 0x1.e9f1560000000p-4, 0x1.b0f7780000000p-6, 0x1.5ea9960000000p-4, 0x1.60cba60000000p-4, -0x1.9d0de00000000p-5, -0x1.538aa60000000p-3, 0x1.06034a0000000p-3, -0x1.20a9cc0000000p-3, 0x1.9020200000000p-3, 0x1.8a1a980000000p-4, -0x1.03cc160000000p-3, 0x1.59367a0000000p-5, 0x1.4bc7da0000000p-3, -0x1.eebffc0000000p-4, 0x1.92ff400000000p-3, -0x1.f928200000000p-4, -0x1.c6700a0000000p-10, -0x1.b2805a0000000p-3, -0x1.b9cca40000000p-7, -0x1.845dba0000000p-3, -0x1.11adec0000000p-8, -0x1.f949400000000p-4, -0x1.cf5d0c0000000p-6, -0x1.2be3520000000p-4, 0x1.5b94100000000p-3, 0x1.4312cc0000000p-3, 0x1.0f7f500000000p-4, 0x1.051f8c0000000p-6, -0x1.1ca1c80000000p-3, -0x1.ae52400000000p-7, 0x1.c7463a0000000p-5, 0x1.8e8ed60000000p-3, -0x1.5c047a0000000p-4, -0x1.e38c3e0000000p-4, -0x1.6d84f20000000p-3, 0x1.e92bfc0000000p-4, 0x1.0d01e60000000p-3, -0x1.1e2a5e0000000p-4, -0x1.9398c20000000p-3, 0x1.7191de0000000p-3, -0x1.e0b0da0000000p-5, 0x1.2ae7fa0000000p-5, 0x1.2b94aa0000000p-5, 0x1.6746960000000p-3, -0x1.2d18960000000p-4, 0x1.99ca400000000p-8, -0x1.57145c0000000p-5, 0x1.41c3b40000000p-5, 0x1.7f644a0000000p-4, -0x1.88d4740000000p-4, -0x1.c8ff760000000p-4, -0x1.632c9c0000000p-3, -0x1.4e0ba20000000p-3, 0x1.8fce9c0000000p-3, 0x1.061d9a0000000p-3, 0x1.2c539e0000000p-3, -0x1.f8fe5a0000000p-5, -0x1.2d1c5c0000000p-5, -0x1.7c74680000000p-3, 0x1.89a8d40000000p-5, 0x1.9bd10c0000000p-6, 0x1.57243e0000000p-3, 0x1.4209ec0000000p-3, -0x1.e09e240000000p-6}
}
;

#undef INPUT_SAMPLES
#undef FC_UNITS