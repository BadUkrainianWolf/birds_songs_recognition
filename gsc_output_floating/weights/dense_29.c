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


const float dense_29_bias[FC_UNITS] = {0x1.33bc5c0000000p-5, -0x1.3a1cd60000000p-3, -0x1.dec5ea0000000p-4, -0x1.e479b00000000p-3, 0x1.59f1000000000p-3, 0x1.db7ab40000000p-3, 0x1.2d4b020000000p-4, 0x1.f5312e0000000p-2, -0x1.a79b0a0000000p-2, -0x1.35fdde0000000p-4}
;

const float dense_29_kernel[FC_UNITS][INPUT_SAMPLES] = {{0x1.9f7df00000000p-2, 0x1.b8c4080000000p-7, -0x1.ca505e0000000p-3, -0x1.4281340000000p-1, -0x1.8b09520000000p-2, 0x1.5352ce0000000p-3, 0x1.a8a30c0000000p-3, 0x1.3241340000000p-3, 0x1.a9bb220000000p-4, 0x1.0cb1de0000000p-2, -0x1.9351400000000p-3, 0x1.a2173c0000000p-2, -0x1.1c750e0000000p-2, 0x1.d9d1fa0000000p-3, 0x1.0ac8e20000000p-4, 0x1.45a2ac0000000p-4, -0x1.8817b40000000p-8, 0x1.b055320000000p-3, -0x1.9910b60000000p-4, 0x1.d3cbb60000000p-6, 0x1.473ffe0000000p-4, -0x1.9b77160000000p-5, 0x1.6cc11c0000000p-3, 0x1.1503900000000p-3, 0x1.d18e620000000p-3, -0x1.ce15ee0000000p-5, -0x1.1c2fe00000000p-1, -0x1.19513e0000000p-2, 0x1.a57ed00000000p-1, -0x1.35cf200000000p-3, -0x1.b8a8340000000p-4, -0x1.cb1dc80000000p-3, -0x1.196b660000000p-3, -0x1.aa3ab40000000p-3, -0x1.dbacce0000000p-3, 0x1.c413900000000p-7, -0x1.4f02980000000p-5, -0x1.0acdc60000000p-9, -0x1.5a90060000000p-2, 0x1.9c6e1c0000000p-7, -0x1.e440400000000p-2, -0x1.1e1e4e0000000p-3, 0x1.3fe8b00000000p-3, -0x1.21763a0000000p-1, 0x1.82fcd20000000p-2, 0x1.d02c9e0000000p-4, -0x1.5f6af80000000p-2, -0x1.4cd1c40000000p-4, -0x1.5dcb4e0000000p-2, -0x1.24331e0000000p+0, 0x1.fdc05e0000000p-5, -0x1.2eac760000000p-2, 0x1.0492620000000p-5, 0x1.f1a0720000000p-2, -0x1.26e0700000000p-3, -0x1.861a040000000p-2, 0x1.c631d00000000p-3, 0x1.b9b62e0000000p-2, -0x1.748ba80000000p-2, 0x1.af62020000000p-3, -0x1.77773a0000000p-2, -0x1.1af1400000000p-3, -0x1.ed116e0000000p-3, 0x1.4886dc0000000p-1}
, {0x1.d664e60000000p-4, -0x1.439b0e0000000p-3, 0x1.a94c800000000p-3, 0x1.e8598c0000000p-3, 0x1.6c2dca0000000p-4, -0x1.e6eaca0000000p-3, -0x1.0b0f100000000p-5, -0x1.9de3ce0000000p-4, -0x1.ba84120000000p-4, 0x1.78daaa0000000p-2, 0x1.4c1f880000000p-4, 0x1.e57c180000000p-6, 0x1.8461a40000000p-3, 0x1.2b4da00000000p-9, 0x1.a652ce0000000p-4, -0x1.1d6dce0000000p-5, 0x1.73bb600000000p-3, -0x1.8558ba0000000p-4, 0x1.5056020000000p-4, -0x1.1793260000000p-3, 0x1.a9241c0000000p-5, 0x1.45883c0000000p-2, 0x1.229fc40000000p-3, -0x1.655a1c0000000p-6, 0x1.5077fa0000000p-3, -0x1.a274b60000000p-4, -0x1.96b8460000000p-4, -0x1.9067540000000p-3, -0x1.00308a0000000p-5, -0x1.5007780000000p-5, 0x1.54cf880000000p-3, -0x1.b1560e0000000p-5, -0x1.c0db5a0000000p-5, -0x1.c9f0e80000000p-4, -0x1.633a1c0000000p-3, -0x1.48f2ac0000000p-1, -0x1.6e930e0000000p-4, 0x1.7894880000000p-4, 0x1.d5d6ac0000000p-4, -0x1.8852cc0000000p-3, -0x1.36b2f60000000p-2, 0x1.5d13500000000p-3, -0x1.78fed20000000p-7, -0x1.20c5000000000p-4, 0x1.e753e60000000p-3, 0x1.ff394a0000000p-3, -0x1.1eebd00000000p-1, -0x1.1ff3800000000p-3, 0x1.eb8da80000000p-2, -0x1.76a54a0000000p-3, -0x1.b4cf100000000p-2, -0x1.8d9b760000000p-3, 0x1.d5a77a0000000p-4, 0x1.11dd780000000p-4, 0x1.3706b80000000p-5, -0x1.5218d60000000p-3, 0x1.c311ec0000000p-3, -0x1.b3b6500000000p-1, 0x1.602b4c0000000p-10, -0x1.db19ce0000000p-3, 0x1.03ac800000000p-3, -0x1.0545640000000p-3, 0x1.3270a20000000p-3, -0x1.44731a0000000p-2}
, {-0x1.c1325c0000000p-4, -0x1.2279a40000000p-3, 0x1.39d6f40000000p-2, -0x1.8e17060000000p-7, 0x1.5d0d760000000p-3, 0x1.9f35cc0000000p-12, 0x1.d1d1b00000000p-3, -0x1.7b52380000000p-3, -0x1.fe76920000000p-3, 0x1.d8c4440000000p-5, 0x1.16b1880000000p-3, -0x1.88aa380000000p-3, -0x1.8e44fa0000000p-2, -0x1.ef86400000000p-4, -0x1.24cf8c0000000p-4, -0x1.77c5900000000p-3, -0x1.80fb740000000p-3, -0x1.2740f60000000p-2, 0x1.263b980000000p-4, 0x1.49528a0000000p-4, -0x1.e195160000000p-2, -0x1.c540800000000p-5, -0x1.d938da0000000p-2, 0x1.c8a7f00000000p-3, -0x1.91f41e0000000p-4, -0x1.d7c7440000000p-2, 0x1.12d0500000000p-2, -0x1.5d856a0000000p-2, -0x1.9041260000000p-1, -0x1.4c5d400000000p-3, 0x1.2aa9a20000000p-3, -0x1.3ad6120000000p-3, -0x1.42ac920000000p-2, 0x1.3681880000000p-2, -0x1.1a12d20000000p-2, 0x1.3f053c0000000p-3, 0x1.0c97be0000000p-4, 0x1.be73200000000p-6, 0x1.0cb6e80000000p-5, 0x1.c6971c0000000p-3, -0x1.b0a5620000000p-1, -0x1.631fa00000000p-2, 0x1.28f7c00000000p-5, -0x1.5eb18c0000000p-3, -0x1.74f0620000000p-1, -0x1.b00c980000000p-2, 0x1.12e4ee0000000p-7, 0x1.97a8500000000p-4, 0x1.9a2e080000000p-4, 0x1.c249f00000000p-3, 0x1.a0cab60000000p-4, -0x1.5c9c8e0000000p-2, 0x1.99947e0000000p-7, 0x1.fdf0a20000000p-3, -0x1.aa90b80000000p-5, 0x1.3e26aa0000000p-2, -0x1.d7f8d60000000p-3, -0x1.21a6620000000p-1, -0x1.82ae6c0000000p-6, 0x1.0ca8ea0000000p-5, 0x1.be3ff00000000p-4, -0x1.3f9de60000000p-4, 0x1.fbd0a80000000p-3, 0x1.5058a80000000p-2}
, {0x1.aa6e860000000p-3, 0x1.4ca1100000000p-3, 0x1.f865c60000000p-3, 0x1.616efc0000000p-2, 0x1.862e8c0000000p-2, -0x1.a43eac0000000p-3, -0x1.32a1740000000p-2, 0x1.afc3440000000p-6, 0x1.164ef80000000p-8, -0x1.cf0c660000000p-2, 0x1.79ee5a0000000p-3, -0x1.2d753e0000000p-4, -0x1.06973e0000000p-1, 0x1.ae712a0000000p-3, -0x1.03b9740000000p-8, -0x1.88d8e80000000p-4, -0x1.9d401c0000000p-3, 0x1.1edf1e0000000p-5, -0x1.3187360000000p-2, -0x1.96d6820000000p-2, 0x1.145b000000000p-3, 0x1.e8989c0000000p-7, -0x1.b8a7980000000p-6, -0x1.2d1e520000000p-2, 0x1.2c371a0000000p-4, -0x1.838af60000000p-3, -0x1.1a79ca0000000p-2, 0x1.c29a240000000p-8, -0x1.62f7020000000p-2, 0x1.4476000000000p-3, 0x1.15e5920000000p-5, -0x1.4433580000000p-3, 0x1.088d220000000p-1, 0x1.013a420000000p-4, 0x1.a6dc3e0000000p-3, 0x1.0871e20000000p-4, 0x1.2358280000000p-1, 0x1.89f2780000000p-4, -0x1.22917c0000000p-5, 0x1.1fa85c0000000p-5, -0x1.5fae7a0000000p-2, 0x1.62d9180000000p-3, -0x1.f7f4a80000000p-5, -0x1.43409a0000000p-3, 0x1.af86ec0000000p-3, 0x1.f2b7180000000p-5, -0x1.f080700000000p-4, -0x1.fd8aa60000000p-2, 0x1.2aadc20000000p-6, -0x1.7964f20000000p-2, -0x1.0811060000000p-3, -0x1.e19b600000000p-6, -0x1.9678e40000000p-2, -0x1.4c46880000000p-2, -0x1.5a527e0000000p-4, 0x1.937dca0000000p-5, -0x1.497c660000000p-3, -0x1.9ba07e0000000p-1, 0x1.16dc880000000p-6, 0x1.5a474a0000000p-2, -0x1.5d720e0000000p-3, 0x1.ef463e0000000p-5, -0x1.61d58c0000000p-2, 0x1.1ece180000000p-2}
, {-0x1.ac59660000000p-4, -0x1.0f0f340000000p-1, -0x1.7839620000000p-3, -0x1.b0003e0000000p-4, 0x1.cf0c460000000p-3, 0x1.edd5d20000000p-4, -0x1.a59d120000000p-2, -0x1.13d7620000000p-3, 0x1.1a2ec20000000p-2, -0x1.a566f80000000p-4, -0x1.5b55d40000000p-1, 0x1.6f14620000000p-5, 0x1.88cb100000000p-4, 0x1.3b6bdc0000000p-2, 0x1.402b5e0000000p-3, -0x1.5d99b40000000p-2, 0x1.d291080000000p-3, -0x1.1f29bc0000000p-2, -0x1.667c520000000p-2, 0x1.07425a0000000p-4, -0x1.d8adf00000000p-2, 0x1.bd93ae0000000p-3, -0x1.3818ac0000000p-4, -0x1.49fe2e0000000p-1, -0x1.6e57b00000000p-4, -0x1.9a026c0000000p-3, 0x1.1d81820000000p-4, 0x1.2dd46e0000000p-2, 0x1.3adcc60000000p-3, 0x1.7e81480000000p-3, -0x1.8a6cac0000000p-2, 0x1.79a0540000000p-4, -0x1.348ac20000000p-3, -0x1.9122620000000p-2, 0x1.2ec9b80000000p-5, -0x1.73af6e0000000p-4, 0x1.0073a80000000p-2, -0x1.f3e4660000000p-4, -0x1.1e27720000000p-3, 0x1.bbe7ba0000000p-4, -0x1.495cf80000000p-4, 0x1.20fe580000000p-2, -0x1.0a15d20000000p-2, -0x1.12aa200000000p-1, 0x1.1ee4900000000p-4, -0x1.f5352e0000000p-4, -0x1.2601a20000000p-1, -0x1.1f25280000000p-2, -0x1.e88a9c0000000p-4, 0x1.77cd6a0000000p-4, -0x1.f6acaa0000000p-2, -0x1.dbf3a20000000p-7, -0x1.2cf8600000000p-2, 0x1.f09dde0000000p-5, -0x1.e56c580000000p-3, -0x1.23dfee0000000p-3, 0x1.95c2680000000p-3, 0x1.b89f340000000p-3, -0x1.ae51680000000p-2, 0x1.35be700000000p-3, 0x1.228e4a0000000p-4, 0x1.525c140000000p-4, 0x1.4d15d00000000p-3, 0x1.b42b340000000p-4}
, {-0x1.0daf000000000p-3, 0x1.5980a40000000p-2, 0x1.b642580000000p-7, -0x1.096c9e0000000p-5, -0x1.97359a0000000p-3, -0x1.e64c680000000p-3, 0x1.1bda920000000p-3, -0x1.385a340000000p-3, -0x1.398dcc0000000p-3, -0x1.5026b20000000p-3, 0x1.5312040000000p-3, -0x1.984c2e0000000p-4, 0x1.8eaeb00000000p-5, 0x1.7b27de0000000p-4, -0x1.48300a0000000p-1, -0x1.b590580000000p-2, -0x1.4c4b860000000p-6, 0x1.0793580000000p-3, 0x1.37b64c0000000p-3, 0x1.4ac4bc0000000p-3, 0x1.e71c820000000p-3, -0x1.85d0460000000p-3, 0x1.9314820000000p-4, -0x1.673d320000000p-2, 0x1.d5aa3a0000000p-4, 0x1.ea139c0000000p-2, 0x1.c5541e0000000p-4, 0x1.28cbe60000000p-3, -0x1.1142020000000p-1, 0x1.ec32040000000p-3, 0x1.29f7b00000000p-4, 0x1.b55a4e0000000p-3, 0x1.bb32e00000000p-3, -0x1.0579cc0000000p-2, 0x1.8e5cb60000000p-3, 0x1.accb8a0000000p-5, -0x1.475be80000000p-3, -0x1.2d1e160000000p-1, 0x1.8ba15a0000000p-7, 0x1.9d65f80000000p-3, -0x1.770b880000000p-1, 0x1.aa6cb20000000p-4, -0x1.6bfea80000000p-6, 0x1.29848a0000000p-4, -0x1.8d0a4e0000000p-5, -0x1.d555e80000000p-2, -0x1.18e3f80000000p-7, -0x1.9a189a0000000p-2, -0x1.552d760000000p-3, -0x1.e8ccde0000000p-3, 0x1.91c1220000000p-4, -0x1.383e860000000p-4, -0x1.7caf4c0000000p-1, -0x1.27fa340000000p-7, -0x1.32bd4c0000000p-3, -0x1.24b4220000000p-2, -0x1.27607e0000000p-4, -0x1.855f900000000p-3, 0x1.1059820000000p-3, -0x1.84429a0000000p-2, 0x1.0783aa0000000p-2, -0x1.2749d80000000p-3, 0x1.5c6f180000000p-4, -0x1.37cf5e0000000p-2}
, {0x1.514dd00000000p-4, -0x1.15dc0e0000000p-1, -0x1.b8adba0000000p-3, 0x1.9193a20000000p-3, -0x1.15e4020000000p-3, -0x1.961a3a0000000p-6, 0x1.56b6080000000p-7, 0x1.89602c0000000p-3, -0x1.34907c0000000p-2, -0x1.a181120000000p-5, 0x1.3e28840000000p-2, 0x1.89c08e0000000p-4, 0x1.9a243e0000000p-3, -0x1.7254360000000p-1, -0x1.29705a0000000p-2, -0x1.1cf0c80000000p-3, 0x1.0179160000000p-4, 0x1.8c78b40000000p-4, -0x1.b9666a0000000p-3, -0x1.5d91e80000000p-3, 0x1.f0341a0000000p-3, 0x1.b193c40000000p-4, -0x1.9bd05c0000000p-3, -0x1.68c7160000000p-3, 0x1.6658d80000000p-4, -0x1.7d6b6c0000000p-2, 0x1.4bf1840000000p-6, 0x1.10f2c00000000p-2, -0x1.c86b360000000p-2, 0x1.92df8c0000000p-5, 0x1.158e7e0000000p-2, 0x1.c0d2d80000000p-4, 0x1.2d0e060000000p-3, -0x1.16dc360000000p-2, -0x1.9542800000000p-2, -0x1.019e440000000p-4, 0x1.26e5480000000p-2, -0x1.4d27360000000p-4, 0x1.86ff260000000p-8, -0x1.9112f60000000p-4, 0x1.db899e0000000p-3, -0x1.3d6f480000000p-2, -0x1.94be660000000p-4, 0x1.3cf63e0000000p-2, -0x1.4770de0000000p-1, -0x1.a5415c0000000p-2, -0x1.6dc4ee0000000p-3, -0x1.2456640000000p-4, -0x1.60b1800000000p-2, 0x1.8777ba0000000p-3, 0x1.017a440000000p-5, 0x1.516ad60000000p-1, 0x1.21cb880000000p-6, 0x1.374f360000000p-2, 0x1.b6fc700000000p-3, -0x1.657dfc0000000p-1, 0x1.b1db8e0000000p-5, -0x1.f7c3ac0000000p-2, 0x1.0393600000000p-1, 0x1.eb8a140000000p-3, -0x1.b1c39a0000000p-2, 0x1.2f74f60000000p-1, 0x1.3908fe0000000p-6, -0x1.ce4c420000000p-2}
, {-0x1.e532640000000p-2, -0x1.0dccac0000000p-2, -0x1.7cf2500000000p-1, -0x1.135db80000000p-2, -0x1.290eca0000000p-1, -0x1.cb514a0000000p-3, 0x1.ee5b620000000p-2, 0x1.d4957c0000000p-4, -0x1.4b944e0000000p-3, 0x1.e4dda20000000p-5, 0x1.38b4c20000000p-3, 0x1.240f860000000p-5, -0x1.66d26a0000000p-1, -0x1.755aa40000000p-1, 0x1.d3493e0000000p-3, 0x1.8fefb20000000p-7, -0x1.320fa60000000p-1, -0x1.0b93040000000p-4, 0x1.edad9e0000000p-6, 0x1.5aad180000000p-6, -0x1.908ffe0000000p-5, 0x1.bcb0640000000p-4, -0x1.4fcd2e0000000p-1, -0x1.df418a0000000p-2, 0x1.a658e40000000p-4, 0x1.a040e60000000p-4, -0x1.2b2bf20000000p-2, -0x1.2487060000000p-3, -0x1.7d78f20000000p-2, -0x1.340b080000000p-11, -0x1.09821a0000000p-4, -0x1.d7a97a0000000p-4, -0x1.358ae00000000p-1, 0x1.3a3aa60000000p-3, 0x1.df09660000000p-4, 0x1.9161bc0000000p-1, -0x1.cd5d820000000p-4, -0x1.5d78fc0000000p-7, 0x1.7951740000000p-3, 0x1.156b240000000p-4, -0x1.d436560000000p-2, 0x1.1686a80000000p-5, 0x1.d495f00000000p-5, 0x1.90075a0000000p-5, -0x1.e09a4a0000000p-2, -0x1.4a89560000000p-2, -0x1.50bd660000000p-1, -0x1.37c7ce0000000p-3, 0x1.3e72080000000p-3, 0x1.e5142e0000000p-5, 0x1.cc8f040000000p-4, -0x1.262f460000000p-1, -0x1.61b1c00000000p-3, -0x1.74d91c0000000p-3, -0x1.8d40dc0000000p-3, 0x1.8bdbda0000000p-6, 0x1.0222aa0000000p-1, -0x1.3ada1a0000000p-1, -0x1.c83f720000000p-3, 0x1.3f47da0000000p-5, -0x1.02d6080000000p-2, 0x1.622b460000000p-3, -0x1.11fc060000000p-5, 0x1.69666c0000000p-5}
, {0x1.3f3d600000000p-2, 0x1.414fca0000000p-3, 0x1.d6e9860000000p-2, -0x1.5d7a760000000p-3, -0x1.f24ab80000000p-2, -0x1.83f2660000000p-4, 0x1.c7fc8c0000000p-2, -0x1.68ea260000000p-5, 0x1.aa4c560000000p-7, 0x1.0176b00000000p-4, -0x1.b3cf640000000p-2, -0x1.0aa0720000000p-1, 0x1.b2e17e0000000p-3, 0x1.2ee1080000000p-3, -0x1.b6bcf20000000p-2, 0x1.bc50680000000p-4, -0x1.4d6a7a0000000p-1, -0x1.998bae0000000p-3, -0x1.15ac2c0000000p-3, -0x1.d4af660000000p-3, 0x1.a08cfe0000000p-2, -0x1.92636e0000000p-4, 0x1.9faf540000000p-5, 0x1.9c7b6a0000000p-2, -0x1.2f7aba0000000p+0, -0x1.5f73320000000p-4, 0x1.4d17a60000000p-5, 0x1.86a61a0000000p-4, -0x1.fd28380000000p-2, -0x1.a129120000000p-2, 0x1.009e800000000p-2, 0x1.848dcc0000000p-2, 0x1.e14eec0000000p-6, -0x1.2ad9b20000000p-1, -0x1.75a1600000000p-1, -0x1.1a5dbc0000000p-1, -0x1.fcae020000000p-1, 0x1.dabca80000000p-7, 0x1.c450120000000p-5, -0x1.92ba7a0000000p-2, -0x1.f894100000000p-2, -0x1.3092380000000p-5, -0x1.5750900000000p-3, -0x1.a684ea0000000p-5, -0x1.2cec0c0000000p-1, -0x1.f20dde0000000p-4, -0x1.6e49aa0000000p-2, -0x1.df749a0000000p-4, 0x1.13447c0000000p-2, -0x1.4309300000000p-1, -0x1.379d1e0000000p-3, 0x1.56dcde0000000p-2, 0x1.03f45c0000000p-2, -0x1.b2faa40000000p-2, -0x1.d0afd80000000p-4, -0x1.6e77860000000p-3, -0x1.578af40000000p-5, 0x1.1d89d60000000p-1, -0x1.985c7a0000000p-2, 0x1.831c140000000p-3, -0x1.84eae00000000p-3, -0x1.08cf1c0000000p-1, -0x1.7b55de0000000p-1, -0x1.cd40c40000000p-1}
, {-0x1.e557d40000000p-6, -0x1.f944320000000p-8, 0x1.a7c1a20000000p-3, 0x1.87df2e0000000p-8, -0x1.30628e0000000p-3, 0x1.77a0680000000p-3, -0x1.6d54ce0000000p-2, -0x1.5116c40000000p-4, -0x1.ba43360000000p-4, 0x1.4d5f9e0000000p-5, 0x1.e9b20e0000000p-3, 0x1.fb79ee0000000p-2, 0x1.a8bedc0000000p-2, 0x1.b24cb60000000p-3, 0x1.1dd50a0000000p-3, -0x1.a9f0780000000p-3, -0x1.7be7c60000000p-4, 0x1.ef35980000000p-6, 0x1.5b4c280000000p-4, -0x1.3cad640000000p-2, -0x1.dd18ee0000000p-3, -0x1.8199020000000p-5, 0x1.9923800000000p-3, -0x1.a274860000000p-5, -0x1.5b53a20000000p-3, 0x1.dd5a020000000p-4, -0x1.247f300000000p-4, 0x1.0735460000000p-2, 0x1.ff573c0000000p-4, 0x1.062b600000000p-2, -0x1.eebdbc0000000p-3, -0x1.158d0e0000000p-1, -0x1.1655240000000p-4, 0x1.b0a5a60000000p-4, -0x1.d7896c0000000p-6, 0x1.692c660000000p-4, -0x1.cfb8dc0000000p-4, 0x1.d808420000000p-2, -0x1.1197ee0000000p-2, -0x1.91db240000000p-2, -0x1.4897500000000p-5, -0x1.2136120000000p-1, -0x1.0b13a80000000p-3, 0x1.35ae0a0000000p-5, -0x1.903ede0000000p-4, 0x1.200d140000000p-2, -0x1.c159920000000p-3, -0x1.01005e0000000p-4, -0x1.73d2540000000p-2, 0x1.4920500000000p-2, -0x1.3e02560000000p-3, -0x1.bce4ce0000000p-3, -0x1.a897780000000p-2, -0x1.905e2e0000000p-3, 0x1.db3d3e0000000p-2, -0x1.f050ee0000000p-4, 0x1.9140b60000000p-5, -0x1.c18fee0000000p-1, 0x1.b39a980000000p-4, -0x1.2168700000000p-2, -0x1.c49c5e0000000p-2, 0x1.a0294a0000000p-4, -0x1.4ec8c20000000p-2, 0x1.65cc3c0000000p-3}
}
;

#undef INPUT_SAMPLES
#undef FC_UNITS