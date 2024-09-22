//#include <LibLacuna.h>
#include <I2S.h>
#include <SPI.h>
#include <LibLacuna.h>
// #include <rfthings_sx126x.h>

#include "gsc_model_fixed.h"

#ifndef REGION
#define REGION R_EU868
#endif

#define I2S_SAMPLE_RATE 16000  // [16000, 48000] supported by the microphone
#define I2S_BITS_PER_SAMPLE 32 // Data is sent in 32-bit packets over I2S but only 18 bits are used by the microphone, remaining least significant bits are set to 0

static input_t inputs; // 1-channel, 16000 samples for 16kHz over 1s
static volatile size_t sample_i = 0;
static output_t outputs;
static volatile boolean ready_for_inference = false;

static byte networkKey[] = {0xFF, 0x68, 0x20, 0xE4, 0xDA, 0xEF, 0x35, 0x58, 0x71, 0x8D, 0xD6, 0xB2, 0x5A, 0xE2, 0xE1, 0xB6};
static byte appKey[] = {0xD7, 0x36, 0xE9, 0xF7, 0xF6, 0x41, 0x43, 0x42, 0x28, 0x97, 0xAD, 0xBD, 0xB0, 0x96, 0xD8, 0xE6};
static byte deviceAddress[] = {0x26, 0x0B, 0x4F, 0x03};

static lsLoraWANParams loraWANParams;
static lsLoraTxParams txParams;

void processI2SData(uint8_t *data, size_t size) {
    int32_t *data32 = (int32_t*)data;

    // Send signed 32-bit PCM little endian 2 channels
    //Serial.write(data, size);

    // Copy first channel into model inputs
    size_t i = 0;
    for (i = 0; i < size / 8 && sample_i + i < MODEL_INPUT_DIM_0; i++, sample_i++) {
      inputs[sample_i][0] = data32[i * 2] >> 14; // Drop 32 - 18 = 14 unused bits
    }

    if (sample_i >= MODEL_INPUT_DIM_0) {
      ready_for_inference = true;
    }
}

void onI2SReceive() {
  size_t size = I2S.available();
  static uint8_t data[I2S_BUFFER_SIZE];

  if (size > 0) {
    I2S.read(data, size);
    processI2SData(data, size);
  }
}

void setup() {
  Serial.begin(115200);
  
  // For RFThing-DKAIoT
  pinMode(PIN_LED, OUTPUT);
  pinMode(LS_GPS_ENABLE, OUTPUT);
  digitalWrite(LS_GPS_ENABLE, LOW);
  pinMode(LS_GPS_V_BCKP, OUTPUT);
  digitalWrite(LS_GPS_V_BCKP, LOW);
  pinMode(SD_ON_OFF, OUTPUT);
  digitalWrite(SD_ON_OFF, HIGH);

  delay(100); // Wait for peripheral power rail to stabilize after setting SD_ON_OFF

  // start I2S
  if (!I2S.begin(I2S_PHILIPS_MODE, I2S_SAMPLE_RATE, I2S_BITS_PER_SAMPLE, false)) {
    Serial.println("Failed to initialize I2S!");
    while (1); // do nothing
  }

  I2S.onReceive(onI2SReceive);

  // Trigger a read to start DMA
  I2S.peek();

  //Serial.println("Initializing DONE");

  // LoRa
  // SX1262 configuration for lacuna LS200 board
  lsSX126xConfig cfg;
  lsCreateDefaultSX126xConfig(&cfg, BOARD_VERSION);

  // Special configuration for DKAIoT Board
  cfg.nssPin = E22_NSS;                           //19
  cfg.resetPin = E22_NRST;                        //14
  cfg.antennaSwitchPin = E22_RXEN;                //1
  cfg.busyPin = E22_BUSY;                         //2
  cfg.dio1Pin = E22_DIO1;                         //39

  // Initialize SX1262
  int result = lsInitSX126x(&cfg, REGION);

  // LoRaWAN session parameters
  lsCreateDefaultLoraWANParams(&loraWANParams, networkKey, appKey, deviceAddress);
  loraWANParams.txPort = 1;
  loraWANParams.rxEnable = true;
 
  // transmission parameters for terrestrial LoRa
  lsCreateDefaultLoraTxParams(&txParams, REGION);
  txParams.spreadingFactor = lsLoraSpreadingFactor_7;
  txParams.frequency = 868100000;
  //txParams.power = 1;
}

void loop() {
  if (ready_for_inference) {
    // Input buffer full, perform inference

    // Turn LED on during preprocessing/prediction
    digitalWrite(PIN_LED, HIGH);

    // Start timer
    long long t_start = millis();


    // Compute DC offset
    int32_t dc_offset = 0;

    for (size_t i = 0; i < sample_i; i++) { // Accumulate samples
      dc_offset += inputs[i][0];
    }

    dc_offset = dc_offset / (int32_t)sample_i; // Compute average over samples

    // Filtering
    for (size_t i = 0; i < sample_i; i++) {
      // Remove DC offset
      inputs[i][0] -= dc_offset;

      // Amplify
      inputs[i][0] = inputs[i][0] << 2;
    }

    // Send signed 16-bit PCM little endian 1 channel
    //Serial.write((uint8_t*)inputs[0], MODEL_INPUT_DIM_0 * 2);

    // Normalize
    // int32_t max_sample = inputs[0][0];
    // for (size_t i = 1; i < MODEL_INPUT_DIM_0; i++) {
    //   if (max_sample < inputs[0][i])
    //     max_sample = inputs[0][i];
    // }
    // for (size_t i = 0; i < MODEL_INPUT_DIM_0; i++) {
    //   inputs[0][i] = ((int32_t)inputs[0][i] << FIXED_POINT) / max_sample;
    // }

    // Predict
    cnn(inputs, outputs);

    // Get output class
    unsigned int label = 0;
    float max_val = outputs[0];
    for (unsigned int i = 1; i < MODEL_OUTPUT_SAMPLES; i++) {
      if (max_val < outputs[i]) {
        max_val = outputs[i];
        label = i;
      }
    }
    
    static char msg[32];
    snprintf(msg, sizeof(msg), "%d,%f,%d", label, (double)max_val, (int)(millis() - t_start));
    Serial.println(msg);

    // static unsigned char data[sizeof(msg)];

    // for (size_t i = 0; i < sizeof(msg); i++) {
    //     data[i] = (unsigned char)msg[i];
    // }
    // int lora_result  = lsSendLoraWAN(&loraWANParams, &txParams, data, sizeof(data));

    // Turn LED off after prediction has been sent
    digitalWrite(PIN_LED, LOW);
    
    ready_for_inference = false;
    sample_i = 0;
  }
}
