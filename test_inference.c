#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern void dense_relu   (float* in, int in_size, float* w, float* b, float* out, int out_size);
extern void dense_softmax(float* in, int in_size, float* w, float* b, float* out, int out_size);
extern int  argmax       (float* arr, int size);

#define L0_IN  784
#define L0_OUT 256
#define L1_IN  256
#define L1_OUT 128
#define L2_IN  128
#define L2_OUT 64
#define L3_IN  64
#define L3_OUT 10

#define OFF_W0 0
#define OFF_B0 (OFF_W0 + L0_IN * L0_OUT * sizeof(float))
#define OFF_W1 (OFF_B0 + L0_OUT         * sizeof(float))
#define OFF_B1 (OFF_W1 + L1_IN * L1_OUT * sizeof(float))
#define OFF_W2 (OFF_B1 + L1_OUT         * sizeof(float))
#define OFF_B2 (OFF_W2 + L2_IN * L2_OUT * sizeof(float))
#define OFF_W3 (OFF_B2 + L2_OUT         * sizeof(float))
#define OFF_B3 (OFF_W3 + L3_IN * L3_OUT * sizeof(float))
#define MODEL_SIZE_BYTES (OFF_B3 + L3_OUT * sizeof(float))

int main(void) {
  
    FILE* f = fopen("data/model.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open data/model.bin\n"); return 1; }

    float* model = malloc(MODEL_SIZE_BYTES);
    if (fread(model, 1, MODEL_SIZE_BYTES, f) != MODEL_SIZE_BYTES) {
        fprintf(stderr, "Short read: data/model.bin\n"); return 1;
    }
    fclose(f);


    float* w0 = (float*)((char*)model + OFF_W0);
    float* b0 = (float*)((char*)model + OFF_B0);
    float* w1 = (float*)((char*)model + OFF_W1);
    float* b1 = (float*)((char*)model + OFF_B1);
    float* w2 = (float*)((char*)model + OFF_W2);
    float* b2 = (float*)((char*)model + OFF_B2);
    float* w3 = (float*)((char*)model + OFF_W3);
    float* b3 = (float*)((char*)model + OFF_B3);

 
    FILE* df = fopen("data/test_digit.bin", "rb");
    if (!df) { fprintf(stderr, "Cannot open data/test_digit.bin\n"); return 1; }
    float* input = malloc(L0_IN * sizeof(float));
    fread(input, sizeof(float), L0_IN, df);
    fclose(df);


    float* buf0 = malloc(L0_OUT * sizeof(float));
    float* buf1 = malloc(L1_OUT * sizeof(float));
    float* buf2 = malloc(L2_OUT * sizeof(float));
    float* buf3 = malloc(L3_OUT * sizeof(float));


    dense_relu   (input, L0_IN, w0, b0, buf0, L0_OUT);
    dense_relu   (buf0,  L1_IN, w1, b1, buf1, L1_OUT);
    dense_relu   (buf1,  L2_IN, w2, b2, buf2, L2_OUT);
    dense_softmax(buf2,  L3_IN, w3, b3, buf3, L3_OUT);
    int predicted = argmax(buf3, L3_OUT);

    printf("Softmax output:\n");
    for (int i = 0; i < L3_OUT; i++)
        printf("  digit %d: %.4f%s\n", i, buf3[i],
               i == predicted ? "  <-- predicted" : "");
    printf("\nPredicted : %d\n", predicted);

    free(model); free(input);
    free(buf0);  free(buf1); free(buf2); free(buf3);
    return 0;
}