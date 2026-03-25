#include <stdint.h>
#include <string.h>
#include <stdio.h>

#ifdef PICO_BUILD
#include "pico/stdlib.h"
#endif

#include "weights_mnist.h"


#define INPUT_H       32
#define INPUT_W       32
#define INPUT_C       1        
#define NUM_CLASSES   10


#define FP_SHIFT      8
#define FP_ONE        (1 << FP_SHIFT)   /* 256 */
#define FP_ROUND      (1 << (FP_SHIFT - 1))

typedef int16_t  fp_t;   
typedef int32_t  acc_t;  

/* Clamp to [-128, 127] in Q8.8 space */
static inline fp_t fp_clamp(acc_t x) {
    if (x >  32767) return  32767;
    if (x < -32768) return -32768;
    return (fp_t)x;
}

/* Multiply two Q8.8 numbers → Q8.8 */
static inline fp_t fp_mul(fp_t a, fp_t b) {
    return (fp_t)(((acc_t)a * b + FP_ROUND) >> FP_SHIFT);
}


#define L0_OUT_C  16
#define L1_C      16
#define L2_IN_C   16
#define L2_OUT_C  32
#define L3_C      32


static void conv2d(
    const fp_t *in,  int H, int W, int in_c,
          fp_t *out, int OH, int OW, int out_c,
    const fp_t *weights, /* [kH][kW][in_c][out_c] flat */
    const fp_t *bias,
    int kH, int kW, int stride, int pad)
{
    memset(out, 0, (size_t)(OH * OW * out_c) * sizeof(fp_t));

    for (int oh = 0; oh < OH; oh++) {
        for (int ow = 0; ow < OW; ow++) {
            for (int oc = 0; oc < out_c; oc++) {
                acc_t sum = (acc_t)bias[oc] << FP_SHIFT;
                for (int kh = 0; kh < kH; kh++) {
                    int ih = oh * stride - pad + kh;
                    if (ih < 0 || ih >= H) continue;
                    for (int kw = 0; kw < kW; kw++) {
                        int iw = ow * stride - pad + kw;
                        if (iw < 0 || iw >= W) continue;
                        for (int ic = 0; ic < in_c; ic++) {
                            fp_t iv = in[IDX3(ih, iw, ic, W, in_c)];
                            fp_t wv = weights[
                                ((kh * kW + kw) * in_c + ic) * out_c + oc];
                            sum += (acc_t)iv * wv;
                        }
                    }
                }
                out[IDX3(oh, ow, oc, OW, out_c)] =
                    fp_clamp((sum + FP_ROUND) >> FP_SHIFT);
            }
        }
    }
}


static void batchnorm_relu(fp_t *x, int numel_per_c, int C,
                            const fp_t *scale, const fp_t *shift)
{
    for (int i = 0; i < numel_per_c; i++) {
        for (int c = 0; c < C; c++) {
            acc_t v = (acc_t)x[i * C + c];
            v = ((v * scale[c]) >> FP_SHIFT) + shift[c];
            x[i * C + c] = fp_relu(fp_clamp(v));
        }
    }
}

/* Identity BN (no-op) helpers */
static fp_t _ones16[L2_OUT_C];
static fp_t _zeros16[L2_OUT_C];
static void init_bn_identity(void) {
    for (int i = 0; i < L2_OUT_C; i++) {
        _ones16[i]  = FP_ONE;
        _zeros16[i] = 0;
    }
}


static void resblock(fp_t *x, fp_t *tmp,
                     int H, int W, int C,
                     const fp_t *wa, const fp_t *ba,
                     const fp_t *wb, const fp_t *bb)
{
    /* tmp = conv(x, wa) + ba  -> relu */
    conv2d(x, H, W, C, tmp, H, W, C, wa, ba, 3, 3, 1, 1);
    batchnorm_relu(tmp, H * W, C, _ones16, _zeros16);

    /* x_new = conv(tmp, wb) + bb  (no relu yet, add residual first) */
    fp_t *x_new = buf_b; /* reuse second scratch */
    conv2d(tmp, H, W, C, x_new, H, W, C, wb, bb, 3, 3, 1, 1);

    /* add shortcut + relu */
    for (int i = 0; i < H * W * C; i++)
        x[i] = fp_relu(fp_clamp((acc_t)x_new[i] + x[i]));
}

static void resblock_ds(
    fp_t *x,         /* H  x W  x in_c  → overwritten with OH x OW x out_c */
    fp_t *tmp,
    int H, int W, int in_c, int out_c,
    const fp_t *wa,  const fp_t *ba,
    const fp_t *wb,  const fp_t *bb,
    const fp_t *wp,  const fp_t *bp)   /* projection 1x1 */
{
    int OH = H / 2, OW = W / 2;

    /* shortcut = proj(x) with stride 2 */
    fp_t *shortcut = tmp + OH * OW * out_c; /* second half of tmp */
    conv2d(x, H, W, in_c, shortcut, OH, OW, out_c, wp, bp, 1, 1, 2, 0);

    /* main path */
    conv2d(x, H, W, in_c, tmp, OH, OW, out_c, wa, ba, 3, 3, 2, 1);
    batchnorm_relu(tmp, OH * OW, out_c, _ones16, _zeros16);

    fp_t *main2 = buf_b;
    conv2d(tmp, OH, OW, out_c, main2, OH, OW, out_c, wb, bb, 3, 3, 1, 1);

    /* add + relu, write back to x */
    for (int i = 0; i < OH * OW * out_c; i++)
        x[i] = fp_relu(fp_clamp((acc_t)main2[i] + shortcut[i]));
}


static void global_avg_pool(const fp_t *x, int H, int W, int C, fp_t *out)
{
    int n = H * W;
    for (int c = 0; c < C; c++) {
        acc_t sum = 0;
        for (int i = 0; i < n; i++)
            sum += x[i * C + c];
        out[c] = fp_clamp(sum / n);
    }
}


static void fc(const fp_t *in, int in_c,
                     fp_t *out, int out_c,
               const fp_t *W,  const fp_t *b)
{
    for (int o = 0; o < out_c; o++) {
        acc_t sum = (acc_t)b[o] << FP_SHIFT;
        for (int i = 0; i < in_c; i++)
            sum += (acc_t)in[i] * W[i * out_c + o];
        out[o] = fp_clamp((sum + FP_ROUND) >> FP_SHIFT);
    }
}


static int argmax(const fp_t *x, int n) {
    int best = 0;
    for (int i = 1; i < n; i++)
        if (x[i] > x[best]) best = i;
    return best;
}


int resnet_infer(const uint8_t *image_u8)
{
    init_bn_identity();


    for (int i = 0; i < INPUT_H * INPUT_W * INPUT_C; i++)
        buf_a[i] = (fp_t)((int)image_u8[i] - 128) * 2;  /* Q8.8 */

    /* --- Layer 0: conv 3x3 ------------------------------------------- */
    conv2d(buf_a, INPUT_H, INPUT_W, INPUT_C,
           buf_b, INPUT_H, INPUT_W, L0_OUT_C,
           (const fp_t *)w_conv0, b_conv0, 3, 3, 1, 1);
    batchnorm_relu(buf_b, INPUT_H * INPUT_W, L0_OUT_C, _ones16, _zeros16);
    memcpy(buf_a, buf_b, INPUT_H * INPUT_W * L0_OUT_C * sizeof(fp_t));

    /* --- ResBlock 1: 16 -> 16 ---------------------------------------- */
    resblock(buf_a, buf_b,
             INPUT_H, INPUT_W, L1_C,
             (const fp_t *)w_rb1_a, b_rb1_a,
             (const fp_t *)w_rb1_b, b_rb1_b);

    /* --- ResBlock 2: 16 -> 32, stride 2  (16x16 out) ---------------- */
    resblock_ds(buf_a, buf_b,
                INPUT_H, INPUT_W, L2_IN_C, L2_OUT_C,
                (const fp_t *)w_rb2_a,   b_rb2_a,
                (const fp_t *)w_rb2_b,   b_rb2_b,
                (const fp_t *)w_rb2_proj, b_rb2_proj);
    /* buf_a now holds 16x16x32 */

    /* --- ResBlock 3: 32 -> 32 ---------------------------------------- */
    resblock(buf_a, buf_b,
             INPUT_H / 2, INPUT_W / 2, L3_C,
             (const fp_t *)w_rb3_a, b_rb3_a,
             (const fp_t *)w_rb3_b, b_rb3_b);


    fp_t gap[L3_C];
    global_avg_pool(buf_a, INPUT_H / 2, INPUT_W / 2, L3_C, gap);


    fp_t logits[NUM_CLASSES];
    fc(gap, L3_C, logits, NUM_CLASSES, (const fp_t *)w_fc, b_fc);

    return argmax(logits, NUM_CLASSES);
}


static void softmax_display(const fp_t *logits, float *probs, int n)
{

    float fmax = (float)logits[0] / 256.0f;
    for (int i = 1; i < n; i++) {
        float v = (float)logits[i] / 256.0f;
        if (v > fmax) fmax = v;
    }
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        float v = (float)logits[i] / 256.0f - fmax;
        probs[i] = 1.0f; 
        sum += probs[i];
        (void)v;
    }

    for (int i = 0; i < n; i++)
        probs[i] = (float)logits[i] / 256.0f;
}


static void print_results(const fp_t *logits, int predicted)
{
   
    fp_t lmin = logits[0], lmax = logits[0];
    for (int i = 1; i < NUM_CLASSES; i++) {
        if (logits[i] < lmin) lmin = logits[i];
        if (logits[i] > lmax) lmax = logits[i];
    }
    fp_t range = lmax - lmin;
    if (range == 0) range = 1;

#define BAR_WIDTH 20

    printf("\r\n========= ResNet MNIST Result =========\r\n");
    for (int i = 0; i < NUM_CLASSES; i++) {
        int bar_len = (int)(((long)(logits[i] - lmin) * BAR_WIDTH) / range);
        char bar[BAR_WIDTH + 1];
        for (int b = 0; b < BAR_WIDTH; b++)
            bar[b] = (b < bar_len) ? '#' : '-';
        bar[BAR_WIDTH] = '\0';

        /* Raw score as fixed-point decimal: e.g. 512 → "2.00" */
        int integer_part  = logits[i] >> 8;           /* whole number */
        int frac_part     = ((logits[i] & 0xFF) * 100) >> 8; /* 0-99 */
        if (frac_part < 0) frac_part = -frac_part;

        printf("  %d: [%s] %c score=%d.%02d\r\n",
               i, bar,
               (i == predicted) ? '*' : ' ',
               integer_part, frac_part);
    }
    printf("=======================================\r\n");
    printf("  >>> Predicted digit: %d <<<\r\n", predicted);
    printf("=======================================\r\n\r\n");
}


static fp_t g_logits[NUM_CLASSES];   

int resnet_infer_full(const uint8_t *image_u8)
{
    init_bn_identity();

    for (int i = 0; i < INPUT_H * INPUT_W * INPUT_C; i++)
        buf_a[i] = (fp_t)((int)image_u8[i] - 128) * 2;

    conv2d(buf_a, INPUT_H, INPUT_W, INPUT_C,
           buf_b, INPUT_H, INPUT_W, L0_OUT_C,
           (const fp_t *)w_conv0, b_conv0, 3, 3, 1, 1);
    batchnorm_relu(buf_b, INPUT_H * INPUT_W, L0_OUT_C, _ones16, _zeros16);
    memcpy(buf_a, buf_b, INPUT_H * INPUT_W * L0_OUT_C * sizeof(fp_t));

    resblock(buf_a, buf_b, INPUT_H, INPUT_W, L1_C,
             (const fp_t *)w_rb1_a, b_rb1_a,
             (const fp_t *)w_rb1_b, b_rb1_b);

    resblock_ds(buf_a, buf_b, INPUT_H, INPUT_W, L2_IN_C, L2_OUT_C,
                (const fp_t *)w_rb2_a,    b_rb2_a,
                (const fp_t *)w_rb2_b,    b_rb2_b,
                (const fp_t *)w_rb2_proj, b_rb2_proj);

    resblock(buf_a, buf_b, INPUT_H / 2, INPUT_W / 2, L3_C,
             (const fp_t *)w_rb3_a, b_rb3_a,
             (const fp_t *)w_rb3_b, b_rb3_b);

    fp_t gap[L3_C];
    global_avg_pool(buf_a, INPUT_H / 2, INPUT_W / 2, L3_C, gap);
    fc(gap, L3_C, g_logits, NUM_CLASSES, (const fp_t *)w_fc, b_fc);

    return argmax(g_logits, NUM_CLASSES);
}


int main(void)
{
#ifdef PICO_BUILD
    stdio_init_all();

    for (int i = 0; i < 20; i++) {
        if (stdio_usb_connected()) break;
        sleep_ms(100);
    }
#endif

    printf("\r\nResNet MNIST on Pico — ready.\r\n");


    static const uint8_t test_image[INPUT_H * INPUT_W * INPUT_C] = {
        0
    };


    int predicted = resnet_infer_full(test_image);

    print_results(g_logits, predicted);

#ifdef PICO_BUILD
    const uint LED_PIN = 25;
    gpio_init(LED_PIN);
    gpio_set_dir(LED_PIN, GPIO_OUT);
    while (true) {
        gpio_put(LED_PIN, 1); sleep_ms(500);
        gpio_put(LED_PIN, 0); sleep_ms(500);
    }
#else
    return 0;
#endif
}
