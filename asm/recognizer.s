.section __TEXT,__text
.align 2

.global _dense_relu
_dense_relu:
    stp     x29, x30, [sp, #-96]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     x23, x24, [sp, #48]
    stp     x25, x26, [sp, #64]
    stp     x27, x28, [sp, #80]

    mov     x19, x0
    mov     x20, x1
    mov     x21, x2
    mov     x22, x3
    mov     x23, x4
    mov     x24, x5
    mov     x25, #0

.Lrelu_outer:
    cmp     x25, x24
    b.ge    .Lrelu_done

    ldr     s0, [x22, x25, lsl #2]
    mov     x26, #0
.Lrelu_inner:
    cmp     x26, x20
    b.ge    .Lrelu_inner_done
    ldr     s1, [x19, x26, lsl #2]
    mul     x9, x26, x24
    add     x9, x9, x25
    ldr     s2, [x21, x9, lsl #2]
    fmadd   s0, s1, s2, s0
    add     x26, x26, #1
    b       .Lrelu_inner
.Lrelu_inner_done:
    fmov    s1, wzr
    fcmp    s0, s1
    fcsel   s0, s0, s1, gt
    str     s0, [x23, x25, lsl #2]
    add     x25, x25, #1
    b       .Lrelu_outer

.Lrelu_done:
    ldp     x27, x28, [sp, #80]
    ldp     x25, x26, [sp, #64]
    ldp     x23, x24, [sp, #48]
    ldp     x21, x22, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #96
    ret

.global _dense_softmax
_dense_softmax:
    stp     x29, x30, [sp, #-96]!
    mov     x29, sp
    stp     x19, x20, [sp, #16]
    stp     x21, x22, [sp, #32]
    stp     x23, x24, [sp, #48]
    stp     x25, x26, [sp, #64]
    stp     x27, x28, [sp, #80]

    mov     x19, x0
    mov     x20, x1
    mov     x21, x2
    mov     x22, x3
    mov     x23, x4
    mov     x24, x5

    mov     x25, #0
.Lsm_logit_outer:
    cmp     x25, x24
    b.ge    .Lsm_logit_done
    ldr     s0, [x22, x25, lsl #2]
    mov     x26, #0
.Lsm_logit_inner:
    cmp     x26, x20
    b.ge    .Lsm_logit_inner_done
    ldr     s1, [x19, x26, lsl #2]
    mul     x9, x26, x24
    add     x9, x9, x25
    ldr     s2, [x21, x9, lsl #2]
    fmadd   s0, s1, s2, s0
    add     x26, x26, #1
    b       .Lsm_logit_inner
.Lsm_logit_inner_done:
    str     s0, [x23, x25, lsl #2]
    add     x25, x25, #1
    b       .Lsm_logit_outer
.Lsm_logit_done:

    ldr     s8, [x23]
    mov     x25, #1
.Lsm_max:
    cmp     x25, x24
    b.ge    .Lsm_max_done
    ldr     s1, [x23, x25, lsl #2]
    fcmp    s1, s8
    fcsel   s8, s1, s8, gt
    add     x25, x25, #1
    b       .Lsm_max
.Lsm_max_done:

    fmov    s9, wzr
    mov     x25, #0
.Lsm_exp:
    cmp     x25, x24
    b.ge    .Lsm_exp_done
    ldr     s0, [x23, x25, lsl #2]
    fsub    s0, s0, s8
    bl      _expf
    fadd    s9, s9, s0
    str     s0, [x23, x25, lsl #2]
    add     x25, x25, #1
    b       .Lsm_exp
.Lsm_exp_done:

    mov     x25, #0
.Lsm_norm:
    cmp     x25, x24
    b.ge    .Lsm_norm_done
    ldr     s0, [x23, x25, lsl #2]
    fdiv    s0, s0, s9
    str     s0, [x23, x25, lsl #2]
    add     x25, x25, #1
    b       .Lsm_norm
.Lsm_norm_done:

    ldp     x27, x28, [sp, #80]
    ldp     x25, x26, [sp, #64]
    ldp     x23, x24, [sp, #48]
    ldp     x21, x22, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp], #96
    ret

.global _argmax
_argmax:
    ldr     s0, [x0]
    mov     x2, #0
    mov     x3, #1
.Largmax_loop:
    cmp     x3, x1
    b.ge    .Largmax_done
    ldr     s1, [x0, x3, lsl #2]
    fcmp    s1, s0
    b.le    .Largmax_next
    fmov    s0, s1
    mov     x2, x3
.Largmax_next:
    add     x3, x3, #1
    b       .Largmax_loop
.Largmax_done:
    mov     x0, x2
    ret