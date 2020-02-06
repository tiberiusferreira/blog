---
title: "Machine Learning, Rust and SIMD - II"
date: 2020-02-06T15:37:05-03:00
draft: true
---

It's been a while and a lot happened since the last time. 

The plan with this post was to show the results of optimizing Yolo's hot path for the Raspberry Pi 3B+

I did get into Yolo's codebase and did some profiling. Turns out, most (85%+) of the time was spent in a single function:
```C
void cpu_gemm_nn(int TA, int TB, int M, int N, int K, float ALPHA,
        float *A, int lda,
        float *B, int ldb,
        float BETA,
        float *C, int ldc)
{
    int i,j,k;
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            PUT_IN_REGISTER float A_PART = ALPHA * A[i * lda + k];
            for(j = 0; j < N; ++j){
                C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
}
``` 

This looked like a weird named function, but it didn't seem to be doing much. 

>I later discovered it was part of the Basic Linear Algebra Subprograms (BLAS) specification and was an acronym for general matrix multiplication.

Naively I rewrote it in (safe) Rust and got horrible performance, over 2x slower than the C version.
Turns out, in such a tight loop bounds checking has a high price. 

I then disabled bounds checking using unsafe and got back to the same performance as C! 
```Rust 
#[no_mangle]
pub extern "C" fn gemm_nn_rust_unsafe(n: usize, k: usize, alpha: f32,
                                      a: *const f32, lda: usize,
                                      b: *const f32, ldb: usize,
                                      c: *const f32, ldc: usize){
    let size_a = lda + k;
    let a_n;
    unsafe {
        a_n = std::slice::from_raw_parts(a as *const f32, size_a);
    }

    let size_b = k*ldb + n;
    let b_n;
    unsafe {
        b_n = std::slice::from_raw_parts(b as *const f32, size_b);
    }

    let size_c = ldc + n;
    let c_n;
    unsafe {
        c_n = std::slice::from_raw_parts_mut(c as *mut f32, size_c);
    }
    unsafe {
        let i = 0;
        for k_index in 0..k {
            let a_part: f32 = alpha * *a_n.get_unchecked( k_index);
            let mut j = 0;
            while j < n {           // stride a = 1 // stride b = ldb // stride c = 1
                // rows a = 1
                *c_n.get_unchecked_mut( j) += a_part * (*b_n.get_unchecked(k_index * (ldb) + j));
                j = j+1;
            }

        }
    }
}
```

Awesome, we have baseline! Now we can get into SIMD!

Adding SIMD was quite fun and pleasant thanks to the [Packed SIMD](https://rust-lang.github.io/packed_simd/packed_simd/) crate and the awesome Rust community.

```Rust
#[no_mangle]
pub extern "C" fn gemm_nn_rust_simd(n: usize, k: usize, alpha: f32,
                                    a: *const f32, lda: usize,
                                    b: *const f32, ldb: usize,
                                    c: *const f32, ldc: usize)
{
    let size_a = lda + k;
    let a_n;
    unsafe {
        a_n = std::slice::from_raw_parts(a as *const f32, size_a);
    }

    let size_b = k*ldb + n;
    let b_n;
    unsafe {
        b_n = std::slice::from_raw_parts(b as *const f32, size_b);
    }

    let size_c = ldc + n;
    let c_n;
    unsafe {
        c_n = std::slice::from_raw_parts_mut(c as *mut f32, size_c);
    }
    let chunks = 4 as usize;
    let integer = n/chunks;
    unsafe {
        let i = 0;
        for k_index in 0..k {
            let a_part: f32 = alpha * *a_n.get_unchecked(i * (lda) + k_index);
//            let a_part_simd = f32x4::splat(a_part);
            let mut j = 0;
            let c_ind = i * (ldc);
            let b_ind = k_index * (ldb);
            while j < chunks * integer {
                let c_ind_inner = c_ind + j;
                let c_var = f32x4::from_slice_unaligned_unchecked(&c_n[c_ind_inner ..]);
                let b_var = f32x4::from_slice_unaligned_unchecked(&b_n[(b_ind + j) ..]);

                let res = c_var + a_part * b_var;
                res.write_to_slice_unaligned_unchecked(&mut c_n[c_ind_inner ..]);
                j = j + chunks;
            }

            while j < n {
                *c_n.get_unchecked_mut(c_ind + j) += a_part * (*b_n.get_unchecked(b_ind + j));
                j = j+1;
            }

        }
    }

}
```

As expected it did boost performance quite a bit!

Here are the final results:

|                | C ARM     | Rust ARM Neon | C ARM OpenMP | Rust ARM Neon OpenMP |
|----------------|-----------|---------------|--------------|----------------------|
| Real           | 5m36.687s | 2m26.723s     | 1m44.369s    | 1m1.510s             |
| User           | 5m12.121s | 2m13.472s     | 5m26.866s    | 2m40.456s            |
| Sys            | 0m1.680s  | 0m1.630s      | 0m1.541s     | 0m1.469s             |
| Real (s)       | 336.687   | 146.723       | 104.369      | 61.510               |
| Real Speedup % | 100       | 229.47        | 100          | 169.67               |


The plan now was to switch to safe Rust using iterators and maybe replace OpenMP multithreading approach to using the fantastic [Rayon](https://github.com/rayon-rs/rayon)
library.

However, before getting to it [Smart Campus](http://smartcampus.prefeitura.unicamp.br) told me plans had changed and the code would
actually run in the University's servers, instead of the Raspberry Pi 3B+. Since Yolo's code has handwritten x86 AVX assembly for this function it made
little sense to try to beat it and the code itself ran in less than 10s already.

Nevertheless it was quite a fun project and got me wondering what kind of magic Yolo is using to get suc 

   