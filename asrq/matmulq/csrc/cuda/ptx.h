// Inline PTX helpers used by the mma-based GEMM kernel.
//
// All operations target sm_80+ (Ampere) and are forward-compatible with sm_90
// (Hopper), which is the eventual target for the wgmma upgrade.
#pragma once

#include <cuda_fp16.h>
#include <cstdint>

namespace asrq_ptx {

// ---------------------------------------------------------------------------
// Address-space conversion: generic pointer (in shared mem) -> 32-bit shared
// memory offset, as required by ldmatrix / cp.async / mma instructions.
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint32_t cvta_to_shared(const void* ptr) {
    uint32_t addr;
    asm("{\n"
        "  .reg .u64 u;\n"
        "  cvta.to.shared.u64 u, %1;\n"
        "  cvt.u32.u64 %0, u;\n"
        "}\n"
        : "=r"(addr)
        : "l"(ptr));
    return addr;
}

// ---------------------------------------------------------------------------
// ldmatrix: warp-collective load of four 8x8 matrices of .b16 from shared
// memory into 4 registers per thread.  Each thread provides ONE pointer; the
// 32 threads' pointers describe the rows of the 4 fragments (8 rows per
// fragment, fragments laid out sequentially across lane / 8).
// ---------------------------------------------------------------------------
__device__ __forceinline__ void ldmatrix_x4(uint32_t (&d)[4], uint32_t smem_ptr) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(smem_ptr));
}

// ldmatrix with per-fragment transpose: used to deliver B operands of an
// mma.row.col when B is stored row-major in shared memory.
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t (&d)[4], uint32_t smem_ptr) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
        "{%0, %1, %2, %3}, [%4];\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(smem_ptr));
}

// ---------------------------------------------------------------------------
// mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
//   D[16x8] = A[16x16] * B[16x8] + C[16x8]
// A operand: 4 .b16x2 regs/thread (== 4 uint32)
// B operand: 2 .b16x2 regs/thread (== 2 uint32)
// C/D operand: 4 .f32 regs/thread
// ---------------------------------------------------------------------------
__device__ __forceinline__ void mma_m16n8k16_f32_f16(
    float (&d)[4],
    const uint32_t (&a)[4],
    const uint32_t (&b)[2],
    const float (&c)[4]
) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

// ---------------------------------------------------------------------------
// Vectorized 16-byte (== 8 fp16) shared-memory store via .ca cached PTX load
// from global, written through a register pair.  Implemented as plain ld/st;
// cp.async is available in a separate helper below.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void ld_global_v4_u32(uint32_t (&d)[4], const void* gmem_ptr) {
    asm volatile(
        "ld.global.v4.u32 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "l"(gmem_ptr));
}

__device__ __forceinline__ void st_shared_v4_u32(uint32_t smem_ptr, const uint32_t (&v)[4]) {
    asm volatile(
        "st.shared.v4.u32 [%0], {%1, %2, %3, %4};\n"
        :
        : "r"(smem_ptr), "r"(v[0]), "r"(v[1]), "r"(v[2]), "r"(v[3]));
}

// ---------------------------------------------------------------------------
// cp.async: asynchronous global -> shared copy (sm_80+).  Only the 16-byte
// .cg ("cache global", bypasses L1) variant is needed by the GEMM.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void cp_async_16(uint32_t smem_ptr, const void* gmem_ptr) {
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :
        : "r"(smem_ptr), "l"(gmem_ptr));
}

// Commit all previously-issued cp.async ops into a new group.
__device__ __forceinline__ void cp_async_commit_group() {
    asm volatile("cp.async.commit_group;\n");
}

// Wait until at most N most-recent groups are still in flight.
template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

// Wait until all cp.async ops have completed.
__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;\n");
}

// ---------------------------------------------------------------------------
// Block-wide barrier (== __syncthreads, but expressed via PTX).
// ---------------------------------------------------------------------------
__device__ __forceinline__ void cta_sync() {
    asm volatile("bar.sync 0;\n" ::: "memory");
}

// Warp-wide barrier (all 32 lanes converge).
__device__ __forceinline__ void warp_sync() {
    asm volatile("bar.warp.sync 0xffffffff;\n" ::: "memory");
}

// ===========================================================================
// sm_90 (Hopper) primitives: mbarrier, TMA, wgmma
// (helpers always declared; inline-asm bodies only assemble on sm_90+)
// ===========================================================================

// ---------------------------------------------------------------------------
// mbarrier (shared-memory async barrier) used for TMA completion.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void mbarrier_init(uint64_t* bar, uint32_t count) {
    uint32_t bar_ptr = cvta_to_shared(bar);
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n"
                 :: "r"(bar_ptr), "r"(count));
}

// One thread arrives and pre-declares the expected TMA byte count.
__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t* bar, uint32_t tx_count) {
    uint32_t bar_ptr = cvta_to_shared(bar);
    asm volatile(
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
        :: "r"(bar_ptr), "r"(tx_count)
    );
}

// Plain arrive (no transaction count) — used by consumers to release `empty` barrier.
__device__ __forceinline__ void mbarrier_arrive(uint64_t* bar) {
    uint32_t bar_ptr = cvta_to_shared(bar);
    asm volatile(
        "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];\n"
        :: "r"(bar_ptr)
        : "memory"
    );
}

// Block until barrier flips parity. Spin-loop on try_wait.parity.
__device__ __forceinline__ void mbarrier_wait(uint64_t* bar, uint32_t phase) {
    uint32_t bar_ptr = cvta_to_shared(bar);
    asm volatile(
        "{\n"
        ".reg .pred P1;\n"
        "LAB_WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
        "@P1 bra DONE_%=;\n"
        "bra LAB_WAIT_%=;\n"
        "DONE_%=:\n"
        "}\n"
        :: "r"(bar_ptr), "r"(phase)
    );
}

// ---------------------------------------------------------------------------
// TMA 2D async load (cp.async.bulk.tensor.2d).  `tma_desc` is a pointer
// (constant param) to a CUtensorMap; (cx, cy) are the 2D coordinates of the
// box's top-left corner in the global tensor.  Completion signals `mbar`.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void tma_load_2d(
    void* dst_smem, const void* tma_desc,
    int cx, int cy, uint64_t* mbar)
{
    uint32_t dst_ptr = cvta_to_shared(dst_smem);
    uint32_t bar_ptr = cvta_to_shared(mbar);
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];\n"
        :: "r"(dst_ptr), "l"(tma_desc), "r"(cx), "r"(cy), "r"(bar_ptr)
        : "memory"
    );
}

// TMA 3D async load — used with the canonical 128B-swizzled smem layout where
// the global tensor is viewed as shape (k_inner=64, H, W/64).
__device__ __forceinline__ void tma_load_3d(
    void* dst_smem, const void* tma_desc,
    int c0, int c1, int c2, uint64_t* mbar)
{
    uint32_t dst_ptr = cvta_to_shared(dst_smem);
    uint32_t bar_ptr = cvta_to_shared(mbar);
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4}], [%5];\n"
        :: "r"(dst_ptr), "l"(tma_desc), "r"(c0), "r"(c1), "r"(c2), "r"(bar_ptr)
        : "memory"
    );
}

// Multicast variant: data is loaded once from global and broadcast to all CTAs
// in the cluster whose bit is set in `cta_mask`.  Each receiving CTA's mbarrier
// (at the same smem offset as `mbar`) is decremented by the bytes it received.
__device__ __forceinline__ void tma_load_3d_multicast(
    void* dst_smem, const void* tma_desc,
    int c0, int c1, int c2, uint64_t* mbar, uint16_t cta_mask)
{
    uint32_t dst_ptr = cvta_to_shared(dst_smem);
    uint32_t bar_ptr = cvta_to_shared(mbar);
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.tile."
        "mbarrier::complete_tx::bytes.multicast::cluster"
        " [%0], [%1, {%2, %3, %4}], [%5], %6;\n"
        :: "r"(dst_ptr), "l"(tma_desc),
           "r"(c0), "r"(c1), "r"(c2), "r"(bar_ptr), "h"(cta_mask)
        : "memory"
    );
}

// Cluster-shared mbarrier arrive on a remote CTA's mbarrier.  `cta_mbar` must
// be a generic shared-cluster address pointing to the mbarrier in some CTA in
// the cluster.
__device__ __forceinline__ void mbarrier_arrive_cluster(uint32_t cta_mbar) {
    asm volatile(
        "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];\n"
        :: "r"(cta_mbar) : "memory");
}

// Cluster barrier (arrive + wait). One per cluster per call. Uses the
// .aligned variant which must be called by ALL threads in the cluster with
// the same arguments — typically used once at init after mbarrier setup.
__device__ __forceinline__ void cluster_arrive() {
    asm volatile("barrier.cluster.arrive.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void cluster_wait() {
    asm volatile("barrier.cluster.wait.aligned;\n" ::: "memory");
}

// Set expect_tx on a remote (cluster-shared) mbarrier without arriving.
// Used by the producer of one CTA to declare bytes that will be received
// (via multicast) by a peer CTA's mbarrier.
__device__ __forceinline__ void mbarrier_expect_tx_cluster(
    uint32_t cta_mbar, uint32_t bytes)
{
    asm volatile(
        "mbarrier.expect_tx.relaxed.cluster.shared::cluster.b64 [%0], %1;\n"
        :: "r"(cta_mbar), "r"(bytes) : "memory");
}

// Remote arrive + expect_tx in one op (release semantics) on a peer CTA's
// mbarrier in the same cluster.  This both decrements pending count by 1
// AND adds `bytes` to the tx counter.
__device__ __forceinline__ void mbarrier_arrive_expect_tx_cluster(
    uint32_t cta_mbar, uint32_t bytes)
{
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cluster.shared::cluster.b64 "
        "_, [%0], %1;\n"
        :: "r"(cta_mbar), "r"(bytes) : "memory");
}

__device__ __forceinline__ uint32_t cluster_ctarank() {
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(r));
    return r;
}

// Map a local smem address to the equivalent address in another CTA's smem
// within the same cluster.  `dst_cta` is the rank in the cluster.
__device__ __forceinline__ uint32_t smem_to_cluster_smem(
    uint32_t local_smem, uint32_t dst_cta)
{
    uint32_t r;
    asm volatile(
        "mapa.shared::cluster.u32 %0, %1, %2;\n"
        : "=r"(r) : "r"(local_smem), "r"(dst_cta));
    return r;
}

// Per-warpgroup register repartitioning: producer drops to a low count to free
// regs for the consumer warpgroups (which inc).
template <uint32_t RegCount>
__device__ __forceinline__ void setmaxnreg_inc() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" :: "n"(RegCount));
}
template <uint32_t RegCount>
__device__ __forceinline__ void setmaxnreg_dec() {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" :: "n"(RegCount));
}

// Fence after which prior shared-memory writes (TMA arrivals) are visible
// to subsequent wgmma operations on this warp group.
__device__ __forceinline__ void fence_proxy_async_shared_cta() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

// ---------------------------------------------------------------------------
// wgmma synchronization
// ---------------------------------------------------------------------------
__device__ __forceinline__ void wgmma_fence() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void wgmma_commit_group() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}
template <int N>
__device__ __forceinline__ void wgmma_wait_group() {
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N) : "memory");
}

// ---------------------------------------------------------------------------
// Shared-memory matrix descriptor for wgmma operands.
//
// 64-bit layout (CUTLASS-compatible):
//   bits  0..13 : start_address >> 4        (matrix base addr / 16)
//   bits 16..29 : leading_byte_offset >> 4  (LBO / 16)
//   bits 32..45 : stride_byte_offset >> 4   (SBO / 16)
//   bits 49..51 : base offset (3 bits, used with 128B swizzle; we use 0)
//   bits 62..63 : layout_type / swizzle mode
//                 0=no swizzle, 1=128B, 2=64B, 3=32B
//
// For NO-swizzle row-major fp16 with row_stride_bytes = S:
//   LBO = 16 bytes  (next 8-element core matrix along the contiguous dim)
//   SBO = 8 * S     (next 8-row block along the strided dim)
// For 128B-swizzle K-major (canonical Hopper fast path):
//   LBO = 16 bytes, SBO = 1024 bytes (per fast.cu / CUTLASS).
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint64_t make_smem_desc(
    uint32_t smem_addr,
    uint32_t lbo_bytes,
    uint32_t sbo_bytes,
    int swizzle_mode)
{
    uint64_t desc = 0;
    desc |= (uint64_t)((smem_addr >> 4) & 0x3FFFu);
    desc |= ((uint64_t)((lbo_bytes >> 4) & 0x3FFFu)) << 16;
    desc |= ((uint64_t)((sbo_bytes >> 4) & 0x3FFFu)) << 32;
    desc |= ((uint64_t)(swizzle_mode & 0x3)) << 62;
    return desc;
}

// Convenience: build a descriptor for a 128B-swizzled K-major tile rooted at
// the given smem pointer (e.g. the start of a wgmma operand sub-tile).
__device__ __forceinline__ uint64_t make_smem_desc_128b(const void* smem_ptr) {
    return make_smem_desc(cvta_to_shared(smem_ptr), 16u, 1024u, 1);
}

// ---------------------------------------------------------------------------
// wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16
//   Warp-group (128 thread) async MMA: D[64x128] = A[64x16] * B[16x128] + D
//   A and B both come from shared memory (descriptors).
//   trans_a / trans_b: 0 or 1 (template params -> immediates in asm).
//   Per-thread accumulator: 64 f32 values.
// ---------------------------------------------------------------------------
template <int ScaleD, int TransA, int TransB>
__device__ __forceinline__ void wgmma_m64n128k16_f32_f16_ss(
    float (&d)[64], uint64_t desc_a, uint64_t desc_b)
{
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %66, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        " %16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
        " %32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,"
        " %48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, "
        "%64, %65, p, 1, 1, %67, %68;\n"
        "}\n"
        : "+f"(d[ 0]), "+f"(d[ 1]), "+f"(d[ 2]), "+f"(d[ 3]),
          "+f"(d[ 4]), "+f"(d[ 5]), "+f"(d[ 6]), "+f"(d[ 7]),
          "+f"(d[ 8]), "+f"(d[ 9]), "+f"(d[10]), "+f"(d[11]),
          "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
          "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]),
          "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
          "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),
          "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]),
          "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]),
          "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]),
          "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]),
          "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]),
          "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]),
          "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]),
          "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]),
          "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
        : "l"(desc_a), "l"(desc_b),
          "r"(int(ScaleD)), "n"(TransA), "n"(TransB)
    );
}

// ---------------------------------------------------------------------------
// wgmma.mma_async.sync.aligned.m64n256k16.f32.f16.f16
//   D[64x256] = A[64x16] * B[16x256] + D.  Per-thread accumulator: 128 f32.
// ---------------------------------------------------------------------------
template <int ScaleD, int TransA, int TransB>
__device__ __forceinline__ void wgmma_m64n256k16_f32_f16_ss(
    float (&d)[128], uint64_t desc_a, uint64_t desc_b)
{
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %130, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n256k16.f32.f16.f16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        " %16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
        " %32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,"
        " %48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,"
        " %64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,"
        " %80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,"
        " %96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,"
        " %108,%109,%110,%111,%112,%113,%114,%115,%116,%117,%118,%119,"
        " %120,%121,%122,%123,%124,%125,%126,%127}, "
        "%128, %129, p, 1, 1, %131, %132;\n"
        "}\n"
        : "+f"(d[  0]), "+f"(d[  1]), "+f"(d[  2]), "+f"(d[  3]),
          "+f"(d[  4]), "+f"(d[  5]), "+f"(d[  6]), "+f"(d[  7]),
          "+f"(d[  8]), "+f"(d[  9]), "+f"(d[ 10]), "+f"(d[ 11]),
          "+f"(d[ 12]), "+f"(d[ 13]), "+f"(d[ 14]), "+f"(d[ 15]),
          "+f"(d[ 16]), "+f"(d[ 17]), "+f"(d[ 18]), "+f"(d[ 19]),
          "+f"(d[ 20]), "+f"(d[ 21]), "+f"(d[ 22]), "+f"(d[ 23]),
          "+f"(d[ 24]), "+f"(d[ 25]), "+f"(d[ 26]), "+f"(d[ 27]),
          "+f"(d[ 28]), "+f"(d[ 29]), "+f"(d[ 30]), "+f"(d[ 31]),
          "+f"(d[ 32]), "+f"(d[ 33]), "+f"(d[ 34]), "+f"(d[ 35]),
          "+f"(d[ 36]), "+f"(d[ 37]), "+f"(d[ 38]), "+f"(d[ 39]),
          "+f"(d[ 40]), "+f"(d[ 41]), "+f"(d[ 42]), "+f"(d[ 43]),
          "+f"(d[ 44]), "+f"(d[ 45]), "+f"(d[ 46]), "+f"(d[ 47]),
          "+f"(d[ 48]), "+f"(d[ 49]), "+f"(d[ 50]), "+f"(d[ 51]),
          "+f"(d[ 52]), "+f"(d[ 53]), "+f"(d[ 54]), "+f"(d[ 55]),
          "+f"(d[ 56]), "+f"(d[ 57]), "+f"(d[ 58]), "+f"(d[ 59]),
          "+f"(d[ 60]), "+f"(d[ 61]), "+f"(d[ 62]), "+f"(d[ 63]),
          "+f"(d[ 64]), "+f"(d[ 65]), "+f"(d[ 66]), "+f"(d[ 67]),
          "+f"(d[ 68]), "+f"(d[ 69]), "+f"(d[ 70]), "+f"(d[ 71]),
          "+f"(d[ 72]), "+f"(d[ 73]), "+f"(d[ 74]), "+f"(d[ 75]),
          "+f"(d[ 76]), "+f"(d[ 77]), "+f"(d[ 78]), "+f"(d[ 79]),
          "+f"(d[ 80]), "+f"(d[ 81]), "+f"(d[ 82]), "+f"(d[ 83]),
          "+f"(d[ 84]), "+f"(d[ 85]), "+f"(d[ 86]), "+f"(d[ 87]),
          "+f"(d[ 88]), "+f"(d[ 89]), "+f"(d[ 90]), "+f"(d[ 91]),
          "+f"(d[ 92]), "+f"(d[ 93]), "+f"(d[ 94]), "+f"(d[ 95]),
          "+f"(d[ 96]), "+f"(d[ 97]), "+f"(d[ 98]), "+f"(d[ 99]),
          "+f"(d[100]), "+f"(d[101]), "+f"(d[102]), "+f"(d[103]),
          "+f"(d[104]), "+f"(d[105]), "+f"(d[106]), "+f"(d[107]),
          "+f"(d[108]), "+f"(d[109]), "+f"(d[110]), "+f"(d[111]),
          "+f"(d[112]), "+f"(d[113]), "+f"(d[114]), "+f"(d[115]),
          "+f"(d[116]), "+f"(d[117]), "+f"(d[118]), "+f"(d[119]),
          "+f"(d[120]), "+f"(d[121]), "+f"(d[122]), "+f"(d[123]),
          "+f"(d[124]), "+f"(d[125]), "+f"(d[126]), "+f"(d[127])
        : "l"(desc_a), "l"(desc_b),
          "r"(int(ScaleD)), "n"(TransA), "n"(TransB)
    );
}

} // namespace asrq_ptx
