import torch
import triton
import triton.language as tl

def main():

    batch, N = 1, 8
    pw_matrix = torch.ones(batch, N, N, dtype=torch.float32, device=torch.device("cuda"))

    triton_out = reduction(pw_matrix)
    torch_out = reduction_torch(pw_matrix)

    is_correct = torch.allclose(triton_out, torch_out)

    print(pw_matrix)
    print(f"{triton_out=}\n{torch_out=}")
    print(f"{is_correct=}")

@triton.jit
def reduction_kernel(
        output_ptr, stride_output_B, stride_output_N,
        pw_matrix_ptr, stride_pw_matrix_B, stride_pw_matrix_NI, stride_pw_matrix_NJ,
        batch, N,
        BLOCK_B:tl.constexpr, BLOCK_NI:tl.constexpr, BLOCK_NJ:tl.constexpr
    ):
    B = tl.program_id(0) * BLOCK_B + tl.arange(0, BLOCK_B)[:, None, None] # B x 1 x 1
    NI = tl.program_id(1) * BLOCK_NI + tl.arange(0, BLOCK_NI)[None, :, None] # 1 x NI x 1 
    NJ = tl.program_id(2) * BLOCK_NJ + tl.arange(0, BLOCK_NJ)[None, None, :] # 1 x 1 x NJ

    pw_matrix_ptrs = pw_matrix_ptr + (B*stride_pw_matrix_B) + (NI*stride_pw_matrix_NI) + (NJ*stride_pw_matrix_NJ) # B x NI x NJ
    pw_matrix_mask = (B < batch) & (NI<N) & (NJ<N) # B x NI x NJ

    pw_matrix = tl.load(pw_matrix_ptrs, mask=pw_matrix_mask, other=0)
    pw_matrix_sum = tl.sum(pw_matrix, axis=-1, keep_dims=True) + (NJ*0) # B x NI x NJ

    output_ptrs = output_ptr + (B*stride_output_B) + (NI*stride_output_N) + (NJ*0) # B x NI x NJ
    output_mask = pw_matrix_mask & (NJ % BLOCK_NJ == 0) # B x NI x NJ

    tl.atomic_add(output_ptrs, pw_matrix_sum, output_mask)

def reduction(pw_matrix):

    batch, N, _ = pw_matrix.shape
    pw_matrix = pw_matrix.contiguous()
    output = torch.zeros(batch, N, dtype=torch.float32, device=pw_matrix.device).contiguous()

    BLOCK_B = 1
    BLOCK_NI = 4
    BLOCK_NJ = 2

    grid_B = (batch // BLOCK_B) + 1
    grid_NI = (N // BLOCK_NI) + 1
    grid_NJ = (N // BLOCK_NJ) + 1

    grid = (grid_B, grid_NI, grid_NJ)

    reduction_kernel[grid](
        output, output.stride(0), output.stride(1),
        pw_matrix, pw_matrix.stride(0), pw_matrix.stride(1), pw_matrix.stride(2),
        batch, N,
        BLOCK_B, BLOCK_NI, BLOCK_NJ
    )

    return output


def reduction_torch(pw_matrix):
    return pw_matrix.sum(dim=-1)



if __name__ == '__main__':
    main()
